"""Matrix-valued (d_k x d_v) generalization of the scalar-state derivation in
gated_lasp_verify.py, matching a real GLA / Mamba2-SSD chunked state-space recurrence:

    M_s = A_s * M_{s-1} + k_s (x) v_s        (outer product update, M is d_k x d_v)

`a` (the decay) may be:
  - shape (N,)      -- scalar decay per step, shared across the whole state matrix.
                        This is exactly Mamba2/SSD's restriction (A = scalar * I), the
                        structural constraint that makes "state space duality" hold.
  - shape (N, d_k)   -- per-row (per key-channel) diagonal decay gate, applied as
                        diag(a_s) @ M_{s-1}. This is GLA's actual gate structure
                        (Yang et al. 2023). Because diagonal matrices commute, the
                        cumulative decay for each row is still an independent scalar
                        chain -- so the log-space trick from the toy model applies
                        per-row (vectorized over d_k), with no new math required.

All reconstruction variants (flat log-space, grouped/hierarchical rebasing, naive
materialize-then-divide) are implemented once, parametrized over this (scalar | vector)
decay shape, so the same code path covers both Mamba2-SSD and GLA-style gating.
"""
import torch


def _is_vector_decay(a: torch.Tensor) -> bool:
    return a.dim() == 2


def sequential_reference(a: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          dtype=torch.float64) -> torch.Tensor:
    """Ground truth: run the recurrence one step at a time, no chunking at all."""
    a, k, v = a.to(dtype), k.to(dtype), v.to(dtype)
    N, d_k = k.shape
    d_v = v.shape[1]
    vector_decay = _is_vector_decay(a)
    M = torch.zeros(d_k, d_v, dtype=dtype)
    outs = torch.zeros(N, d_k, d_v, dtype=dtype)
    for s in range(N):
        decay = a[s].unsqueeze(-1) if vector_decay else a[s]
        M = decay * M + torch.outer(k[s], v[s])
        outs[s] = M
    return outs


def _chunk_local(a_c: torch.Tensor, k_c: torch.Tensor, v_c: torch.Tensor, dtype):
    """Purely local per-chunk quantities: final local state S_hat (ignoring incoming
    state), total chunk decay Gamma, and the local-only running sequence."""
    C, d_k = k_c.shape
    d_v = v_c.shape[1]
    vector_decay = _is_vector_decay(a_c)
    M = torch.zeros(d_k, d_v, dtype=dtype)
    local_seq = torch.zeros(C, d_k, d_v, dtype=dtype)
    for j in range(C):
        decay = a_c[j].unsqueeze(-1) if vector_decay else a_c[j]
        M = decay * M + torch.outer(k_c[j], v_c[j])
        local_seq[j] = M
    S_hat = M
    Gamma = a_c.prod(dim=0) if vector_decay else a_c.prod()
    return S_hat, Gamma, local_seq


def _chunk_decompose(a: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_chunks: int,
                      dtype):
    N, d_k = k.shape
    C = N // num_chunks
    assert N % num_chunks == 0
    S_hat_list, Gamma_list, local_seqs = [], [], []
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        S_hat, Gamma, local_seq = _chunk_local(a[sl], k[sl], v[sl], dtype)
        S_hat_list.append(S_hat)
        Gamma_list.append(Gamma)
        local_seqs.append(local_seq)
    return torch.stack(S_hat_list), torch.stack(Gamma_list), local_seqs, C


def flat_logspace_reconstruction(a: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                  num_chunks: int, dtype=torch.float64,
                                  logspace_dtype=None) -> torch.Tensor:
    """The candidate scheme: single all-gather's worth of (S_hat_t, Gamma_t) pairs,
    reconstructed via log-space difference (never materializing the raw cumulative
    product). `logspace_dtype` lets the log/exp accumulation run in higher precision
    than the storage dtype, mirroring how real kernels do softmax-style reductions in
    fp32 even when everything else is bf16.
    """
    a, k, v = a.to(dtype), k.to(dtype), v.to(dtype)
    N, d_k = k.shape
    d_v = v.shape[1]
    vector_decay = _is_vector_decay(a)
    ls_dtype = logspace_dtype or dtype

    S_hat, Gamma, local_seqs, C = _chunk_decompose(a, k, v, num_chunks, dtype)
    logGamma = torch.log(Gamma.to(ls_dtype))
    zero_shape = (1, d_k) if vector_decay else (1,)
    cum = torch.cat([torch.zeros(zero_shape, dtype=ls_dtype), torch.cumsum(logGamma, dim=0)], dim=0)

    M_incoming = torch.zeros(num_chunks, d_k, d_v, dtype=dtype)
    for t in range(1, num_chunks):
        if vector_decay:
            w = torch.exp(cum[t].unsqueeze(0) - cum[1:t + 1]).to(dtype)  # (t, d_k)
            M_incoming[t] = (w.unsqueeze(-1) * S_hat[:t]).sum(dim=0)
        else:
            w = torch.exp(cum[t] - cum[1:t + 1]).to(dtype)  # (t,)
            M_incoming[t] = (w.view(-1, 1, 1) * S_hat[:t]).sum(dim=0)

    outs = torch.zeros(N, d_k, d_v, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        a_c = a[sl]
        if vector_decay:
            decay_to_j = torch.exp(torch.cumsum(torch.log(a_c.to(ls_dtype)), dim=0)).to(dtype)  # (C, d_k)
            outs[sl] = local_seqs[t] + decay_to_j.unsqueeze(-1) * M_incoming[t].unsqueeze(0)
        else:
            decay_to_j = torch.cumprod(a_c, dim=0)  # (C,)
            outs[sl] = local_seqs[t] + decay_to_j.view(-1, 1, 1) * M_incoming[t].unsqueeze(0)
    return outs


def grouped_rebase_reconstruction(a: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   num_chunks: int, rebase_group: int = 8,
                                   dtype=torch.float64, logspace_dtype=None) -> torch.Tensor:
    """Hierarchical/grouped variant: combine in groups of `rebase_group` chunks first
    (bounded exponent range), carrying only a group-level running state between groups
    -- one hop per group instead of per chunk."""
    a, k, v = a.to(dtype), k.to(dtype), v.to(dtype)
    N, d_k = k.shape
    d_v = v.shape[1]
    vector_decay = _is_vector_decay(a)
    ls_dtype = logspace_dtype or dtype

    S_hat, Gamma, local_seqs, C = _chunk_decompose(a, k, v, num_chunks, dtype)
    logGamma = torch.log(Gamma.to(ls_dtype))

    M_incoming = torch.zeros(num_chunks, d_k, d_v, dtype=dtype)
    state_shape = (d_k, d_v)
    running_state = torch.zeros(state_shape, dtype=dtype)

    for g_start in range(0, num_chunks, rebase_group):
        g_end = min(g_start + rebase_group, num_chunks)
        local_logGamma = logGamma[g_start:g_end]
        zero_shape = (1, d_k) if vector_decay else (1,)
        local_cum = torch.cat([torch.zeros(zero_shape, dtype=ls_dtype),
                                torch.cumsum(local_logGamma, dim=0)], dim=0)
        for i, t in enumerate(range(g_start, g_end)):
            if vector_decay:
                decay_from_group_start = torch.exp(local_cum[i]).to(dtype)  # (d_k,)
                incoming = decay_from_group_start.unsqueeze(-1) * running_state
                if i > 0:
                    w = torch.exp(local_cum[i].unsqueeze(0) - local_cum[1:i + 1]).to(dtype)  # (i, d_k)
                    incoming = incoming + (w.unsqueeze(-1) * S_hat[g_start:g_start + i]).sum(dim=0)
            else:
                decay_from_group_start = torch.exp(local_cum[i]).to(dtype)  # scalar
                incoming = decay_from_group_start * running_state
                if i > 0:
                    w = torch.exp(local_cum[i] - local_cum[1:i + 1]).to(dtype)  # (i,)
                    incoming = incoming + (w.view(-1, 1, 1) * S_hat[g_start:g_start + i]).sum(dim=0)
            M_incoming[t] = incoming

        group_total = local_cum[-1]
        n = g_end - g_start
        if vector_decay:
            group_decay = torch.exp(group_total).to(dtype)  # (d_k,)
            w_group = torch.exp(group_total.unsqueeze(0) - local_cum[1:n + 1]).to(dtype)  # (n, d_k)
            running_state = group_decay.unsqueeze(-1) * running_state + \
                (w_group.unsqueeze(-1) * S_hat[g_start:g_end]).sum(dim=0)
        else:
            group_decay = torch.exp(group_total).to(dtype)  # scalar
            w_group = torch.exp(group_total - local_cum[1:n + 1]).to(dtype)  # (n,)
            running_state = group_decay * running_state + \
                (w_group.view(-1, 1, 1) * S_hat[g_start:g_end]).sum(dim=0)

    outs = torch.zeros(N, d_k, d_v, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        a_c = a[sl]
        if vector_decay:
            decay_to_j = torch.exp(torch.cumsum(torch.log(a_c.to(ls_dtype)), dim=0)).to(dtype)
            outs[sl] = local_seqs[t] + decay_to_j.unsqueeze(-1) * M_incoming[t].unsqueeze(0)
        else:
            decay_to_j = torch.cumprod(a_c, dim=0)
            outs[sl] = local_seqs[t] + decay_to_j.view(-1, 1, 1) * M_incoming[t].unsqueeze(0)
    return outs


def naive_materialized_reconstruction(a: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                       num_chunks: int, dtype=torch.float32):
    """The failure mode from Test 4 of gated_lasp_verify.py, ported to matrix state:
    materialize the global cumulative product Gamma_global as an array and reconstruct
    weights by division, at low precision. Returns (outs, Gamma_global) so callers can
    check how many entries underflowed to exactly 0."""
    a, k, v = a.to(dtype), k.to(dtype), v.to(dtype)
    N, d_k = k.shape
    d_v = v.shape[1]
    vector_decay = _is_vector_decay(a)

    S_hat, Gamma, local_seqs, C = _chunk_decompose(a, k, v, num_chunks, dtype)

    one_shape = (1, d_k) if vector_decay else (1,)
    Gamma_global = torch.cat([torch.ones(one_shape, dtype=dtype), torch.cumprod(Gamma, dim=0)], dim=0)

    M_incoming = torch.zeros(num_chunks, d_k, d_v, dtype=dtype)
    for t in range(1, num_chunks):
        if vector_decay:
            weights = Gamma_global[t].unsqueeze(0) / Gamma_global[1:t + 1]  # (t, d_k)
            M_incoming[t] = (weights.unsqueeze(-1) * S_hat[:t]).sum(dim=0)
        else:
            weights = Gamma_global[t] / Gamma_global[1:t + 1]  # (t,)
            M_incoming[t] = (weights.view(-1, 1, 1) * S_hat[:t]).sum(dim=0)

    outs = torch.zeros(N, d_k, d_v, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        a_c = a[sl]
        if vector_decay:
            decay_to_j = torch.cumprod(a_c, dim=0)
            outs[sl] = local_seqs[t] + decay_to_j.unsqueeze(-1) * M_incoming[t].unsqueeze(0)
        else:
            decay_to_j = torch.cumprod(a_c, dim=0)
            outs[sl] = local_seqs[t] + decay_to_j.view(-1, 1, 1) * M_incoming[t].unsqueeze(0)
    return outs, Gamma_global
