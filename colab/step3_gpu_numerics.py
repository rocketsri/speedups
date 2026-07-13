"""Step 3: reproduce the numerical failure mode on real GPU hardware (bf16/fp32), matrix-
valued state, at varying chunk counts and decay ranges -- and confirm the log-space fix
(and, if needed, grouped/hierarchical rebasing) holds there too.

This is single-process / single-GPU -- no torch.distributed needed, since the question here
is purely about numerics (does exp(cumulative log-decay) survive in bf16/fp32 on a real GPU
kernel), not about the communication pattern. Runs fine on a single Colab L4.

COLAB USAGE:
    !nvidia-smi
    !python step3_gpu_numerics.py
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    print("WARNING: no CUDA device visible -- running on CPU. This script is meant to be "
          "run on a GPU runtime (Colab: Runtime > Change runtime type > GPU, e.g. L4).")


def sequential_reference(a, k, v, dtype=torch.float64):
    a, k, v = a.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype)
    N, d_k = k.shape
    d_v = v.shape[1]
    vector_decay = a.dim() == 2
    M = torch.zeros(d_k, d_v, device=DEVICE, dtype=dtype)
    outs = torch.zeros(N, d_k, d_v, device=DEVICE, dtype=dtype)
    for s in range(N):
        decay = a[s].unsqueeze(-1) if vector_decay else a[s]
        M = decay * M + torch.outer(k[s], v[s])
        outs[s] = M
    return outs


def chunk_local(a_c, k_c, v_c, dtype):
    C, d_k = k_c.shape
    d_v = v_c.shape[1]
    vector_decay = a_c.dim() == 2
    M = torch.zeros(d_k, d_v, device=DEVICE, dtype=dtype)
    local_seq = torch.zeros(C, d_k, d_v, device=DEVICE, dtype=dtype)
    for j in range(C):
        decay = a_c[j].unsqueeze(-1) if vector_decay else a_c[j]
        M = decay * M + torch.outer(k_c[j], v_c[j])
        local_seq[j] = M
    Gamma = a_c.prod(dim=0) if vector_decay else a_c.prod()
    return M, Gamma, local_seq


def chunk_decompose(a, k, v, num_chunks, dtype):
    N, d_k = k.shape
    C = N // num_chunks
    S_hat_list, Gamma_list, local_seqs = [], [], []
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        S_hat, Gamma, local_seq = chunk_local(a[sl], k[sl], v[sl], dtype)
        S_hat_list.append(S_hat)
        Gamma_list.append(Gamma)
        local_seqs.append(local_seq)
    return torch.stack(S_hat_list), torch.stack(Gamma_list), local_seqs, C


def flat_logspace_reconstruction(a, k, v, num_chunks, dtype, logspace_dtype=torch.float32):
    a, k, v = a.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype)
    N, d_k = k.shape
    vector_decay = a.dim() == 2
    S_hat, Gamma, local_seqs, C = chunk_decompose(a, k, v, num_chunks, dtype)
    logGamma = torch.log(Gamma.to(logspace_dtype))
    zero_shape = (1, d_k) if vector_decay else (1,)
    cum = torch.cat([torch.zeros(zero_shape, device=DEVICE, dtype=logspace_dtype),
                      torch.cumsum(logGamma, dim=0)], dim=0)

    M_incoming = torch.zeros(num_chunks, *S_hat.shape[1:], device=DEVICE, dtype=dtype)
    for t in range(1, num_chunks):
        if vector_decay:
            w = torch.exp(cum[t].unsqueeze(0) - cum[1:t + 1]).to(dtype)
            M_incoming[t] = (w.unsqueeze(-1) * S_hat[:t]).sum(dim=0)
        else:
            w = torch.exp(cum[t] - cum[1:t + 1]).to(dtype)
            M_incoming[t] = (w.view(-1, 1, 1) * S_hat[:t]).sum(dim=0)

    outs = torch.zeros(N, *S_hat.shape[1:], device=DEVICE, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        a_c = a[sl]
        if vector_decay:
            decay_to_j = torch.exp(torch.cumsum(torch.log(a_c.to(logspace_dtype)), dim=0)).to(dtype)
            outs[sl] = local_seqs[t] + decay_to_j.unsqueeze(-1) * M_incoming[t].unsqueeze(0)
        else:
            decay_to_j = torch.cumprod(a_c, dim=0)
            outs[sl] = local_seqs[t] + decay_to_j.view(-1, 1, 1) * M_incoming[t].unsqueeze(0)
    return outs


def grouped_rebase_reconstruction(a, k, v, num_chunks, rebase_group, dtype,
                                   logspace_dtype=torch.float32):
    a, k, v = a.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype)
    N, d_k = k.shape
    d_v = v.shape[1]
    vector_decay = a.dim() == 2
    S_hat, Gamma, local_seqs, C = chunk_decompose(a, k, v, num_chunks, dtype)
    logGamma = torch.log(Gamma.to(logspace_dtype))

    M_incoming = torch.zeros(num_chunks, d_k, d_v, device=DEVICE, dtype=dtype)
    running_state = torch.zeros(d_k, d_v, device=DEVICE, dtype=dtype)

    for g_start in range(0, num_chunks, rebase_group):
        g_end = min(g_start + rebase_group, num_chunks)
        n = g_end - g_start
        local_logGamma = logGamma[g_start:g_end]
        zero_shape = (1, d_k) if vector_decay else (1,)
        local_cum = torch.cat([torch.zeros(zero_shape, device=DEVICE, dtype=logspace_dtype),
                                torch.cumsum(local_logGamma, dim=0)], dim=0)
        for i, t in enumerate(range(g_start, g_end)):
            if vector_decay:
                decay0 = torch.exp(local_cum[i]).to(dtype)
                incoming = decay0.unsqueeze(-1) * running_state
                if i > 0:
                    w = torch.exp(local_cum[i].unsqueeze(0) - local_cum[1:i + 1]).to(dtype)
                    incoming = incoming + (w.unsqueeze(-1) * S_hat[g_start:g_start + i]).sum(dim=0)
            else:
                decay0 = torch.exp(local_cum[i]).to(dtype)
                incoming = decay0 * running_state
                if i > 0:
                    w = torch.exp(local_cum[i] - local_cum[1:i + 1]).to(dtype)
                    incoming = incoming + (w.view(-1, 1, 1) * S_hat[g_start:g_start + i]).sum(dim=0)
            M_incoming[t] = incoming

        group_total = local_cum[-1]
        if vector_decay:
            group_decay = torch.exp(group_total).to(dtype)
            w_group = torch.exp(group_total.unsqueeze(0) - local_cum[1:n + 1]).to(dtype)
            running_state = group_decay.unsqueeze(-1) * running_state + \
                (w_group.unsqueeze(-1) * S_hat[g_start:g_end]).sum(dim=0)
        else:
            group_decay = torch.exp(group_total).to(dtype)
            w_group = torch.exp(group_total - local_cum[1:n + 1]).to(dtype)
            running_state = group_decay * running_state + \
                (w_group.view(-1, 1, 1) * S_hat[g_start:g_end]).sum(dim=0)

    outs = torch.zeros(N, d_k, d_v, device=DEVICE, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        a_c = a[sl]
        if vector_decay:
            decay_to_j = torch.exp(torch.cumsum(torch.log(a_c.to(logspace_dtype)), dim=0)).to(dtype)
            outs[sl] = local_seqs[t] + decay_to_j.unsqueeze(-1) * M_incoming[t].unsqueeze(0)
        else:
            decay_to_j = torch.cumprod(a_c, dim=0)
            outs[sl] = local_seqs[t] + decay_to_j.view(-1, 1, 1) * M_incoming[t].unsqueeze(0)
    return outs


def naive_materialized_reconstruction(a, k, v, num_chunks, dtype):
    a, k, v = a.to(DEVICE, dtype), k.to(DEVICE, dtype), v.to(DEVICE, dtype)
    N, d_k = k.shape
    vector_decay = a.dim() == 2
    S_hat, Gamma, local_seqs, C = chunk_decompose(a, k, v, num_chunks, dtype)
    one_shape = (1, d_k) if vector_decay else (1,)
    Gamma_global = torch.cat([torch.ones(one_shape, device=DEVICE, dtype=dtype),
                               torch.cumprod(Gamma, dim=0)], dim=0)

    M_incoming = torch.zeros(num_chunks, *S_hat.shape[1:], device=DEVICE, dtype=dtype)
    for t in range(1, num_chunks):
        if vector_decay:
            weights = Gamma_global[t].unsqueeze(0) / Gamma_global[1:t + 1]
            M_incoming[t] = (weights.unsqueeze(-1) * S_hat[:t]).sum(dim=0)
        else:
            weights = Gamma_global[t] / Gamma_global[1:t + 1]
            M_incoming[t] = (weights.view(-1, 1, 1) * S_hat[:t]).sum(dim=0)

    outs = torch.zeros(N, *S_hat.shape[1:], device=DEVICE, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t * C, (t + 1) * C)
        decay_to_j = torch.cumprod(a[sl], dim=0)
        if vector_decay:
            outs[sl] = local_seqs[t] + decay_to_j.unsqueeze(-1) * M_incoming[t].unsqueeze(0)
        else:
            outs[sl] = local_seqs[t] + decay_to_j.view(-1, 1, 1) * M_incoming[t].unsqueeze(0)
    return outs, Gamma_global


def max_abs_err(x, y):
    return (x.double() - y.double()).abs().max().item()


def main():
    torch.manual_seed(0)
    d_k, d_v = 64, 64  # closer to real head dims than the toy script's scalar case
    print(f"device = {DEVICE}, name = "
          f"{torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'cpu'}\n")

    configs = [
        # (N, num_chunks, decay_lo, decay_hi, label)
        (2048, 16, 0.90, 0.999, "mild decay, few chunks"),
        (8192, 512, 0.90, 0.999, "mild decay, many chunks (stress)"),
        (8192, 512, 0.95, 0.9999, "decay closer to 1 (slower forgetting, longer eff. range)"),
        (8192, 512, 0.99, 0.99999, "decay very close to 1 (near-undecayed regime)"),
        (16384, 1024, 0.90, 0.999, "very many chunks"),
    ]

    for vector_decay in (False, True):
        decay_kind = "vector (GLA-style)" if vector_decay else "scalar (Mamba2-SSD-style)"
        print("=" * 100)
        print(f"decay = {decay_kind}")
        print("=" * 100)
        for N, num_chunks, lo, hi, label in configs:
            if vector_decay:
                a = torch.empty(N, d_k).uniform_(lo, hi)
            else:
                a = torch.empty(N).uniform_(lo, hi)
            k = torch.randn(N, d_k)
            v = torch.randn(N, d_v)

            ref = sequential_reference(a, k, v, dtype=torch.float64)

            for dtype in (torch.float32, torch.bfloat16):
                naive_out, Gamma_global = naive_materialized_reconstruction(a, k, v, num_chunks, dtype)
                safe_out = flat_logspace_reconstruction(a, k, v, num_chunks, dtype)

                n_zero = (Gamma_global == 0).sum().item()
                total = Gamma_global.numel()
                naive_err = max_abs_err(ref, naive_out)
                safe_err = max_abs_err(ref, safe_out)
                naive_has_nan = torch.isnan(naive_out).any().item()

                print(f"[{label}] N={N} chunks={num_chunks} decay=[{lo},{hi}] dtype={dtype}")
                print(f"    Gamma_global underflowed to 0 at {n_zero}/{total} entries "
                      f"({'NaN present' if naive_has_nan else 'no NaN'} in naive output)")
                print(f"    max abs error, NAIVE (materialize-then-divide) = {naive_err:.3e}")
                print(f"    max abs error, SAFE  (log-space difference)    = {safe_err:.3e}")

            if num_chunks >= 512:
                grouped_out = grouped_rebase_reconstruction(a, k, v, num_chunks,
                                                             rebase_group=16, dtype=torch.bfloat16)
                grouped_err = max_abs_err(ref, grouped_out)
                flat_out_bf16 = flat_logspace_reconstruction(a, k, v, num_chunks, dtype=torch.bfloat16)
                flat_err_bf16 = max_abs_err(ref, flat_out_bf16)
                print(f"    [bf16] flat log-space error = {flat_err_bf16:.3e}  vs  "
                      f"grouped rebasing (group=16) error = {grouped_err:.3e}")
            print()


if __name__ == "__main__":
    main()
