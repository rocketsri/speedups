import numpy as np

np.random.seed(0)

def sequential_reference(a, k, v):
    """Ground truth: M_s = a_s * M_{s-1} + k_s^T v_s, elementwise on scalar state (d=1 simplification)."""
    N = len(a)
    M = 0.0
    outs = []
    for s in range(N):
        M = a[s] * M + k[s] * v[s]
        outs.append(M)
    return np.array(outs)

def chunked_allgather_reconstruction(a, k, v, num_chunks, use_rebasing=False, rebase_group=8):
    """Simulate: each rank computes local (S_hat_t, Gamma_t) for its chunk independently,
    then a single 'all-gather' round shares all (S_hat_t, Gamma_t) pairs,
    then each rank reconstructs incoming state via weighted prefix sum, purely locally."""
    N = len(a)
    C = N // num_chunks
    assert N % num_chunks == 0

    S_hat = np.zeros(num_chunks)
    Gamma = np.zeros(num_chunks)
    local_outs_within_chunk = []  # per-chunk arrays of "local-only" running values (ignoring incoming state)

    for t in range(num_chunks):
        sl = slice(t*C, (t+1)*C)
        at, kt, vt = a[sl], k[sl], v[sl]
        M_local = 0.0
        local_seq = []
        for j in range(C):
            M_local = at[j]*M_local + kt[j]*vt[j]
            local_seq.append(M_local)
        S_hat[t] = M_local          # local final state, ignoring incoming state
        Gamma[t] = np.prod(at)      # total decay across this chunk
        local_outs_within_chunk.append(np.array(local_seq))

    # ---- "all-gather" phase: every rank now has all (S_hat, Gamma) pairs ----
    logGamma = np.log(Gamma)  # log-space cumulative decay weights

    if not use_rebasing:
        # flat global log-cumsum (single reference point) -- this is what can blow up
        cum_logGamma = np.concatenate(([0.0], np.cumsum(logGamma)))  # cum_logGamma[t] = sum_{u<t} logGamma[u]
        M_incoming = np.zeros(num_chunks)
        for t in range(num_chunks):
            # weight for term t' < t: exp(cum_logGamma[t] - cum_logGamma[t'+1])
            if t == 0:
                continue
            weights = np.exp(cum_logGamma[t] - cum_logGamma[1:t+1])
            M_incoming[t] = np.sum(weights * S_hat[:t])
    else:
        # grouped/hierarchical rebasing: combine in groups of `rebase_group` chunks first (bounded range),
        # then combine group-level summaries (bounded range at that level too)
        M_incoming = np.zeros(num_chunks)
        # process group by group, carrying only a group-level running state (like the sequential ring,
        # but only ONE hop per group instead of per chunk -- reduces sequential depth by factor rebase_group
        # while keeping every intra-group exponent local/bounded)
        running_state_before_group = 0.0
        for g_start in range(0, num_chunks, rebase_group):
            g_end = min(g_start+rebase_group, num_chunks)
            # within this group, weights are LOCAL (bounded: at most rebase_group hops of decay)
            local_logGamma = logGamma[g_start:g_end]
            local_cum = np.concatenate(([0.0], np.cumsum(local_logGamma)))
            for i, t in enumerate(range(g_start, g_end)):
                # incoming = decayed running_state_before_group + weighted sum of S_hat within this group so far
                decay_from_group_start = np.exp(local_cum[i])
                within_group_weights = np.exp(local_cum[i] - local_cum[1:i+1]) if i > 0 else np.array([])
                within_group_contrib = np.sum(within_group_weights * S_hat[g_start:g_start+i]) if i > 0 else 0.0
                M_incoming[t] = decay_from_group_start * running_state_before_group + within_group_contrib
            group_total_logGamma = local_cum[-1]
            running_state_before_group = np.exp(group_total_logGamma) * running_state_before_group + S_hat[g_start:g_end] @ np.exp(local_cum[-1] - local_cum[1:g_end-g_start+1])

    # reconstruct full output sequence: for each chunk, add decayed incoming state to local within-chunk sequence
    outs = np.zeros(N)
    for t in range(num_chunks):
        sl = slice(t*C, (t+1)*C)
        at = a[sl]
        local_seq = local_outs_within_chunk[t]
        # decay factors from chunk start to each position j: prod_{i=1}^{j} a[i]
        decay_to_j = np.cumprod(at)
        outs[sl] = local_seq + decay_to_j * M_incoming[t]
    return outs

# ---------------- Test 1: correctness on a "safe" regime ----------------
N = 512
num_chunks = 16
a = np.random.uniform(0.9, 0.999, N)   # mild decay, safe range
k = np.random.randn(N)
v = np.random.randn(N)

ref = sequential_reference(a, k, v)
recon = chunked_allgather_reconstruction(a, k, v, num_chunks, use_rebasing=False)
err = np.max(np.abs(ref - recon))
print(f"[Safe regime] N={N}, chunks={num_chunks}, max abs error (flat all-gather) = {err:.3e}")

# ---------------- Test 2: same math, hierarchical/grouped rebasing ----------------
recon_grouped = chunked_allgather_reconstruction(a, k, v, num_chunks, use_rebasing=True, rebase_group=4)
err_grouped = np.max(np.abs(ref - recon_grouped))
print(f"[Safe regime] max abs error (grouped rebasing)  = {err_grouped:.3e}")

# ---------------- Test 3: stress regime -- long sequence, many chunks, decay close to boundary ----------------
N2 = 8192
num_chunks2 = 512   # many chunks -> long cumulative log-range
a2 = np.random.uniform(0.90, 0.999, N2)
k2 = np.random.randn(N2)
v2 = np.random.randn(N2)

ref2 = sequential_reference(a2, k2, v2)
recon2_flat = chunked_allgather_reconstruction(a2, k2, v2, num_chunks2, use_rebasing=False)
recon2_grouped = chunked_allgather_reconstruction(a2, k2, v2, num_chunks2, use_rebasing=True, rebase_group=16)

err2_flat = np.max(np.abs(ref2 - recon2_flat))
err2_grouped = np.max(np.abs(ref2 - recon2_grouped))
print(f"\n[Stress regime] N={N2}, chunks={num_chunks2}")
print(f"  max abs error (flat all-gather)    = {err2_flat:.3e}")
print(f"  max abs error (grouped rebasing)   = {err2_grouped:.3e}")
print(f"  reference output magnitude range   = [{ref2.min():.3e}, {ref2.max():.3e}]")

# ---------------- Test 4: the ACTUAL naive failure mode (materializing exp() separately, fp32) ----------------
print("\n[Test 4] Deliberately naive version: materialize Gamma_global_t = exp(cumsum) as an array,\n"
      "then reconstruct weights as division, in float32 (representative of GPU training precision).")

def naive_materialized_exp_reconstruction(a, k, v, num_chunks, dtype=np.float32):
    N = len(a)
    C = N // num_chunks
    a = a.astype(dtype)
    S_hat = np.zeros(num_chunks, dtype=dtype)
    Gamma = np.zeros(num_chunks, dtype=dtype)
    local_outs_within_chunk = []
    for t in range(num_chunks):
        sl = slice(t*C, (t+1)*C)
        at, kt, vt = a[sl], k[sl].astype(dtype), v[sl].astype(dtype)
        M_local = dtype(0.0)
        local_seq = []
        for j in range(C):
            M_local = at[j]*M_local + kt[j]*vt[j]
            local_seq.append(M_local)
        S_hat[t] = M_local
        Gamma[t] = np.prod(at.astype(np.float64)).astype(dtype)  # chunk decay factor
        local_outs_within_chunk.append(np.array(local_seq, dtype=dtype))

    # THE NAIVE STEP: materialize the global cumulative product directly (not log-space difference)
    Gamma_global = np.concatenate(([dtype(1.0)], np.cumprod(Gamma)))  # can underflow to 0 in fp32!

    M_incoming = np.zeros(num_chunks, dtype=dtype)
    for t in range(num_chunks):
        if t == 0:
            continue
        # naive weight: Gamma_global[t] / Gamma_global[t'+1]  (division of two possibly-underflowed numbers)
        weights = Gamma_global[t] / Gamma_global[1:t+1]
        M_incoming[t] = np.sum(weights * S_hat[:t])

    outs = np.zeros(N, dtype=dtype)
    for t in range(num_chunks):
        sl = slice(t*C, (t+1)*C)
        at = a[sl]
        local_seq = local_outs_within_chunk[t]
        decay_to_j = np.cumprod(at)
        outs[sl] = local_seq + decay_to_j * M_incoming[t]
    return outs, Gamma_global

N3 = 8192
num_chunks3 = 512
a3 = np.random.uniform(0.90, 0.999, N3)
k3 = np.random.randn(N3)
v3 = np.random.randn(N3)

ref3 = sequential_reference(a3, k3, v3)
naive_out, Gamma_global = naive_materialized_exp_reconstruction(a3, k3, v3, num_chunks3, dtype=np.float32)
safe_out_fp32 = chunked_allgather_reconstruction(a3.astype(np.float32), k3.astype(np.float32),
                                                   v3.astype(np.float32), num_chunks3, use_rebasing=False)

n_zero = np.sum(Gamma_global == 0.0)
print(f"  Gamma_global underflowed to exactly 0.0 at {n_zero}/{len(Gamma_global)} chunk boundaries")
print(f"  max abs error, NAIVE (materialize-then-divide, fp32) = {np.max(np.abs(ref3 - naive_out)):.3e}")
print(f"  max abs error, SAFE  (log-space difference, fp32)    = {np.max(np.abs(ref3 - safe_out_fp32)):.3e}")
print(f"  reference magnitude range = [{ref3.min():.3e}, {ref3.max():.3e}]")
