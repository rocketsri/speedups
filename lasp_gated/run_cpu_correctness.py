"""Matrix-valued, CPU-only correctness check -- the direct analog of gated_lasp_verify.py's
Tests 1-4, but with a real d_k x d_v state instead of a scalar, and covering both decay
shapes (Mamba2-SSD scalar decay, GLA-style per-row vector decay).

This is Step 1 + Step 2 of the task: port the derivation to matrix-valued state and check
correctness in fp32/bf16 against a single-device sequential reference. No GPU required --
run mode: `python3 run_cpu_correctness.py`.
"""
import torch
from matrix_lasp import (
    sequential_reference,
    flat_logspace_reconstruction,
    grouped_rebase_reconstruction,
    naive_materialized_reconstruction,
)

torch.manual_seed(0)


def make_inputs(N, d_k, d_v, vector_decay, decay_lo=0.90, decay_hi=0.999):
    if vector_decay:
        a = torch.empty(N, d_k).uniform_(decay_lo, decay_hi)
    else:
        a = torch.empty(N).uniform_(decay_lo, decay_hi)
    k = torch.randn(N, d_k)
    v = torch.randn(N, d_v)
    return a, k, v


def max_abs_err(x, y):
    return (x.to(torch.float64) - y.to(torch.float64)).abs().max().item()


def run_regime(tag, N, num_chunks, d_k, d_v, vector_decay, decay_lo, decay_hi,
                test_dtype=torch.float64, rebase_group=8):
    a, k, v = make_inputs(N, d_k, d_v, vector_decay, decay_lo, decay_hi)
    decay_kind = "vector (GLA-style)" if vector_decay else "scalar (Mamba2-SSD-style)"

    ref = sequential_reference(a, k, v, dtype=torch.float64)

    flat = flat_logspace_reconstruction(a, k, v, num_chunks, dtype=test_dtype)
    grouped = grouped_rebase_reconstruction(a, k, v, num_chunks, rebase_group=rebase_group,
                                             dtype=test_dtype)

    err_flat = max_abs_err(ref, flat)
    err_grouped = max_abs_err(ref, grouped)

    print(f"[{tag}] decay={decay_kind}, dtype={test_dtype}, N={N}, chunks={num_chunks}, "
          f"d_k={d_k}, d_v={d_v}")
    print(f"    max abs error (flat log-space all-gather) = {err_flat:.3e}")
    print(f"    max abs error (grouped rebasing, group={rebase_group}) = {err_grouped:.3e}")
    return err_flat, err_grouped


print("=" * 88)
print("Test 1/2 analog: safe decay regime, matrix state, both decay shapes, fp64")
print("=" * 88)
for vector_decay in (False, True):
    run_regime("Safe regime", N=512, num_chunks=16, d_k=8, d_v=8,
               vector_decay=vector_decay, decay_lo=0.9, decay_hi=0.999,
               test_dtype=torch.float64, rebase_group=4)

print()
print("=" * 88)
print("Test 3 analog: stress regime -- long sequence, many chunks, fp64")
print("=" * 88)
for vector_decay in (False, True):
    run_regime("Stress regime", N=8192, num_chunks=512, d_k=8, d_v=8,
               vector_decay=vector_decay, decay_lo=0.90, decay_hi=0.999,
               test_dtype=torch.float64, rebase_group=16)

print()
print("=" * 88)
print("Step 2: fp32 and bf16 precision, matrix state (log-space math accumulated in fp32")
print("even under bf16 storage, matching standard kernel practice)")
print("=" * 88)
for vector_decay in (False, True):
    for dtype in (torch.float32, torch.bfloat16):
        a, k, v = make_inputs(8192, 8, 8, vector_decay, 0.90, 0.999)
        ref = sequential_reference(a, k, v, dtype=torch.float64)
        flat = flat_logspace_reconstruction(a, k, v, 512, dtype=dtype, logspace_dtype=torch.float32)
        err = max_abs_err(ref, flat)
        decay_kind = "vector" if vector_decay else "scalar"
        print(f"  decay={decay_kind:7s} dtype={str(dtype):20s} max abs error = {err:.3e}")

print()
print("=" * 88)
print("Test 4 analog: naive materialize-then-divide failure mode, matrix state, fp32")
print("=" * 88)
for vector_decay in (False, True):
    N3, num_chunks3 = 8192, 512
    a3, k3, v3 = make_inputs(N3, 8, 8, vector_decay, 0.90, 0.999)
    ref3 = sequential_reference(a3, k3, v3, dtype=torch.float64)

    naive_out, Gamma_global = naive_materialized_reconstruction(a3, k3, v3, num_chunks3,
                                                                  dtype=torch.float32)
    safe_out = flat_logspace_reconstruction(a3, k3, v3, num_chunks3, dtype=torch.float32,
                                             logspace_dtype=torch.float32)

    n_zero = (Gamma_global == 0.0).sum().item()
    total = Gamma_global.numel()
    decay_kind = "vector (GLA-style)" if vector_decay else "scalar (Mamba2-SSD-style)"
    print(f"[decay={decay_kind}]")
    print(f"    Gamma_global underflowed to exactly 0.0 at {n_zero}/{total} entries")
    print(f"    max abs error, NAIVE (materialize-then-divide, fp32) = "
          f"{max_abs_err(ref3, naive_out):.3e}")
    print(f"    max abs error, SAFE  (log-space difference, fp32)    = "
          f"{max_abs_err(ref3, safe_out):.3e}")
    print(f"    reference magnitude range = [{ref3.min():.3e}, {ref3.max():.3e}]")
