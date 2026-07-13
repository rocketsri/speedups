# Colab-ready GPU scripts (Steps 3-4)

Target: a single Colab GPU runtime (L4, or whatever you're given — T4/A100 also fine for
Step 3; Step 4's absolute numbers depend on the GPU/interconnect, see caveats below).

Both scripts were smoke-tested end-to-end in this sandbox on CPU (no GPU available here) to
confirm the launch mechanics, argument handling, and reconstruction logic are correct. GPU
execution (real bf16 tensor cores, real NCCL/gloo transport) has **not** been observed by me
— that's the actual point of running them yourself.

## Setup (one cell)

```
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Colab GPU runtimes ship with `torch` preinstalled already matched to the CUDA driver, so no
install step should be needed. If `torch.cuda.is_available()` is False, go to
`Runtime > Change runtime type` and pick a GPU (e.g. L4).

## Step 3 — numerical failure-mode reproduction on real GPU bf16/fp32

```
!python step3_gpu_numerics.py
```

Single process, single GPU, no `torch.distributed` needed — this is purely about whether
`exp(cumulative log-decay)` survives in real GPU bf16/fp32 arithmetic, at varying chunk
counts (16 to 1024) and decay ranges (0.90-0.999 up to 0.99-0.99999), for both a
Mamba2-SSD-style scalar decay and a GLA-style per-row vector decay. Prints, per config:
number of `Gamma_global` entries that underflowed to exactly 0, whether the naive
materialize-then-divide reconstruction produced NaN, and max abs error of both the naive and
the log-space-safe reconstruction against a float64 sequential reference. At `num_chunks >=
512` it also compares flat log-space vs. grouped/hierarchical rebasing in bf16.

CPU smoke-test results (for reference — expect GPU numbers to differ in exact error
magnitudes, especially for bf16, since real tensor-core bf16 accumulation semantics differ
from CPU bf16 emulation):
- fp32: naive reconstruction hits NaN as soon as num_chunks reaches a few hundred at
  decay ~0.9-0.999 (hundreds of `Gamma_global` entries underflow to exact 0). The log-space
  version stays around 1e-4 to 5e-4 max abs error throughout.
- bf16: the log-space version's error is dominated by bf16's mantissa precision itself
  (already ~0.3-1.7 max abs error with *zero* chunking at all — see
  `lasp_gated/run_cpu_correctness.py`'s output), not by the reconstruction algorithm. Grouped
  rebasing did not measurably help at these scales in the CPU smoke test, because the
  problem it targets (exponent overflow/underflow) is already fixed by flat log-space with an
  fp32 accumulator — it only matters if the *flat* log-space sum itself starts breaking down,
  which didn't happen up to 1024 chunks here. Check whether this holds on real GPU bf16, and
  whether it changes at chunk counts beyond 1024.

## Step 4 — wall-clock benchmark: send/recv ring vs. all_gather

```
!torchrun --nproc_per_node=8 step4_benchmark.py --chunk-len 4096 --d-k 128 --d-v 128 --dtype bf16
```

Run once per world size `P` you want to test (e.g. 2, 4, 8, 16, 32, 64 — `--nproc_per_node`
*is* `P`, one chunk/rank each) and compare the printed median times across runs to look for
the crossover the task asks about: does send/recv's `O(P)` sequential communication depth
ever cost more than all_gather's `O(1)` round, and if so at what `P`?

Flags:
- `--chunk-len` (local sequence length per chunk/rank, default 4096)
- `--d-k`, `--d-v` (state matrix dims, default 128x128 — realistic head-dim scale)
- `--vector-decay` (GLA-style per-row decay instead of the Mamba2-SSD scalar default)
- `--dtype {fp32,bf16}`
- `--iters`, `--warmup` (timed / untimed repetitions, default 20 / 5)

**Single-GPU caveat (this is the realistic Colab case):** with only 1 physical GPU, all `P`
ranks share it. NCCL point-to-point generally assumes one GPU per rank, so when
`world_size > torch.cuda.device_count()` the script auto-falls-back to the `gloo` backend and
explicitly stages every collective through host memory. That's real inter-process
communication (not a loop-simulated fake), but the *absolute* numbers won't reflect true
NVLink/PCIe P2P bandwidth — ring's per-hop cost here is mostly process-wakeup + host-copy
latency. The `O(P)` vs `O(1)` *round-count* structure is still real and is what the task's
crossover question is actually about. If you get access to a multi-GPU instance
(`P <= torch.cuda.device_count()`), the script automatically switches to NCCL with one GPU
per rank, which gives a bandwidth-representative number instead — worth trying if you can get
e.g. an A100 x2/x4 Colab Enterprise instance or a rented multi-GPU box, since single-L4
numbers alone can't fully settle the crossover claim.

Each run also does a real distributed correctness check (ring- vs all_gather-reconstructed
incoming state, max diff across all ranks) before timing, printed as
`[correctness] max abs diff ...` — should be ~1e-6 (fp32) or bf16-noise-floor (bf16), matching
the CPU `gloo` run already validated in `lasp_gated/distributed_harness.py`.
