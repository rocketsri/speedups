# speedups

Investigation: does LASP-2's single-all-gather sequence parallelism generalize to
gated/decaying linear attention & SSMs (GLA, Mamba2/SSD, RetNet)?

Start here: **`docs/findings_summary.md`**.

- `docs/step0_findings.md` — literature/code check of LASP-2 (arXiv:2502.07563) and
  `OpenSparseLLMs/Linear-MoE`: does the existing all-gather path already handle decay?
- `docs/gated_lasp_verify_reference.py` — original scalar-state CPU derivation/reference this
  work is based on.
- `lasp_gated/` — matrix-valued (`d_k x d_v` state) port of the derivation, CPU-only:
  - `matrix_lasp.py` — sequential reference, flat log-space all-gather reconstruction,
    grouped/hierarchical rebasing, naive materialize-then-divide failure mode. Covers both
    Mamba2-SSD-style scalar decay and GLA-style per-row vector decay.
  - `run_cpu_correctness.py` — correctness across fp64/fp32/bf16, safe and stress regimes.
  - `distributed_harness.py` — real `torch.distributed` (gloo/CPU) multi-rank run comparing
    a sequential send/recv ring vs. a single `all_gather` + local reconstruction.
- `colab/` — GPU scripts (no GPU available in this dev environment; run these on a Colab
  GPU runtime, e.g. L4). See `colab/README.md` for exact commands.
  - `step3_gpu_numerics.py` — reproduce the numerical failure mode in real bf16/fp32 GPU
    arithmetic across chunk counts and decay ranges.
  - `step4_benchmark.py` — wall-clock benchmark, send/recv ring vs. all_gather, via
    `torchrun`, auto-adapting from a single GPU up to multi-GPU.
