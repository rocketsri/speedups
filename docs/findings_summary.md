# Findings: does LASP-2's single-all-gather generalize to gated/decaying models?

Summary against the task's steps. Full detail in `step0_findings.md`, code in
`lasp_gated/`, GPU scripts + usage in `colab/`.

## Step 0 — is this already solved? No.

Checked the full LASP-2 paper (arXiv:2502.07563, Eq. 4 and Appendix A.3/Algorithm 7) and the
released code (`OpenSparseLLMs/Linear-MoE`). The paper's recurrence and its all-gather combine
step (`M_{1:T} = Sum([M_t])`) are strictly additive/undecayed. In the code, `lasp2.py`'s
all-gather path takes only `(q, k, v)` — no decay tensor accepted at all — while `gla.py` and
`gated_deltanet.py` (the actual gated models) call single-device `fla` kernels directly with
zero sequence-parallel/context-parallel integration. Today, in this codebase, "has
sequence-parallelism" and "has decay" are mutually exclusive. See `step0_findings.md` for the
full citation trail.

## Step 1 — port to matrix-valued state: done (`lasp_gated/matrix_lasp.py`)

Ported the scalar toy derivation to a real `d_k x d_v` state matrix, covering both:
- **Mamba2/SSD-style scalar decay** (`a_s` a scalar, `M_s = a_s M_{s-1} + k_s (x) v_s`) — the
  direct generalization, state becomes a matrix but the decay chain is unchanged.
- **GLA-style vector/diagonal decay** (`a_s` a `d_k`-vector gate, `diag(a_s)` applied to
  `M_{s-1}`) — a strict generalization; since diagonal matrices commute, each row is an
  independent scalar decay chain, so the same log-space trick applies per-row (vectorized),
  with no new math needed.

Implemented all three reconstruction variants from the reference script, matrix-valued: flat
log-space all-gather, grouped/hierarchical rebasing, and the naive materialize-then-divide
failure mode (for comparison).

**Baseline vs. candidate, as real distributed code, not loop simulation**
(`lasp_gated/distributed_harness.py`): sequential send/recv ring across real
`torch.distributed` ranks (matching Mamba-2's documented systems design) vs. a single
`all_gather` + local log-space reconstruction, using the `gloo` backend on CPU (no GPU needed
for this correctness check). Both real 8-rank runs agree with each other to ~1e-6 (fp32) and
to within the bf16 noise floor (bf16) — see next section.

## Step 2 — correctness fp32/bf16: done on CPU, holds

`lasp_gated/run_cpu_correctness.py` output (fp64/fp32/bf16, scalar and vector decay, N up to
8192 / 512 chunks):
- fp64: matches sequential reference to ~1e-13-1e-15 (machine precision), both decay shapes.
- fp32: ~1e-4 max abs error (expected fp32 accumulation noise over long sequences).
- bf16: ~0.2-0.3 max abs error — **but this is inherent to bf16 storage precision itself, not
  something the chunked/log-space reconstruction adds**. Verified directly: a *pure
  sequential* recurrence computed entirely in bf16 (zero chunking, zero reconstruction) has
  ~0.27 max abs error against the fp64 reference on the same data — same order of magnitude.
  bf16 has ~8 bits of mantissa; over an 8192-step accumulation that's the expected floor
  regardless of algorithm.
- Naive materialize-then-divide reproduces the failure mode in the matrix case exactly as in
  the scalar toy script: 387-3110 out of ~500-4100 `Gamma_global` entries underflow to exact
  0.0 in fp32, producing NaN throughout the naive output, while the log-space version stays
  at ~1.2e-4.

## Step 3 — real GPU hardware: scripts ready, **not yet run** (no GPU in this environment)

`colab/step3_gpu_numerics.py` — single-GPU (targets Colab L4), sweeps chunk counts 16-1024 and
decay ranges from mild (0.90-0.999) to near-undecayed (0.99-0.99999), fp32 and bf16, both
decay shapes. Logic was smoke-tested end-to-end on CPU (fallback device) in this sandbox and
runs without error, reproducing the same qualitative failure mode as Step 2. **Real bf16
tensor-core behavior on GPU has not been observed** — the task doc explicitly flags this as
the actual unknown, and that's still true. Run it and report back the printed numbers;
particularly whether grouped rebasing starts mattering at chunk counts beyond ~1024 (in the
CPU smoke test it did not, because flat log-space with an fp32 accumulator hadn't broken down
yet at that scale — Step 5 stays conditional on this).

## Step 4 — wall-clock benchmark: script ready, **not yet run**

`colab/step4_benchmark.py`, launched via `torchrun --nproc_per_node=P`. Auto-detects NCCL
(if `P <= visible GPU count`, bandwidth-representative) vs. gloo-with-host-staging fallback
(realistic single-L4 Colab case — real processes, real send/recv/all_gather calls, but not
NVLink/PCIe-bandwidth-representative). Smoke-tested on CPU with 4 and 8 simulated ranks via
real `torchrun` launches — correctness check passes (~1e-11 fp32 CPU baseline), timing
numbers printed successfully end to end. **No claim about the actual crossover point is made
here** — that requires running on GPU across a range of `P`, which is exactly what's queued up
for you to do next. See `colab/README.md` for exact commands and the single-GPU caveat in
full.

## Step 5 — grouped/hierarchical rebasing: implemented, not yet shown to be necessary

Implemented in both `lasp_gated/matrix_lasp.py` (CPU) and `colab/step3_gpu_numerics.py` (GPU).
In the CPU stress tests up to 1024 chunks, flat log-space (fp32 accumulator) never actually
broke down — grouped rebasing gave the same error, not better — so per the task's own
condition ("only if Step 3 shows trouble"), it isn't yet demonstrated to be needed. Whether
real GPU numerics at larger chunk counts change this conclusion is an open question for
Step 3's actual GPU run.

## Bottom line

The gap is real (Step 0), the math generalizes cleanly to matrix state for both Mamba2-SSD
and GLA-style decay with no new derivation needed (Step 1), and it's been verified correct
against a sequential reference in fp64/fp32/bf16 using real (not simulated) distributed
communication (Steps 1-2). What's *not* yet verified is GPU-specific: real bf16 tensor-core
numerics (Step 3) and whether send/recv's sequential depth actually costs more than a single
all_gather at practical GPU counts (Step 4) — both scripts are ready in `colab/`, written to
auto-adapt from a single L4 up to whatever GPU count you actually have.
