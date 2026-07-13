# Step 0: Does LASP-2's all-gather already handle gated/decaying recurrences?

**Short answer: No.** The gap described in the task doc is real, in both the paper and the
released code. Below is what was actually checked (arXiv:2502.07563 full HTML, and the
`OpenSparseLLMs/Linear-MoE` source), not re-derived from the abstract alone.

## Paper (arXiv:2502.07563, full text + Appendix A.3)

- The core recurrence given in the Preliminary/Method section (Eq. 4) is:

  ```
  M_s = M_{s-1} + k_s^T v_s,   o_s = q_s M_s
  ```

  Strictly additive. No decay/gate/forget multiplier on `M_{s-1}` anywhere in this equation.

- Appendix A.3 ("AllGather-based Context Parallelism", Algorithm 7): after every device
  gathers all per-chunk memory states via a single `all_gather`, they are combined as

  ```
  M_{1:T} = Sum([M_t]_1^T)
  ```

  Plain summation. No weighting term, no cumulative-product/decay factor, no log-space
  anything — because there is nothing to weight in the undecayed model.

- GLA is listed in the experiments table (Table 2, loss comparison) alongside Lightning
  Attention, Retention, Based, Rebased — but the paper gives **no separate formula** for how
  GLA's decay gate would interact with the all-gather combine step. It's evaluated as one of
  several "linear attention variants" under the same undecayed communication algorithm; the
  decay is implicitly treated as intra-chunk-only, consistent with how it's actually
  implemented (see below).

- No discussion anywhere of numerical stability, cumulative decay products, or log-space
  computation — unsurprising, since the paper's own math never produces a decay term that
  could underflow/overflow in the first place.

## Code (`github.com/OpenSparseLLMs/Linear-MoE`, `main` branch)

`linear_moe/sequence_modeling/` has separate directories per model: `lasp2/`, `gla/`,
`mamba2/`, `gated_deltanet/`, `retention/`, etc.

- **`lasp2/lasp2.py`** — the actual sequence-parallel all-gather implementation. Its
  `forward()` signature is:

  ```python
  def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
      ...
      output = self._la_impl(q, k, v, get_context_parallel_group())
  ```

  Only `q, k, v` — **no decay/gate tensor is accepted at all**. The combine logic is delegated
  to `lasp2_with_mask_triton_op` / `lasp2_without_mask_triton_op`, but there's no decay input
  to weight anything with in the first place.

- **`gla/gla.py`** — imports `chunk_gla, fused_chunk_gla, fused_recurrent_gla` from the `fla`
  library and calls one of those directly. No import of `lasp2`, no reference to
  `sequence_parallel`/`context_parallel`/`get_context_parallel_group`, no cross-device
  communication of any kind. It is a single-device module.

- **`gated_deltanet/gated_deltanet.py`** — same pattern: imports `chunk_gated_delta_rule` from
  `fla`, `forward(self, q, k, v, beta, gk)`, zero sequence-parallel integration.

- `mamba2/` follows the same directory convention as the other single-device modules (no file
  in it references `lasp2` or context-parallel group lookups either).

**Conclusion:** in this codebase, sequence parallelism (the all-gather path) and gating/decay
are mutually exclusive today. `lasp2.py` gets you SP but only for plain linear attention.
`gla.py` / `gated_deltanet.py` get you decay but are single-device only — if you wanted to
train GLA or a gated model at long context across multiple GPUs with this codebase as-is, there
is no code path that does both simultaneously with a single collective. The "LASP-2 supports
GLA" claim in the paper's experiments table is about model-quality parity under whatever
parallelism *was* used for that experiment (single device, or plain data/tensor parallelism) —
not evidence that GLA's decay is handled inside the all-gather reconstruction.

This matches the task doc's stated gap exactly. The rest of this work (matrix-valued port,
correctness tests, GPU numerics, benchmarking) proceeds on that basis.
