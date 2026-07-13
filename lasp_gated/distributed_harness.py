"""Real torch.distributed correctness harness (CPU, gloo backend) -- not a single-process
loop simulation. Each rank actually owns one chunk, and inter-rank communication happens
through real collectives (`dist.send`/`dist.recv`, `dist.all_gather`).

Baseline: sequential send/recv ring, matching Mamba-2's documented systems design (Tri Dao,
"SSD Part IV -- The Systems", 2024) and LASP-1's P2P pattern -- rank t waits to receive the
incoming state from rank t-1, adds its own local contribution, decays, and forwards to t+1.
This is O(P) sequential communication depth.

Candidate: a single all_gather of every rank's local (S_hat, Gamma) pair, then each rank
reconstructs its own incoming state purely locally via the log-space weighted prefix sum.
This is O(1) communication rounds.

Runs on CPU with the `gloo` backend so it works without any GPU -- this validates the
*correctness* of both communication patterns for the matrix-valued, decay-aware
reconstruction. Wall-clock comparison is deliberately NOT done here (gloo/CPU timings are
not representative of GPU interconnect behavior) -- see colab/ for the GPU benchmark.

Usage: python3 distributed_harness.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from matrix_lasp import sequential_reference, _chunk_local


def run_rank(rank, world_size, N, d_k, d_v, vector_decay, dtype, seed, result_queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.manual_seed(seed)
    C = N // world_size
    if vector_decay:
        a_full = torch.empty(N, d_k).uniform_(0.90, 0.999)
    else:
        a_full = torch.empty(N).uniform_(0.90, 0.999)
    k_full = torch.randn(N, d_k)
    v_full = torch.randn(N, d_v)

    sl = slice(rank * C, (rank + 1) * C)
    a_c, k_c, v_c = a_full[sl].to(dtype), k_full[sl].to(dtype), v_full[sl].to(dtype)
    S_hat, Gamma, local_seq = _chunk_local(a_c, k_c, v_c, dtype)

    # ---------------- baseline: sequential send/recv ring ----------------
    if rank == 0:
        incoming_baseline = torch.zeros(d_k, d_v, dtype=dtype)
    else:
        incoming_baseline = torch.zeros(d_k, d_v, dtype=dtype)
        dist.recv(incoming_baseline, src=rank - 1)

    decay_full = Gamma.unsqueeze(-1) if vector_decay else Gamma
    outgoing = decay_full * incoming_baseline + S_hat
    if rank < world_size - 1:
        dist.send(outgoing.contiguous(), dst=rank + 1)

    if vector_decay:
        decay_to_j = torch.exp(torch.cumsum(torch.log(a_c.to(torch.float32)), dim=0)).to(dtype)
        baseline_out = local_seq + decay_to_j.unsqueeze(-1) * incoming_baseline.unsqueeze(0)
    else:
        decay_to_j = torch.cumprod(a_c, dim=0)
        baseline_out = local_seq + decay_to_j.view(-1, 1, 1) * incoming_baseline.unsqueeze(0)

    dist.barrier()

    # ---------------- candidate: single all_gather + local log-space reconstruction ----------------
    S_hat_all = [torch.zeros(d_k, d_v, dtype=dtype) for _ in range(world_size)]
    dist.all_gather(S_hat_all, S_hat.contiguous())
    Gamma_shape = Gamma.shape
    Gamma_all = [torch.zeros(Gamma_shape, dtype=dtype) for _ in range(world_size)]
    dist.all_gather(Gamma_all, Gamma.contiguous())

    S_hat_stack = torch.stack(S_hat_all)
    Gamma_stack = torch.stack(Gamma_all)
    logGamma = torch.log(Gamma_stack.to(torch.float32))
    zero_shape = (1, d_k) if vector_decay else (1,)
    cum = torch.cat([torch.zeros(zero_shape, dtype=torch.float32), torch.cumsum(logGamma, dim=0)], dim=0)

    if rank == 0:
        incoming_candidate = torch.zeros(d_k, d_v, dtype=dtype)
    else:
        if vector_decay:
            w = torch.exp(cum[rank].unsqueeze(0) - cum[1:rank + 1]).to(dtype)
            incoming_candidate = (w.unsqueeze(-1) * S_hat_stack[:rank]).sum(dim=0)
        else:
            w = torch.exp(cum[rank] - cum[1:rank + 1]).to(dtype)
            incoming_candidate = (w.view(-1, 1, 1) * S_hat_stack[:rank]).sum(dim=0)

    if vector_decay:
        candidate_out = local_seq + decay_to_j.unsqueeze(-1) * incoming_candidate.unsqueeze(0)
    else:
        candidate_out = local_seq + decay_to_j.view(-1, 1, 1) * incoming_candidate.unsqueeze(0)

    err_baseline_vs_candidate = (baseline_out.double() - candidate_out.double()).abs().max().item()
    result_queue.put((rank, err_baseline_vs_candidate))

    dist.destroy_process_group()


def main():
    mp.set_start_method("spawn", force=True)
    N, world_size, d_k, d_v = 512, 8, 8, 8
    for vector_decay in (False, True):
        for dtype in (torch.float32, torch.bfloat16):
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            procs = []
            for rank in range(world_size):
                p = ctx.Process(target=run_rank, args=(rank, world_size, N, d_k, d_v,
                                                         vector_decay, dtype, 0, result_queue))
                p.start()
                procs.append(p)
            errs = [result_queue.get() for _ in range(world_size)]
            for p in procs:
                p.join()
            max_err = max(e for _, e in errs)
            decay_kind = "vector (GLA-style)" if vector_decay else "scalar (Mamba2-SSD-style)"
            print(f"decay={decay_kind:20s} dtype={str(dtype):16s} "
                  f"send/recv-ring vs all_gather+log-space, {world_size} real ranks (gloo): "
                  f"max abs diff across ranks = {max_err:.3e}")


if __name__ == "__main__":
    main()
