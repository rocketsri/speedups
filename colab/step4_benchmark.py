"""Step 4: wall-clock benchmark -- sequential send/recv ring vs a single all_gather, for
reconstructing cross-chunk incoming states in gated/decaying linear attention.

COLAB USAGE (target: single L4 GPU; also works with more GPUs if you have them):

    !nvidia-smi
    !pip install -q torch  # already preinstalled on Colab GPU runtimes, usually a no-op
    !torchrun --nproc_per_node=8 step4_benchmark.py --chunk-len 4096 --d-k 128 --d-v 128

Run once per world size P in e.g. {2, 4, 8, 16, 32, 64} (torchrun's --nproc_per_node IS the
world size / number of chunks-ranks for that run) and compare the printed numbers across
runs to look for a crossover where all_gather's O(1) round beats the ring's O(P) sequential
communication depth.

Single-GPU caveat (this is the realistic Colab case): with only 1 physical GPU, torchrun
still launches P real OS processes, but they all target the same device. NCCL point-to-point
generally assumes one GPU per rank, so when world_size > number of visible GPUs this script
automatically falls back to the `gloo` backend and stages every collective through host
memory (CPU) -- real send/recv and all_gather syscalls, real process scheduling, just not
NVLink/PCIe P2P bandwidth. That means the *absolute* numbers you get on a single L4 are a
lower bound on how good all_gather looks (ring's per-hop overhead here is mostly
process-wakeup + host-memory-copy latency, not real network cost) -- but the O(P) vs O(1)
*round count* comparison this task cares about is still real. If you get access to a
multi-GPU box (P <= number of visible GPUs), the script automatically switches to NCCL with
one GPU per rank for a bandwidth-representative number instead.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def build_chunk(rank, chunk_len, d_k, d_v, vector_decay, dtype, device):
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    if vector_decay:
        a = torch.empty(chunk_len, d_k).uniform_(0.90, 0.999, generator=g)
    else:
        a = torch.empty(chunk_len).uniform_(0.90, 0.999, generator=g)
    k = torch.randn(chunk_len, d_k, generator=g)
    v = torch.randn(chunk_len, d_v, generator=g)
    a = a.to(device=device, dtype=dtype)
    k = k.to(device=device, dtype=dtype)
    v = v.to(device=device, dtype=dtype)

    M = torch.zeros(d_k, d_v, device=device, dtype=dtype)
    for j in range(chunk_len):
        decay = a[j].unsqueeze(-1) if vector_decay else a[j]
        M = decay * M + torch.outer(k[j], v[j])
    S_hat = M
    Gamma = a.prod(dim=0) if vector_decay else a.prod()
    return S_hat, Gamma


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stage_out(t, backend, device):
    """Move a tensor to whatever form the backend can transport. gloo's CUDA-tensor
    support for point-to-point send/recv is inconsistent across versions, so when we're
    on gloo we explicitly stage through host memory ourselves rather than relying on it."""
    return t.cpu() if backend == "gloo" and device.type == "cuda" else t


def stage_in(t, backend, device):
    return t.to(device) if backend == "gloo" and device.type == "cuda" else t


def bench_ring(S_hat, Gamma, rank, world_size, device, dtype, vector_decay, iters, backend):
    d_k, d_v = S_hat.shape
    times = []
    S_hat_c = stage_out(S_hat.contiguous(), backend, device)
    Gamma_c = stage_out(Gamma.contiguous(), backend, device)
    for _ in range(iters):
        dist.barrier()
        sync(device)
        t0 = time.perf_counter()
        if rank == 0:
            incoming = torch.zeros_like(S_hat_c)
        else:
            incoming = torch.zeros_like(S_hat_c)
            dist.recv(incoming, src=rank - 1)
        decay = Gamma_c.unsqueeze(-1) if vector_decay else Gamma_c
        outgoing = (decay * incoming + S_hat_c).contiguous()
        if rank < world_size - 1:
            dist.send(outgoing, dst=rank + 1)
        sync(device)
        times.append(time.perf_counter() - t0)
    return times, stage_in(incoming, backend, device)


def bench_allgather(S_hat, Gamma, rank, world_size, device, dtype, vector_decay, iters, backend):
    d_k, d_v = S_hat.shape
    times = []
    S_hat_c = stage_out(S_hat.contiguous(), backend, device)
    Gamma_c = stage_out(Gamma.contiguous(), backend, device)
    comm_device = S_hat_c.device
    for _ in range(iters):
        dist.barrier()
        sync(device)
        t0 = time.perf_counter()
        S_hat_all = [torch.zeros_like(S_hat_c) for _ in range(world_size)]
        dist.all_gather(S_hat_all, S_hat_c)
        Gamma_all = [torch.zeros_like(Gamma_c) for _ in range(world_size)]
        dist.all_gather(Gamma_all, Gamma_c)

        S_hat_stack = torch.stack(S_hat_all).to(device)
        Gamma_stack = torch.stack(Gamma_all).to(device)
        logGamma = torch.log(Gamma_stack.float())
        zero_shape = (1, d_k) if vector_decay else (1,)
        cum = torch.cat([torch.zeros(zero_shape, device=device), torch.cumsum(logGamma, dim=0)], dim=0)
        if rank == 0:
            incoming = torch.zeros(d_k, d_v, device=device, dtype=dtype)
        else:
            if vector_decay:
                w = torch.exp(cum[rank].unsqueeze(0) - cum[1:rank + 1]).to(dtype)
                incoming = (w.unsqueeze(-1) * S_hat_stack[:rank]).sum(dim=0)
            else:
                w = torch.exp(cum[rank] - cum[1:rank + 1]).to(dtype)
                incoming = (w.view(-1, 1, 1) * S_hat_stack[:rank]).sum(dim=0)
        sync(device)
        times.append(time.perf_counter() - t0)
    return times, incoming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-len", type=int, default=4096)
    parser.add_argument("--d-k", type=int, default=128)
    parser.add_argument("--d-v", type=int, default=128)
    parser.add_argument("--vector-decay", action="store_true",
                         help="GLA-style per-row decay instead of Mamba2-SSD scalar decay")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    n_gpus = torch.cuda.device_count()
    use_nccl = n_gpus > 0 and n_gpus >= world_size
    backend = "nccl" if use_nccl else "gloo"
    if n_gpus > 0:
        device = torch.device("cuda", local_rank if use_nccl else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if rank == 0:
        if use_nccl:
            mode = "NCCL, 1 GPU per rank (bandwidth-representative)"
        elif n_gpus > 0:
            mode = f"gloo, {world_size} ranks sharing {n_gpus} GPU(s) (single-GPU emulation, see header)"
        else:
            mode = "gloo, CPU only (no GPU visible -- smoke-test mode)"
        print(f"[rank0] backend={backend} mode={mode} world_size={world_size} "
              f"visible_gpus={n_gpus} chunk_len={args.chunk_len} d_k={args.d_k} d_v={args.d_v} "
              f"dtype={args.dtype} vector_decay={args.vector_decay}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    S_hat, Gamma = build_chunk(rank, args.chunk_len, args.d_k, args.d_v, args.vector_decay, dtype, device)

    # warmup, and a correctness check along the way: ring- vs all_gather-reconstructed
    # incoming state must match at every rank (not just rank 0, whose incoming is trivially 0).
    _, ring_incoming = bench_ring(S_hat, Gamma, rank, world_size, device, dtype, args.vector_decay,
                                   args.warmup, backend)
    _, ag_incoming = bench_allgather(S_hat, Gamma, rank, world_size, device, dtype, args.vector_decay,
                                      args.warmup, backend)
    local_diff = torch.tensor([(ring_incoming.float() - ag_incoming.float()).abs().max().item()],
                               device=device)
    dist.all_reduce(local_diff, op=dist.ReduceOp.MAX)

    ring_times, _ = bench_ring(S_hat, Gamma, rank, world_size, device, dtype, args.vector_decay,
                                args.iters, backend)
    ag_times, _ = bench_allgather(S_hat, Gamma, rank, world_size, device, dtype, args.vector_decay,
                                   args.iters, backend)

    ring_ms = sorted(ring_times)[len(ring_times) // 2] * 1e3
    ag_ms = sorted(ag_times)[len(ag_times) // 2] * 1e3

    ring_t = torch.tensor([ring_ms], device=device)
    ag_t = torch.tensor([ag_ms], device=device)
    dist.all_reduce(ring_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(ag_t, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"[correctness] max abs diff, ring- vs all_gather-reconstructed incoming state "
              f"(max over all {world_size} ranks) = {local_diff.item():.3e}")
        print(f"[timing] world_size={world_size}: median ring send/recv time "
              f"(max over ranks) = {ring_t.item():.3f} ms")
        print(f"[timing] world_size={world_size}: median all_gather+local-reconstruct time "
              f"(max over ranks) = {ag_t.item():.3f} ms")
        print(f"[timing] speedup (ring / all_gather) = {ring_t.item() / ag_t.item():.2f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
