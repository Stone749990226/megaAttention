"""Standalone comparison: Triton-distributed GemmARLayer at the SAME problem as
megaattn's oproj_ar fused GEMM+AllReduce.

megaattn is a K-sharded TP all-reduce: each rank holds O_local[M, local_K] and
W_o[local_K, N], computes a [M,N] partial, and the partials are summed across
ranks. local_K = 2048, world=8 -> global K = 16384, M=8192, N=7168.

Triton's gemm_ar takes the GLOBAL K and splits local_K = K // world_size, with
A[M, local_K] @ weight[N, local_K].T -> [M,N] then all_reduce. So passing
K=16384 reproduces megaattn's per-rank GEMM (A[8192,2048] @ W[7168,2048].T)
exactly, making the two directly comparable.

Run (8 ranks, single node) with the venv that has triton_dist installed. Two
env gotchas on this CUDA-13 box (see compare_gemm_ar.sh which sets them):
  * TRITON_PTXAS_PATH -> the wheel's bundled cu12.8 ptxas (its Triton rejects cu13 ptxas)
  * NVSHMEM_DISABLE_CUDA_VMM=0 -> the AR kernel needs NVLS multicast (remote_mc_ptr);
    disabling VMM makes multimem_ld_reduce illegal-access.
Launch with VENV python via `python -m torch.distributed.run` (NOT system torchrun).
"""
import os
import torch
import torch.distributed as dist

from triton_dist.utils import (initialize_distributed, finalize_distributed, perf_func,
                               nvshmem_barrier_all_on_stream)
from triton_dist.layers.nvidia import GemmARLayer


def main():
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))

    # Same global problem as megaattn (reference.shapes(8)): M, N, global-K.
    M, N, K = 8192, 7168, 2048 * WORLD_SIZE  # local_K = K // W = 2048
    iters = int(os.environ.get("ITERS", "30"))
    warmup = int(os.environ.get("WARMUP", "10"))
    num_comm_sms = int(os.environ.get("NUM_COMM_SMS", "16"))
    dtype = torch.bfloat16

    tp_group = initialize_distributed(42)
    rank = tp_group.rank()
    assert K % WORLD_SIZE == 0
    local_K = K // WORLD_SIZE

    if rank == 0:
        print(f"[triton-gemm_ar] M={M} N={N} K={K} (local_K={local_K}) W={WORLD_SIZE} "
              f"NUM_COMM_SMS={num_comm_sms} dtype={dtype}")

    device = torch.cuda.current_device()
    # per-rank inputs differ so the all_reduce is actually exercised
    g = torch.Generator(device=device).manual_seed(1234 + rank)
    A = (torch.randn(M, local_K, device=device, dtype=dtype, generator=g) * 0.1)
    weight = (torch.randn(N, local_K, device=device, dtype=dtype, generator=g) * (local_K ** -0.5))

    op = GemmARLayer(tp_group, M, N, K, dtype, dtype, LOCAL_WORLD_SIZE,
                     persistent=True, use_ll_kernel=False, copy_to_local=True,
                     NUM_COMM_SMS=num_comm_sms)

    def torch_gemm_ar(A, weight):
        out = torch.matmul(A, weight.T)
        dist.all_reduce(out, group=tp_group)
        return out

    # ---- correctness ----
    ref = torch_gemm_ar(A, weight)
    got = op.forward(A, weight, None)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    max_abs = (got.float() - ref.float()).abs().max().item()
    ref_absmax = ref.float().abs().max().item()
    ok = max_abs <= 6e-2 * max(ref_absmax, 1.0) + 6e-2
    flags = [None] * WORLD_SIZE
    dist.all_gather_object(flags, ok)
    if rank == 0:
        print(f"[triton-gemm_ar] correctness: max_abs={max_abs:.4f} ref|max|={ref_absmax:.3f} "
              f"-> {'ALL PASS' if all(flags) else 'FAIL'}")

    # ---- perf ----
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    _, fused_ms = perf_func(lambda: op.forward(A, weight, None), iters=iters, warmup_iters=warmup)
    torch.cuda.synchronize()
    _, gemm_ms = perf_func(lambda: torch.matmul(A, weight.T), iters=iters, warmup_iters=warmup)
    torch.cuda.synchronize()
    ar_in = torch.matmul(A, weight.T)
    _, torchar_ms = perf_func(lambda: dist.all_reduce(ar_in, group=tp_group), iters=iters, warmup_iters=warmup)
    torch.cuda.synchronize()
    _, torchfused_ms = perf_func(lambda: torch_gemm_ar(A, weight), iters=iters, warmup_iters=warmup)

    if rank == 0:
        print("\n==================== Triton-distributed GemmARLayer ====================")
        print(f"  triton fused GEMM+AR        : {fused_ms:.4f} ms")
        print(f"  torch GEMM only             : {gemm_ms:.4f} ms")
        print(f"  torch all_reduce only (NCCL): {torchar_ms:.4f} ms")
        print(f"  torch GEMM + NCCL AR (unfused): {torchfused_ms:.4f} ms")
        print(f"  triton exposed-AR (fused-gemm): {fused_ms - gemm_ms:.4f} ms")
        print("========================================================================")

    op.finalize()
    finalize_distributed()


if __name__ == "__main__":
    main()
