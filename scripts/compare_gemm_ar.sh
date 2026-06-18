#!/bin/bash
# Side-by-side comparison: megaattn oproj_ar (CuTe DSL, torch symm_mem NVLS) vs
# Triton-distributed GemmARLayer (NVSHMEM NVLS), at the SAME problem
# (M=8192, N=7168, per-rank GEMM [8192,2048]@[7168,2048].T, 8 ranks).
#
# The two cannot share a process: megaattn uses CuTe DSL + torch symmetric memory;
# triton_dist uses a separate venv + a Triton fork + NVSHMEM.
# So we run each in its own env and tabulate. Logs land in /myworkspace/log.
set -u
TS=$(date +%Y%m%d_%H%M%S)
ITERS=${ITERS:-30}; WARMUP=${WARMUP:-10}; G=${G:-8}
cd /myworkspace/megaAttention

echo "===== [1/2] megaattn oproj_ar (CuTe DSL), per-batch G=$G ====="
python -m torch.distributed.run --nproc_per_node=8 benchmarks/bench_oproj_ar.py \
  --iters "$ITERS" --warmup "$WARMUP" --comm-batch-tiles "$G" \
  2>&1 | tee "/myworkspace/log/compare_megaattn_${TS}.log" | grep -E "fused GEMM\+AR|exposed-AR|speedup|correctness:"

echo
echo "===== [2/2] Triton-distributed GemmARLayer (NVSHMEM) ====="
VENV=/myworkspace/.venv-tritondist
source "$VENV/bin/activate"
export NVSHMEM_HOME="$VENV/lib/python3.12/site-packages/nvidia/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"
export TRITON_PTXAS_PATH="$VENV/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"
export NVSHMEM_DISABLE_CUDA_VMM=0 NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=lo NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_SYMMETRIC_SIZE=2000000000 NCCL_DEBUG=ERROR
ITERS=$ITERS WARMUP=$WARMUP NUM_COMM_SMS=${NUM_COMM_SMS:-16} \
  python -m torch.distributed.run --nproc_per_node=8 --rdzv_endpoint=127.0.0.1:23470 \
  benchmarks/bench_triton_gemm_ar.py \
  2>&1 | tee "/myworkspace/log/compare_triton_${TS}.log" | grep -E "fused GEMM\+AR|exposed-AR|correctness:"
