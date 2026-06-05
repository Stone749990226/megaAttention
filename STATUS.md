# megaattn — fused O_proj GEMM + one-shot NVLS AllReduce (SM90) — STATUS

Implements `plan.md` on 8×H200 with CuTe DSL (`nvidia-cutlass-dsl` 4.5.2, `/usr/bin/python`).

## Files
- `megaattn_oproj_ar_sm90.py` — forked from CUTLASS `hopper/dense_gemm_persistent.py`
  (persistent warp-specialized WGMMA GEMM). Added:
  - a 3rd **comm warp group** (DMA=wg0, MMA=wg1, comm=wg2; 384 threads, 1 CTA/SM);
  - symmetric-memory params: local `out`, multicast `c_mc`/`flag_mc` built in-JIT via
    `cute.make_ptr(handle.multicast_ptr, gmem)` + `make_tensor`;
  - producer side: after each tile's partial is TMA-stored into the **symmetric C**,
    `multimem_red_add1(flag_mc+tile_id, sys, release)` marks the tile ready (per-tile,
    rank-independent `tile_id = m_idx*num_n_tiles + n_idx`);
  - comm side: walks the *identical* persistent schedule, `spin_lock_atom_cas_acquire_wait`
    on `flag[tile_id]==W`, then `multimem_ld_reduce_8xbf16(C_mc)` (NVSwitch sums 8 ranks)
    → local 32-bit store into `out`.
- `reference.py` — fp32 shadow: `O_local@W_o → all_reduce`.
- `bench.py` — `torch.distributed._symmetric_memory` init, multicast ptrs, compile/run,
  numeric check vs fp32 ref, timing.

## Run
    /usr/bin/python -m torch.distributed.run --nproc_per_node=8 megaattn/bench.py --iters 30 --warmup 10
(must use `/usr/bin/python` — that's where cutlass-dsl is installed; the `torchrun` on
PATH is vllm-venv's and lacks it.)

## Results (8×H200, M=8192 K=896 N=7168, bf16)
- **Correctness: ALL PASS.** max_abs 0.0078 vs ref |max| 1.633 (~0.5%), mean_abs 3.7e-4 — bf16-accurate.
- **Performance: NOT yet a win.** fused GEMM+AR = **2.62 ms** vs un-fused baseline
  (torch GEMM 0.147 ms + NVLS all_reduce 0.504 ms = 0.65 ms). ~4× slower.

## Why it's slow (diagnosis / next steps)
1. **Per-tile cross-rank barrier** (3584 tiles): each tile does a sys-scope multicast
   `multimem_red_add1` + a CAS-acquire spin. plan §5 calls for **per-SM, per-wave**
   barriers (~132×7 ≈ 924) — batch the barrier over a wave of tiles, not per tile.
2. **`cp_async_bulk_wait_group(0)` per tile** in the epilogue drains the whole TMA-store
   pipeline every tile → serializes store/compute and likely kills GEMM/comm overlap.
   Track per-tile store completion without a full drain.
3. **Comm throughput**: 132 CTAs × 128 threads of `ld_reduce` is not yet saturating
   NVLink the way NCCL NVLS does (0.50 ms). Needs nsys/ncu to confirm overlap and tune
   the comm thread/vector mapping.
4. Confirm with nsys that comm warp group actually overlaps the next tiles' WGMMA.

The mechanism (fusion + one-shot multimem AR + symmetric memory + per-SM-correct flags)
is in place and verified correct; reaching the plan's ~20–25% target is a profiling-driven
optimization pass on items 1–4.
