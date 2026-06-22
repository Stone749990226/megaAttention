# [热身实验] standalone O_proj GEMM + one-shot NVLS AllReduce (SM90)

> **定位说明（重要）**：本文件记录的是一个 **standalone 热身实验**
> （`src/mega_attention/kernels/sm90/oproj_ar.py`），目的是熟悉 CuTe DSL +
> symmetric memory + NVLS multimem 环境，**不是核心 kernel 的状态**。
> 核心 kernel（fused FA + O_proj + NVLS AR persistent kernel）的状态见
> `docs/status/STATUS.md` 与 `README.md`。
>
> **参考价值有限**：本实验用的是**静态调度**（fork 自 CUTLASS persistent dense
> GEMM，固定 tile 调度 + per-batch 跨 rank flag），而核心 kernel 是**动态 task
> 调度的 persistent kernel**（FA→O_proj ready queue、单-owner 动态 AR claim）。
> 两者形式差异很大，这里的 per-batch handshake 调优结论不能直接套用到核心
> kernel 的 AR stage；仅作为"一次性 multimem AR + symmetric memory 能跑通且
> 数值正确"的早期验证留存。

Implements the O_proj + NVLS AllReduce part of the fused-kernel plan on 8×H200
with CuTe DSL (`nvidia-cutlass-dsl` 4.5.2).

## Files
- `src/mega_attention/kernels/sm90/oproj_ar.py` — forked from CUTLASS `hopper/dense_gemm_persistent.py`
  (persistent warp-specialized WGMMA GEMM). Added:
  - a 3rd **comm warp group** (DMA=wg0, MMA=wg1, comm=wg2; 384 threads, 1 CTA/SM);
  - symmetric-memory params: local `out`, multicast `c_mc`/`flag_mc` built in-JIT via
    `cute.make_ptr(handle.multicast_ptr, gmem)` + `make_tensor`;
  - producer side: tiles are TMA-stored into the **symmetric C** as before, but the
    cross-rank handshake is now **per-batch** (G = `comm_batch_tiles`, default 8): only
    at the end of each G-tile batch (or the SM's last tile) does the epi-store warp
    `cp_async_bulk_wait_group(0)`-drain that batch's stores and bump the batch flag once
    via `multimem_red_add1(flag_mc+batch_slot, sys, release)`. Rank-independent
    `batch_slot = sm_id*max_batches_per_sm + (p//G)`, p = local execution index.
  - comm side: walks the *identical* persistent schedule; at each batch start
    (`comm_p % G == 0`) `spin_lock_atom_cas_acquire_wait` on `flag[batch_slot]==W` +
    one comm-barrier, then per tile `multimem_ld_reduce_8xbf16(C_mc)` (NVSwitch sums 8
    ranks) → `multimem_st_4xb32` in place. G=1 reproduces the old per-tile behaviour.
- `src/mega_attention/reference/oproj_ar.py` — fp32 shadow: `O_local@W_o → all_reduce`.
- `benchmarks/bench_oproj_ar.py` — `torch.distributed._symmetric_memory` init, multicast ptrs, compile/run,
  numeric check vs fp32 ref, timing.

## Run
    torchrun --nproc_per_node=8 benchmarks/bench_oproj_ar.py --iters 30 --warmup 10

## Results (8×H200, M=8192 K=2048 N=7168, bf16; 3584 tiles = 64×56, grid=132)
- **Correctness: ALL PASS** at every G (max_abs 0.0078 vs ref |max| 1.57, ~0.5%, mean_abs 3.7e-4).
- **Per-batch handshake is a clear win.** Fused GEMM+AR wall-clock (`benchmarks/bench_oproj_ar.py`, 30 iters):

  | G | num_batch_slots | fused (ms) | exposed-AR (ms) | speedup vs un-fused baseline |
  |---|---|---|---|---|
  | 1 (per-tile, old) | 3696 | 0.7249 | 0.4155 | 1.128× |
  | **4** | 924 | **0.5596** | **0.2486** | **1.465×** |
  | 8 (default) | 528 | 0.5759 | 0.2651 | 1.421× |
  | 16 | 264 | 0.6522 | 0.3415 | 1.254× |
  | 28 (single batch) | 132 | 0.7903 | 0.4796 | 1.038× |

  Logs: `/myworkspace/log/megaattn_batch_G*.log`. un-fused baseline ≈ 0.817 ms,
  torch GEMM-only ≈ 0.31 ms, NVLS all_reduce-only ≈ 0.51 ms.

- **Inflection exactly as planned.** Batching cuts the 3584 serial cross-rank
  round-trips to ~3584/G; exposed-AR drops ~40% (0.42→0.25 ms) at G=4–8. G=28
  (one batch/SM) regresses *below* G=1 — comm wg idles waiting for the whole SM's
  GEMM, killing GEMM↔comm overlap (the documented failure mode). Sweet spot G∈[4,8];
  G=4 edged out G=8 here (within run-to-run noise), default kept at 8 per plan.

## Done / next steps
1. ✅ Per-tile → per-batch cross-rank flag + `cp_async_bulk_wait_group(0)` drain (this change).
2. **Not yet done:** batch-internal "all-`ld_reduce` → all-`st`" to raise in-flight multimem
   (report pt 4); deferred to avoid register spill — current comm side still does per-tile
   `ld→st`. Try with a bumped `comm_register_requirement` if more AR throughput is wanted.
3. Optional ncu re-profile (worker rank + SourceCounters, per `oproj-ar-ncu-gotchas`) to
   confirm the `:910/:922` sync-stall share dropped and multimem in-flight rose.

The mechanism (fusion + one-shot multimem AR + symmetric memory + per-batch rank-independent
flags) is verified correct and now beats the un-fused baseline by ~1.42–1.47× at G=4–8.
