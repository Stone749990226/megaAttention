# 改造方案:批量化 NVLS AllReduce 跨卡屏障(per-tile → per-batch)

## Context(为什么改)

ncu 报告(`profile/oproj_ar_sm90_SUMMARY.md` + `s2_rank4/REPORT.md`)的结论:
`OProjARFusedKernelSM90` **不是 compute/HBM bound,而是 cross-rank AllReduce 同步路径 bound**
(占 ~53% 的 stall:`:922` 命名屏障 27% / `:930` multimem 13% / `:910` spin-lock 5% / `:847`+`:949` membar 5%;
GEMM 仅 22%,HBM 仅 0.06% 峰值带宽)。

根因量级:配置 `M=8192 K=896 N=7168`,tile `(128,128)` → `64×56=3584` 个 tile。
当前实现 **每个 tile 都做一次完整跨卡握手**,即 **3584 次串行 NVLink round-trip**。
`3584 × ~0.7µs ≈ 2.5ms`,几乎精确等于实测的 **2.62ms**(非融合基线只要 0.65ms,慢 ~4×)。

### overlap 机制与"长杆"在哪(讨论澄清)

这是 warp-specialized 持久 kernel:一个 CTA 里 3 个 warp group **同时常驻同一 SM**,各自独立走同一份 tile 调度,
靠 flag 解耦:

- **DMA wg**(wg0):TMA load A/B
- **MMA wg**(wg1):WGMMA + epilogue TMA-store 进对称 C + 升 flag。升完 flag **不等 comm**,立刻算下一个 tile。
- **comm wg**(wg2):spin 等 flag → `multimem_ld_reduce` + `multimem_st`(NVSwitch 求和 + 广播回 C 原地)。

C 是完整 `[M,N]`,每个 tile 各占一块、MMA 只写一次、comm 在其后读+写一次 → **comm 对 MMA 无反压**,MMA 可一路跑到底。

**关键事实**:GEMM 全程 ~0.147ms,comm ~2.5ms。**MMA wg 早早算完所有 tile,真正的长杆是 comm wg。
GEMM 已经 100% 藏在 comm 底下**。因此本方案的收益**不是"改善 GEMM overlap"(已饱和),而是把 comm 这条 ~2.5ms 的临界路径本身砍短**:

1. comm wg 是顺序执行的:`spin(tile)→barrier→ld_reduce/st→下一个`,3584 次串行跨卡 spin 首尾相接 ≈ 2.5ms。
   批量化把 **跨卡 spin 次数 3584 → 3584/G**(大头,spin 延迟主导)。
2. 批内 G 个 tile 的 `ld_reduce` 一起发射 → **在途请求变多**,multimem 吞吐上来(报告第 4 点);当前每 tile 仅 2 个在途。
3. 顺带去掉 per-tile 的 `cp_async_bulk_wait_group(0)` 全排空(`:843`),改每 batch 排空一次(报告第 2 点)。

### 为什么仍要保留 GEMM↔comm overlap(G 不能一把梭到 1 个 batch)

**多个 batch 是 overlap 的前提,必须保留**:

- comm wg 处理 batch k 时,MMA wg 已跑在 batch k+1.. 前面 → 当 comm 做完 batch k 的 reduce 去 spin batch k+1 时,
  **k+1 的 flag 多半早已置位,spin 立即返回 → 跨卡 spin 延迟被前一批的 reduce 工作藏掉**。这是真正的 overlap 收益来源。
- 若 G 取到该 SM 的全部 tile 数(单 batch,G=28):comm wg 只有一个 flag,必须**干等 MMA 把全部 tile 算完且全 rank 就绪**
  才能开始,期间 comm 空转 → 退化成 "GEMM 全done → comm 全做",既丢了 pipelining、又把 0.147ms 暴露出来。

所以目标是**中间粒度**:G 大到能把每 batch 的固定开销(spin+barrier+drain)摊薄、并多发在途 multimem;
又小到留下 `28/G` 个 batch 让 comm wg 始终有活干、且后一批的 spin 延迟能藏在前一批 reduce 后面。
G=28(单 batch)仅作上界 sanity 对照,预期不优甚至变差。

**不做**:comm 调度解耦(就绪队列)、动 occupancy / GEMM tile / stage(报告明确说别碰)。

## 关键事实(已核实)

- 持久调度 `StaticPersistentTileScheduler`:SM 线性 id `s` 顺序处理 linear tile `s, s+grid, s+2·grid, ...`;
  `tile_sched.num_tiles_executed` 给出该 SM 的本地 tile 序号 `p = 0,1,2,...`。
  所有 rank 用 **相同 grid**(`_compute_grid` → `get_grid_shape`,cluster=(1,1) 时 `grid = min(max_active_clusters, num_tiles)`,
  H200 上 = `min(132, 3584) = 132`),故 SM `s` 在每个 rank 上处理 **相同的 linear tile** → flag slot 天然 rank 无关。
- 当前 flag slot = `tile_id = m_idx*num_n_tiles + n_idx`(全局 tile id,3584 个),
  生产侧 `:848 multimem_red_add1(+1, release)`,comm 侧 `:910 spin_lock_atom_cas_acquire_wait(==world_size, reset=0)`。
- `benchmarks/bench_oproj_ar.py:122` flag 分配 `num_tiles + num_sms`(per-tile 区 + per-SM 完成屏障区);`max_active_clusters` 已在 `benchmarks/bench_oproj_ar.py:139`。
- comm 侧每 tile 只发 `rest_m=2` 个 128-bit multimem(`m_shard=128/8=16`,`m_shard//8=2`)→ 在途太少。
- 涉及文件:`megaattn_oproj_ar_sm90.py`(kernel)、`benchmarks/bench_oproj_ar.py`(flag 分配 + 传参)。
- 复用 helper(无需新写):`utils.distributed.{multimem_red_add1, spin_lock_atom_cas_acquire_wait,
  multimem_ld_reduce_8xbf16, multimem_st_4xb32}`;SM 线性 id 算法已在 `:943-:945` 出现。

## 批次 / flag 设计

- 新增可调参数 `comm_batch_tiles`(= G,默认 8)作为 `OProjARFusedKernelSM90.__init__` 入参,存 `self.comm_batch_tiles`;
  benchmarks/bench_oproj_ar.py 暴露 `--comm-batch-tiles` 便于扫参。G=1 退化为现状(数值回归对照)。
- 批次 id(rank 无关)用 **SM 本地批次** 编号:
  - SM 线性 id:`sm_id = bidx + bidy*gdx + bidz*gdx*gdy`(复用 `:943-:945`)。
  - 本地批次 `bk = p // G`(`p = num_tiles_executed`)。
  - `max_batches_per_sm = cdiv(cdiv(num_tiles, grid_size), G)`(G=8 时 = `cdiv(28,8)=4`)。
  - **batch flag slot** = `sm_id * max_batches_per_sm + bk`(在生产侧 / comm 侧 / 完成屏障三处公式必须一致)。
- flag 缓冲新布局(`benchmarks/bench_oproj_ar.py`):
  - `[0, num_batch_slots)` 批次就绪区,`num_batch_slots = grid_size * max_batches_per_sm`;
  - `[num_batch_slots, num_batch_slots + num_sms)` 仍是 per-SM 完成屏障区。
  - G=8 时 slot 数 ≈ `132*4 + 132 = 660`,远小于 `3584 + 132`。

## 实施步骤

### 1. `megaattn_oproj_ar_sm90.py` — 构造参数与常量
- `__init__` 增参 `comm_batch_tiles: int = 8`,存 `self.comm_batch_tiles`。
- kernel 内由 `cute.arch.grid_dim()` 现算 `grid_size`,结合 `num_m_tiles*num_n_tiles` 算
  `max_batches_per_sm = cdiv(cdiv(num_tiles, grid_size), G)`(constexpr);定义 `num_batch_slots = grid_size * max_batches_per_sm`。

### 2. 生产侧(MMA warp group,约 `:833-:855`)
把"每 tile drain + bump flag"改为"**每 batch 末尾一次**":
- 每 tile 的 TMA-store 进对称 C 保持不变(`:795-:831`)。
- 取 `p = tile_sched.num_tiles_executed`;在循环尾部 `advance_to_next_work()` + `get_current_work()` 后,
  用下一 tile 的 `is_valid_tile` 判 `is_last`。**batch flush 条件** = `((p+1) % G == 0) or is_last`。
- 满足 flush 时(且仅 `warp_idx == epi_store_warp_id`):
  - `cute.arch.cp_async_bulk_wait_group(0, read=False)`(一次排空整批 store);
  - `elect_one()` + `fence_proxy("alias")` + `multimem_red_add1(flag_mc.iterator + batch_slot, scope="sys", order="release")`,
    `batch_slot = sm_id * max_batches_per_sm + (p // G)`。
- 注意:flush 判定要在 advance 之后才能拿到 `is_last`,但 `batch_slot` 用的是 **当前刚产出 tile 的 p**;
  实现上在 advance 前先存 `p_cur = num_tiles_executed`,advance/peek 后用 `p_cur` 算 slot 与周期条件。

### 3. comm warp group(约 `:902-:933`)重构为"外层批 / 内层 tile"
- 取 `comm_p = comm_tile_sched.num_tiles_executed`。**batch 起点**(`comm_p % G == 0`)时:
  - `comm_tidx == 0`:`spin_lock_atom_cas_acquire_wait(flag.iterator + batch_slot, expected_val=world_size, reset_val=0, scope="sys")`,
    `batch_slot = sm_id * max_batches_per_sm + (comm_p // G)`;
  - `comm_sync_barrier.arrive_and_wait()`(**每 batch 一次**,不再每 tile)。
- 然后对该 tile 做原有 shard RS+AG(`shard_m_idx`、`thr_copy_cmc.partition_S`、`rest_m` 循环不变),advance,直到 `is_valid_tile == False`。
  尾批自然靠 `is_valid_tile` 守卫(producer 已在末 tile flush 了该 batch 的 flag)。
- **批内多发在途 multimem(报告第 4 点)**:把当前 batch 内 G 个 tile × `rest_m` 个 `multimem_ld_reduce_8xbf16`
  **先全部发射、结果暂存到寄存器数组**,再统一 `multimem_st_4xb32` 写回,以增大同时在途 ld_reduce 数。
  - 一期先实现"**批内 ld 全发 → 全 st**"的简单形式;若寄存器压力大导致 spill,退化为 per-tile `ld→st`,但**仍保留 batch 级 flag/屏障粒度**(报告第 1/2 点照拿)。
  - `comm_register_requirement`(`:111`,当前 40)可能要上调以容纳批内在途结果,扫参时一并试。

### 4. 完成屏障(`:935-:959`)
- 基址 `num_tiles_total = num_m_tiles*num_n_tiles` 改为 `num_batch_slots`;其余不变(`done_off = num_batch_slots + sm_id`)。

### 5. `benchmarks/bench_oproj_ar.py`
- flag 分配(`:119-:123`):由 `num_tiles + num_sms` 改为 `num_batch_slots + num_sms`:
  - `grid_size = min(max_active_clusters, num_tiles)`(`max_active_clusters` 已在 `:139`,但分配在其之前 → 把该行上移或就地重算);
  - `max_batches_per_sm = cdiv(cdiv(num_tiles, grid_size), G)`;`num_batch_slots = grid_size * max_batches_per_sm`。
- 构造 kernel 时传 `comm_batch_tiles=G`(默认 8),暴露 CLI `--comm-batch-tiles`。
- 更新 flag layout 注释(`:119-:121`)。

## 正确性要点(务必保证)

- `batch_slot` 公式在生产侧 / comm 侧 / 完成屏障基址 **三处一致**,且仅依赖 `sm_id、p//G、max_batches_per_sm`(rank 无关)。
- 生产侧必须在 batch 内 **所有 tile 的 store 落地后**(drain 在 bump 之前)才升 flag;comm 侧必须在处理 batch 前 **等到 flag==world_size**。
  release/acquire 顺序保持现状(`:847` fence+release / `:910` acquire)。
- 尾批(< G)生产侧靠 `is_last` flush、comm 侧靠 `is_valid_tile` 守卫,避免 flag 永远等不到或 slot 越界。
- `grid_size`、`max_batches_per_sm` 在 host(benchmarks/bench_oproj_ar.py)与 device(kernel)必须用**同一公式**算出同一值。
- `BLOCK_M % world_size == 0`、`(BLOCK_M/W) % 8 == 0` 约束不变(shard 逻辑未动)。

## 验证(end-to-end)

按 memory 的 env / log / ncu gotchas:

1. **正确性回归**:先以 **G=1** 跑,确认与现状数值一致且 `correctness: ALL PASS`(对 fp32 shadow,max_abs ~0.5%):
   `cd /myworkspace/megaAttention && python -m torch.distributed.run --nproc_per_node=8 benchmarks/bench_oproj_ar.py --iters 30 --warmup 10 --comm-batch-tiles 1`
   输出 `2>&1 | tee /myworkspace/log/megaattn_batch_G1_$(date +%Y%m%d_%H%M%S).log`。
2. **性能扫参**:用 `benchmarks/bench_oproj_ar.py` 墙钟(**不要**用 ncu per-rank duration——按 `oproj-ar-ncu-gotchas` 那是假值)。
   扫 `--comm-batch-tiles {1,4,8,16,28}`,记录融合 GEMM+AR 墙钟 vs 基线 0.65ms,找拐点;
   预期 G∈[4,16] 出现明显下降,G=28(单 batch)因丢 overlap 回升。每个 G 单独存 log。
3. **复测瓶颈**(可选):`ncu_oproj_ar.sh` 取 worker rank + SourceCounters,确认 `:910/:922` 同步 stall 占比下降、multimem 在途上升;
   按 gotcha 拆 `SECTIONS_EXTRA` 避免 pass-37 SIGKILL。
4. 更新 `STATUS.md` 的 Results / next-steps;有新结论写入 memory(更新 `oproj-ar-ncu-gotchas` 或新增)。

## 风险 / 回退

- 批内"全 ld→全 st"寄存器压力导致 spill/occupancy 跌 → 退化为 per-tile `ld→st`,仅保留 batch 级 flag/屏障粒度(仍覆盖报告第 1/2 点)。
- G 过大丢 overlap(comm 干等 GEMM)→ 扫参取拐点;G=1 始终是数值回退路径。
- 改动局限在 `megaattn_oproj_ar_sm90.py` 的生产侧 epilogue + comm warp group 两段,以及 `benchmarks/bench_oproj_ar.py` 的 flag 分配/传参,blast radius 小。
