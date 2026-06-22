# 核心 kernel STATUS — causal varlen prefill FA + O_proj + NVLS AllReduce fused persistent kernel

> 本文件是**核心 kernel** 的开发 living doc，落点
> `src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py`。
> 与 README 的关系：**README 是权威摘要**（模块状态表、P0–P3 分阶段验证、
> 8×H200 性能表 + err_rel），本文件**不复制**这些，只做 README 没有的那层：
> 设计文档逐章实现度对照、当前在做的事、已知风险/未决设计问题、下一步 TODO。
> 数字一律以 README 为准（避免双源漂移）。
>
> standalone 的 O_proj + NVLS AR 热身实验（静态调度，参考价值有限）见
> `docs/status/oproj_ar_experiment.md`。
>
> 设计依据：`docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md`。

当前分支：`perf/fa-tma-q-load`。最近更新：2026-06-22。

---

## 一句话状态

fused kernel 已在同一个 persistent 调度器内跑通 **real FA + real O_proj + real
NVLS AllReduce**，单卡 `tp_size=1` 与 8×H200 `tp_size=8` 两条路径都数值验证通过
（整链对 `full_chain_reference` err_rel ~4e-4–2e-3，bf16 级）。当前工作重心已从
"功能正确"转向 **FA per-tile pipeline 性能优化**。

性能/正确性数字见 README「当前状态」「分阶段验证状态」「性能」三节，不在此复制。

---

## 设计文档逐章实现度对照

状态图例：✅ 已验证 ｜ 🟢 已实现（随整链验证，但无该点的独立断言）｜
🟡 进行中 ｜ ⚪ 未实现/第一版不做 ｜ ❓ 需确认（缺直接证据）

| 设计文档章节 | 状态 | 备注 / 证据 |
| --- | --- | --- |
| 目标范围 / 基本计算 / Tile 尺寸约定 | 🟢 | 第一版 invariant（SM90、causal、varlen prefill、`q_len==k_len`）已落实 |
| FA task 与 O_scratch | ✅ | real FA 在 fused scheduler 验证（含 multi_seq），见 README P0 |
| Runtime task descriptor 与动态 varlen payload | ✅ | 动态 varlen FA tile 已验证；`row_desc` 已实现 |
| causal mask（`q_len==k_len` 完整 prompt prefill） | ✅ | `test_fa_varlen` 含 `valid_m%8≠0` 回归 |
| O_proj row tile / O_proj task identity | ✅ | real O_proj 接入 persistent dispatcher，README P1（err~0.0018） |
| 方案 B: O_proj ready queue（FA→O_proj） | 🟢 | 整链跑通即依赖此队列；无独立 ready-queue 单测断言 → 后续可补 |
| 方案 A: 64-bit ready bitset 评估 | ⚪ | 设计文档中为评估项，实际采用方案 B |
| Persistent Kernel 总体结构 / Task 调度策略 | ✅ | persistent scheduler skeleton 已验证（`test_scheduler_skeleton` 5 passed，README P2） |
| FA Mode Warp Specialization | 🟢 | 整链验证覆盖 |
| FA K/V pipeline 与 intra-wg overlap | 🟡 | **当前分支正在改**：Q 改 1-stage TMA pipeline（commit `33adf62`），删协作 load + 全 CTA barrier |
| FA 到 O_proj 的内存序 | 🟢 | 整链数值正确隐含该 happens-before 成立；无独立压力测试 |
| O_proj/AR Mode Warp Specialization / Pipeline | 🟢 | 整链验证覆盖 |
| 跨 rank ready 方式 | ✅ | symmetric C_sym multicast + 跨 rank owner 寻址，README P3 |
| 非阻塞 AR owner reduce task（单-owner 动态调度） | ✅ | 确定性 owner 映射 + owner-local u64 bitset + exactly-once/terminate，README P2 |
| NVLS AllReduce 执行方式（multimem reduce/store） | ✅ | 8×H200 multimem ld_reduce/st on C_sym multicast view，README P3 |
| Workspace 生命周期与全局同步 / local grid_sync / nvl_barrier | 🟢 | P3 用到 nvl_barrier；workspace size 由 `row_desc` 计算 |
| 长寿命 pipeline state、mbarrier 复用、mode 切换 drain 规则 | ❓ | 整链能跑通说明基本成立，但 mode 切换 drain 的边界正确性缺独立断言，**风险项**，见下 |
| 第一版不做的事情（decode/append/chunked/paged/splitkv…） | ⚪ | 按约束不实现 |

---

## 调度热路径优化记录（2026-06-22）

针对 `schedule_pick` 及其 `try_*` claim helper 的调度税做了一轮评估与改动。

**已落地（保留）**：把调度热路径上 9 处**只读**的 `atomic_add(ptr, 0, sem, scope)`
改为纯 `cute.arch.load(ptr, dtype, sem, scope)`（`try_fa` 的 counter 预检、
`try_pop_oproj` 的 head/tail、`try_claim_ar` 的 cursor 与 ready word、`publish_oproj`
的 publish 自旋、`schedule_pick` 的三个 done 计数）。语义不变（relaxed/acquire +
gpu scope 一一对应），但去掉了 RMW：不再占用 L2 atomic ALU、不再独占 cache line，多个
轮询 leader 可共享 read 同一条 L2 line。`gsync` 网格 barrier 未动。

**尝试后放弃**：idle backoff（三类 source 全 miss 时 PTX `nanosleep` 指数退避）。
原因——persistent grid 下 CTA 常驻，空转 CTA backoff 也不会把 SM 让给别人，经典 backoff
理由不成立；它唯一能削的是空转 leader 对共享 L2 的读流量，而上面的 acquire-load 已把每次
轮询变成廉价可共享 load，边际收益很小；且指数退避到 μs 级会在 pipeline fill/drain/相变
等最该快的窗口给任务拾取加延迟。

**验证（8×H200，9 个真实模型 shape，kineto self device time）**：改动前后整链 `err_rel`
逐位一致；fused 墙钟 delta 落在 run-to-run 噪声内（geomean +0.2%，双向都有），删除 backoff
后 geomean 几乎不变，反向印证 backoff 无贡献。墙钟无明显收益符合预期——调度读成本被 μs~ms
级 FA/O_proj task body 摊薄，这批 shape 是 compute-bound。改动取其**正确性零影响 + 每次
轮询更便宜 + 无墙钟下行**。要量化“更便宜的轮询”需用 ncu 看 `lts__t_sectors_op_atom*`
等 atomic 吞吐指标（比墙钟灵敏），或构造 contention-bound shape，列为后续可选项。

