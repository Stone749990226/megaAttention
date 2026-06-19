# Launch 启发式：FA/O_proj/AR role 软配比 + super_group_n_tiles 自适应（A 类）

> 本文档是 `causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md` 的子设计，只在
> 其 invariant 之上增加 **host 侧 launch 配置层**，不改 kernel 的任何算法 invariant、
> 数据流或同步协议。范围严格限定第一版：Hopper SM90、causal、varlen full prompt
> prefill、`q_len == k_len`、FA + O_proj + NVLS AllReduce fused persistent kernel。

## 1. 背景与动机

当前 8×H200 benchmark（12 组 shape）显示 fused vs (FlashAttention + GEMM + NVLS) 基线
的加速比随 shape 波动很大：平衡区大赢（1.3–1.4×），超长单序列回退到 <1×。

对 causal、`q_len == k_len` 推导 FA 与 O_proj 的 MAC 比（`H_local` 与 `128²·D` 在两边
约掉）：

```
FA_macs    ≈ 2 · Σ_t (m_block[t] + 1)        # ×2 = QK + PV，每 tile causal K-block 数 = m_block+1
OPROJ_macs ≈ num_row_tiles · num_out_n_tiles
r          = FA_macs / OPROJ_macs            # 单序列退化为 r ≈ L / hidden
```

对照 benchmark：**平衡区（r≈1）几乎都大赢，FA 重区（r 大）表现差**。但存在反例——
`[4096,4096]hid4096` 与 `[16384]H16hid4096` 的 `r` 都是 4，前者赢 1.41×、后者输 0.95×。
差别在绝对序列长度与"单一长 causal 序列"。

由此把问题分成两类：

- **A 类（本文档范围）**：中等规模、O_proj 偏重（DeepSeek）、多序列。瓶颈是重叠阶段的
  FA:O_proj:AR 配比与 O_proj/AR 同步粒度（sg）。可用 host 侧 launch 配置解决，不碰 kernel
  invariant。
- **B 类（本文档不做，留作第二步）**：超长单序列（16K/32K）的 <1× 回退。瓶颈大概率是
  fused FA tile 本身在 SMEM overlay 约束下（kv_stages=2、背 O_proj 寄存器/SMEM 占用）干不过
  专用 FlashAttention，属于 kernel 内部 pipeline 问题，launch 启发式治不了。

## 2. 现状（被替换/扩展的部分）

- **role 是软偏好，不是硬分区**：`cls = bidx % 6`（[fused_fa_oproj_ar.py:668](../../src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py)），
  `schedule_pick` 按 cls 给偏好顺序，固定 4:1:1（FA:OPROJ:AR）。每个 CTA 都
  fall-through，所以配比只影响**重叠阶段稳态平衡**与 **mode 切换抖动**，不影响"会不会闲置"。
- **sg 固定**：`super_group_n_tiles` 是 compile-time 常量，benchmark 固定 sg=4。
- **可算量**：`RowDescMeta.m_block`、`num_row_tiles`、`oproj_task_counts` 已能在 host 侧
  精确给出 `r` 所需的全部输入。

## 3. 设计

### 3.1 算力特征 `r`（host 侧精确）

新增 `estimate_work_ratio(meta, H_local, hidden, N_TILE) -> float`：

```
fa_macs    = 2 * sum(meta.m_block + 1)          # int 求和，精确
num_out    = cdiv(hidden, N_TILE)
oproj_macs = meta.num_row_tiles * num_out
r          = fa_macs / oproj_macs
```

`r` 仅作分桶特征，不直接当配比（MAC 比 ≠ 时间比：FA 带 softmax，GEMM 近峰值）。

### 3.2 role 软配比机制（kernel 改动）

把固定的 `cls = bidx % 6` + 硬编码偏好表，替换为 launch 常量 `(w_fa, w_oproj, w_ar)`
驱动：

```
m = w_fa + w_oproj + w_ar          # constexpr，host 传入
k = bidx % m
k <  w_fa            -> 偏好 (FA, OPROJ, AR)
k <  w_fa + w_oproj  -> 偏好 (OPROJ, AR, FA)
否则                 -> 偏好 (AR, FA, OPROJ)
```

- 保留 fall-through（任何 CTA 偏好队列空了仍去抢其它 task）。
- `bidx % m` 天然跨 SM 交错，role 在物理 SM 上均匀分布。
- NVLS AR owner 由 `slot_id % tp_size` 决定，与 role 偏好正交，**不受影响**。
- `tp == 1`：`w_ar = 0`（AR 是 identity，给它 CTA 偏好没意义，并入 FA 偏好）。
  `tp > 1`：`w_ar` 取标定的小份额。
- 兼容性：当 `(w_fa,w_oproj,w_ar) = (4,1,1)` 时必须与现有 `bidx % 6` 行为等价，作为回归基线。

只改 `schedule_pick` 的偏好选择与 kernel 入口的 `cls` 计算；try_fa / try_pop_oproj /
try_claim_ar、队列协议、mbarrier、drain 规则全部不变。

### 3.3 sg 变体选择（compile-time）

sg 是编译期常量，做法是**预编译 {2,4,8} 三个变体，host 按 shape 分派**：

- `r` 小 / 平衡、O_proj 重 → sg 小（2）：O_proj 切细、AR 早发布、铺满更多 CTA。
- `r` 大（FA 重）→ sg 大（8）：O_proj 是细尾巴，减少 AR publish / 调度开销。
- `tp > 1` 偏向更大 sg：砍跨 rank 握手次数。
- 夹住 `num_super_groups`（= `cdiv(num_out, sg)`）不低于下限，保证 spread。

driver 维护一个 `{sg: compiled_kernel}` 变体缓存，按 `choose_launch_config` 返回的 sg
取用。

### 3.4 启发式本体 = `r` 粗桶查表

**不靠闭式公式定常数。** `r` 给分桶轴，每桶 `(w_fa, w_oproj, w_ar, sg)` 由一次性 H200
sweep 标定后写成**粗分段表**（起步 3 档，不够再细化）：

```
# tp == 1 占位结构（具体数值由 §4 标定填入）
r < R_LO            -> (w_fa, w_oproj, 0, sg)   # O_proj 重 / 平衡
R_LO <= r < R_HI    -> (w_fa, w_oproj, 0, sg)   # 中间
r >= R_HI           -> (w_fa, w_oproj, 0, sg)   # FA 重
```

tp>1 单独一张表（含非零 `w_ar`，倾向更大 sg）。阈值 `R_LO/R_HI` 与每桶取值都来自标定，
不在本文档预设具体数字。

### 3.5 模块与接口

```
src/mega_attention/metadata/launch_heuristic.py        # 新增（纯 host，可 CPU 单测）
    estimate_work_ratio(meta, H_local, hidden, N_TILE) -> float
    @dataclass LaunchConfig: w_fa:int, w_oproj:int, w_ar:int, sg:int
    choose_launch_config(meta, H_local, D, hidden, N_TILE, tp_size, num_sms) -> LaunchConfig
    # 内部持有标定粗桶表（tp==1 与 tp>1 两份）

src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py    # 改
    FusedFaOprojAr.__init__ 接收 (w_fa, w_oproj, w_ar)（默认 4,1,1，逐位等价旧 bidx%6）
    kernel(): cls 计算 + schedule_pick 偏好选择改为权重驱动

benchmarks/bench_fused_fa_oproj_ar.py                   # 改
    支持显式传 (w_fa,w_oproj,w_ar,sg) 做 sweep；新增 --auto 走 choose_launch_config
```

## 4. 验证与标定计划（H200 实测，已确认 8×H200）

> CLAUDE.md：性能结论必须基于实际 Hopper 环境。本环境已确认 8×H200。

1. **管线先行（无启发式）**：实现可调 `(w_fa,w_oproj,w_ar)` + sg 变体的 host+kernel 管线，
   手动传参。先验证 `(4,1,1)+sg4` 与现有实现数值/性能等价（回归基线）。
2. **CPU 单测**：`estimate_work_ratio` 对单序列退化为 `L/hidden`、varlen 多序列与逐 tile
   暴力求和一致；`choose_launch_config` 边界（tp1/tp>1、r 跨桶、sg 夹取）正确。
3. **H200 sweep**：12 组 shape × 小网格 `(w_fa:w_oproj, sg ∈ {2,4,8})`，每组记录
   fused 时延与 vs 基线 ratio。产出：
   - 每组最优配置 → 填 §3.4 粗桶表；
   - **验证旋钮真有收益**：若 sweep 显示软配比/ sg 对 A 类基本无改善，说明瓶颈是 mode 切换
     抖动，需回到机制 2（硬保留 FA CTA）——此时停下来与用户确认，不擅自扩范围。
4. **回归确认**：`choose_launch_config` 接标定表，重跑 12 组，确认 A 类（中等/O_proj 重/
   多序列）提升、B 类（长单序列）不回退。

常用验证命令：

```bash
pytest tests/metadata                                   # 含新增 launch_heuristic 单测
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto   # 自适应分派
# sweep（标定）：显式传参网格，见 benchmarks 改动
```

## 5. 明确不做（YAGNI / 防偏航）

- 不做 B 类（长单序列 FA tile pipeline）——独立第二步。
- 不做硬分区 role（机制 2）——仅在 §4.3 测出抖动是瓶颈、且用户确认后再上。
- 不做连续插值/在线自调——粗桶查表起步。
- 不引入任何第一版范围外路径（decode/append/chunked/paged/splitKV/非 causal 等）。
- 不把启发式包装成对外稳定 API。
```
