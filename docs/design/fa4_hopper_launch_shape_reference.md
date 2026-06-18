# FA4 Hopper launch shape 参考笔记

本文记录 FlashAttention-4 CuTe Hopper forward 路径中，kernel launch 的
`grid` 和 `block` shape 是如何确定的。它只作为本项目实现
`causal varlen prefill FA + O_proj + NVLS AR fused persistent kernel` 时的外部参考，
不能覆盖核心设计文档中的第一版 invariant。

对应源码主要在：

- `third_party/flash-attention/flash_attn/cute/interface.py`
- `third_party/flash-attention/flash_attn/cute/flash_fwd_sm90.py`
- `third_party/flash-attention/flash_attn/cute/tile_scheduler.py`

## 一句话结论

FA4 Hopper forward 的 launch shape 分三步确定：

```text
Python interface 按 shape / head_dim / causal / local 选择 tile_m, tile_n
    -> SM90 kernel 类按 WGMMA tile 推出 CTA 内 warp-group 数和 block threads
    -> TileScheduler 按 varlen / causal / split 等路径生成 grid
```

本项目第一版很多参数是固定的，例如 `ROW_M_TILE = 128`、causal varlen full prompt
prefill、`q_len == k_len`、fused persistent scheduler。因此这里要参考的是 FA4 的
推导关系和 Hopper warp-group 组织方式，而不是照搬它的灵活 launch 策略。

## tile_m / tile_n 的选择

FA4 的 Hopper forward 在 `interface.py::_tile_size_fwd_sm90()` 中按
`head_dim`、`head_dim_v`、`causal`、`local` 和 sparse Q block 约束选择 tile。

常见规则：

```text
head_dim <= 64:
    tile_m = 192
    tile_n = 128

head_dim <= 96:
    causal/local: tile_m = 192, tile_n = 128
    otherwise:    tile_m = 192, tile_n = 144

head_dim <= 128:
    tile_m = 128
    tile_n = 128

head_dim <= 192:
    tile_m = 128
    tile_n = 96 / 112 / 128
    # local 或 head_dim_v 不同时调整 tile_n

head_dim == 256:
    local:     tile_n = 64
    otherwise: tile_n = 80
    tile_m = 128
```

如果调用者显式传入 `tile_mn`，FA4 会使用该值覆盖自动选择。

对本项目的参考意义：

- 第一版设计固定使用 128 行 row tile，直接对应 FA4 中 `head_dim <= 128` 常见的
  `tile_m = 128` 路径。
- FA4 对 `head_dim <= 96` 使用 `tile_m = 192` 是性能调参结果；本项目第一版不要因此把
  FA/O_proj/AR 的统一 128 行 tile 改成 192。
- 如果以后专门优化 FA tile 效率，可以参考这些分支做实验，但这属于设计取舍，不能静默改动
  第一版 invariant。

## block shape

SM90 forward 最终 launch 使用一维 CUDA block：

```text
block = [self.num_threads, 1, 1]
```

`num_threads` 不是固定常量，而是由 WGMMA tile 的 MMA 线程数推出：

```text
num_threads_per_warp_group = 128
num_wg_mma = tiled_mma_qk.size / 128
num_threads = 128 * (num_wg_mma + 1)
```

含义：

```text
1 个 producer warp-group
+ num_wg_mma 个 MMA consumer warp-group
```

典型情况：

```text
tile_m = 128:
    atom_layout_mnk = (tile_m / 64, 1, 1) = (2, 1, 1)
    num_wg_mma = 2
    block = [384, 1, 1]

tile_m = 192:
    atom_layout_mnk = (3, 1, 1)
    num_wg_mma = 3
    block = [512, 1, 1]
```

对本项目的参考意义：

- 当前设计的 `ROW_M_TILE = 128` 与两个 MMA consumer WG 自然匹配：

```text
WG0: producer / TMA / scheduler 辅助
WG1: consumer, rows 0..63
WG2: consumer, rows 64..127
```

- 本项目 fused kernel 后续还要在同一个 CTA 内切换 FA、O_proj、AR mode。即使参考 FA4 的
  `384 threads` 组织，也必须同时满足 O_proj/AR mode 的 pipeline state、barrier 复用和
  mode drain 规则。

## grid shape

SM90 forward 在 `flash_fwd_sm90.py` 中先选择 tile scheduler：

```text
if varlen Q:
    TileScheduler = SingleTileVarlenScheduler
else if non-causal or local:
    TileScheduler = SingleTileScheduler
else:
    TileScheduler = SingleTileLPTScheduler
```

然后通过：

```text
tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
```

### non-varlen simple scheduler

普通 non-varlen scheduler 的 grid 近似是：

```text
grid = (
    round_up(num_m_blocks, cluster_m),
    num_heads * num_splits,
    batch_size
)

num_m_blocks = ceil_div(seqlen_q, tile_m)
```

如果 scheduler 使用 cluster index，则 `grid.x` 会按 physical CTA 数扩成
`num_m_blocks * cluster_m`。

### non-varlen causal LPT scheduler

causal non-varlen 路径使用 LPT scheduler，其静态 grid 是：

```text
grid = (
    total_blocks,
    num_splits,
    1
)

total_blocks = num_m_blocks * num_heads * batch_size
```

它把 `(m_block, head, batch)` 压平，便于做 L2-aware / LPT 风格的调度。

### varlen scheduler

varlen Q 路径使用 `SingleTileVarlenScheduler`，grid 是压平后的最大 row tile 数乘 head：

```text
total_blocks_max =
    (total_q + num_batch * (cluster_m * tile_m - 1)) / tile_m

total_blocks_max =
    floor_to_multiple(total_blocks_max, cluster_m)

grid = (
    total_blocks_max * num_heads,
    num_splits,
    1
)
```

这里的 `total_blocks_max` 是一个 padded 上界，不等价于本项目 host 侧精确生成的
`num_fa_row_tiles = sum_b ceil(seqlen_q[b] / 128)`。FA4 scheduler 会在 kernel 内根据
`cu_seqlens_q` / `seqused_q` 还原每个 batch 的真实 tile 范围。

对本项目的参考意义：

- FA4 varlen grid 展示了把 varlen row tile 与 head 压平成一维 work id 的方法。
- 本项目设计稿选择 host 侧预生成 `fa_row_desc[fa_row_tile] = (batch_idx, fa_m_block)`，
  避免 persistent hot path 里反复做 prefix-sum 查找。
- 因此本项目更适合用精确的 `num_fa_row_tiles * H_local` 作为 FA task 总数，再由
  persistent worker 用 atomic counter 或队列领取，而不是照搬 FA4 的 padded grid。

## persistent 差异

需要特别注意：FA4 Hopper SM90 forward 的 causal varlen 路径不是本项目意义上的 fused
persistent kernel。

在 `flash_fwd_sm90.py` 的 Hopper forward launch 中：

```text
is_persistent = False
```

所以它的 causal varlen launch 仍然是“每个 CTA 对应一个或一组 FA tile work”的形态。

本项目第一版要求：

```text
一个 persistent kernel
    FA task
    -> O_proj task
    -> AR owner task
```

同一批常驻 CTA 在 kernel 内动态切换 mode，并通过 ready queue / owner readiness /
NVLS ready-count 协议串联数据流。这一点和 FA4 的普通 forward launch 不同。

## 对本项目固定策略的建议

当前 fused kernel 的 launch shape 不应该设计成 FA4 那种全自动多分支策略。更合适的第一版策略是：

```text
block = [384, 1, 1]
    WG0 producer / scheduler / async copy 相关职责
    WG1 consumer, rows 0..63
    WG2 consumer, rows 64..127

grid = [num_persistent_ctas, 1, 1]
    num_persistent_ctas 按 SM 数和每 SM 常驻 CTA 数确定
```

FA/O_proj/AR 的真实 work 数量不要编码进 CUDA grid，而是作为 kernel 参数或 workspace
metadata 进入 persistent scheduler：

```text
total_fa_tasks        = num_fa_row_tiles * H_local
total_oproj_tasks     = num_fa_row_tiles * num_n_super_groups
total_ar_owner_tasks  = num_fa_row_tiles * num_n_super_groups 的 owner 子集
```

这样与核心设计稿保持一致：

- FA row tile 固定 128 行。
- O_proj / AR row tile 与 FA tile 对齐。
- causal mask 使用 `q_len == k_len` 的完整 prompt prefill 语义。
- persistent scheduler 负责跨 FA、O_proj、AR 三种 task 的动态调度。

## 源码索引

```text
interface.py::_tile_size_fwd_sm90
    选择 SM90 forward 的 tile_m / tile_n。

interface.py::_flash_attn_forward
    根据 arch 创建 FlashAttentionForwardSm90，并传入 tile_m / tile_n。

flash_fwd_sm90.py::FlashAttentionForwardSm90.__call__
    创建 tiled_mma，推导 num_wg_mma / num_threads，选择 TileScheduler，launch kernel。

tile_scheduler.py::SingleTileScheduler.get_grid_shape
    non-varlen simple grid。

tile_scheduler.py::SingleTileLPTScheduler.get_grid_shape
    non-varlen causal / local LPT grid。

tile_scheduler.py::SingleTileVarlenScheduler.get_grid_shape
    varlen Q grid。
```
