# Causal Varlen Prefill FlashAttention + O_proj + NVLS AllReduce 设计稿

将Attention和O_proj和NVLS ALlReduce融合成为一个算子

FlashAttention-4 官方实现作为外部参考固定在仓库 submodule：

```text
third_party/flash-attention
```

该外部仓库用于参考 CuTe DSL、SM90 persistent attention pipeline、varlen block info、
causal mask、online softmax、TMA/WGMMA/mbarrier 协作等实现细节。它不能覆盖本文的
第一版范围和 invariant；如果 FA4 通用路径与本文冲突，以本文设计为准，先讨论再修改。

## 目标范围

本文只讨论第一版设计：

- Hopper SM90。
- causal attention。
- varlen prefill。
- Q token 数量较多。
- decoder-only serving 中的 prompt 阶段。
- 完整 prompt prefill：每个 sequence 满足 `q_len == k_len`。
- FlashAttention 后接 O_proj。
- O_proj 后接 tensor-parallel NVLS AllReduce。
- 使用一个 persistent kernel，把 FA、O_proj、AllReduce 串在同一个 kernel 内。
- 使用python + CudaDSL 进行实现
这里不覆盖 decode、append prefill 或 chunked prefill。decode 通常 `seqlen_q = 1`、
`seqlen_k = cache_len + 1`，Q row 很少，主要并行度来自 head 和 KV split；append/chunked
prefill 通常也会出现 `seqlen_k > seqlen_q`。这些场景需要继承 FA4 的 Q/K 尾部对齐 causal
mask 语义、partial O/LSE combine、paged KV、decode row packing 等额外机制。第一版先把
`q_len == k_len` 的完整 prompt prefill 跑清楚。

## 基本计算

FlashAttention 计算：

```text
O = softmax(QK^T) V
```

第一版只支持完整 prompt prefill。对 varlen batch 中的每个 sequence：

```text
seqlen_q == seqlen_k
Q rows 很多
causal = True
```

因此第一版 causal mask 使用标准 prompt 下三角语义：

```text
k_index <= q_index
```

不支持 `seqlen_k != seqlen_q` 的尾部对齐语义。FA4 仓库里通用 causal mask 在
`seqlen_k != seqlen_q` 时会使用：

```text
k_index <= q_index + (seqlen_k - seqlen_q)
```

第一版 fused kernel 显式排除这种 append/decode/chunked prefill 路径，避免把复杂度混入
第一版 O_proj/AR 融合验证。

### Tile 尺寸约定

第一版让 FA、O_proj 和 AR 使用同一个 128 行 row tile。Hopper WGMMA 的 M atom
以 64 行为基本单位；当 tile M 为 128 时，一个 CTA 内的两个 consumer warp group
可以沿 M 维各处理 64 行：

```text
ROW_M_TILE   = 128
FA_M_TILE    = ROW_M_TILE
OPROJ_M_TILE = ROW_M_TILE
AR_M_TILE    = ROW_M_TILE
```

FA mode 用满 12 warps（WG0 producer + WG1/WG2 consumers）：WG1 处理 FA tile 的
rows 0..63，WG2 处理 rows 64..127。**softmax 规约只在各自 owned 的 64 行内完成，
不跨 WG 合并**。

O_proj/AR mode 也使用同一个 128 行 row tile。对一个 `out_n_tile`，WG1 负责
rows 0..63，WG2 负责 rows 64..127。两个 WG 处理的是同一个 N tile 的不同行，
不需要跨 WG 合并 accumulator。

### FA task 与 O_scratch

FlashAttention 的 FA tile 取 `FA_M_TILE = 128`：

```text
FA task = (fa_row_key, head)
        = (batch_idx, fa_m_block, head)
```

一个 FA task 产出该 head 在整个 128 行 FA tile 上的 attention 输出：

```text
O_scratch[fa_row_tile_id, m, head, d]      # m = 0 .. 127
```

这里：

```text
fa_row_key : 一个 varlen sequence 内的一组 128 个 Q rows，逻辑上 (batch_idx, fa_m_block)
batch_idx  : 当前 serving batch 中的序列编号
fa_m_block : 当前 sequence 内的 128 行 Q tile 编号，不跨 sequence 递增
head       : 当前 TP rank 上的 local attention head
m          : FA tile 内的行位置 (0 .. FA_M_TILE-1)
d          : head_dim 维度
```

物理存储使用压平后的 FA row tile id：

```text
fa_row_tile_id   = cu_fa_m_blocks[batch_idx] + fa_m_block
cu_fa_m_blocks[b] = sum_{i < b} ceil(seqlen_q[i] / FA_M_TILE)
num_fa_row_tiles = cu_fa_m_blocks[num_batch]

O_scratch physical layout = [fa_row_tile_id, FA_M_TILE, head, d]
```

所有依赖关系都按逻辑 `fa_row_key = (batch_idx, fa_m_block)` 理解。这样 causal mask、
最后一个 partial Q tile、不同 sequence 边界都不会被全局 row tile 混在一起。

### O_proj row tile

O_proj/AR 的 row tile 与 FA tile 对齐。某个 FA tile 的所有 local heads 写完后，
O_proj 直接读取该 128 行 tile 的 `O_scratch`，把所有 local heads 拼成 GEMM 的 K 维：

```text
O_row_tile_local = view O_scratch[fa_row_tile_id, :, :, :]
                 = [OPROJ_M_TILE, H_local * D]
```

因此 `O_scratch` 物理 layout 固定为 `[fa_row_tile_id, 128, head, d]`；O_proj
把 `(head, d)` 视为连续 K 维。

例如：

```text
H_local = 4
D       = 128
OPROJ_M_TILE = 128
Hidden  = 4096
N_TILE  = 128

O_row_tile_local : [128, 4 * 128] = [128, 512]
W_o_local        : [512, 4096]
Y_partial        : [128, 4096]
```

按 output hidden 维切 tile 后：

```text
O_proj task = (fa_row_key, out_n_tile)
            = (batch_idx, fa_m_block, out_n_tile)

[OPROJ_M_TILE, H_local * D] @ [H_local * D, N_TILE]
    -> [OPROJ_M_TILE, N_TILE]
```

第一版实现中，调度任务把连续的 output N tile 组成一个 super group：

```text
n_super_group       : 一组连续 out_n_tile
super_group_n_tiles : 每个 super group 包含的 out_n_tile 数量，建议从 2 或 4 开始
O_proj CTA task     = (fa_row_tile_id, n_super_group)
```

一个 CTA 领取一个 `n_super_group`。CTA 内按 `out_n_tile` 逐个推进 super group：
每一轮 WG1/WG2 共同计算同一个 `out_n_tile` 的不同行，WG1 负责 rows 0..63，
WG2 负责 rows 64..127。这样一个 output tile 的 M 维由两个 consumer WG 分担，
同时把 O_proj task 数量和 AR ready 同步粒度降低到单个 out_n_tile 粒度的
`1 / super_group_n_tiles`。第一版中一个 super group 对应一次 AR ready 发布。

所以 FA 的计算粒度是 `(fa_row_key, head)`（128 行），但 O_proj 的启动条件按 FA tile 触发：

```text
同一个 fa_row_key 的所有 local heads 都已经写入 O_scratch
    -> 发布该 fa_row_tile_id 的所有 O_proj super group task
```

## Persistent Kernel 总体结构

整个设计是一个 kernel launch。kernel 内每个常驻 CTA 都是 worker。

每个 CTA 在循环中动态选择任务：

```text
while not done:
    如果有合适的 FA task:
        进入 FA mode
    否则如果有 O_proj task:
        进入 O_proj mode，计算 local partial 并发布 AR owner task
    否则如果有 AR owner task:
        进入 AR owner reduce mode，执行 NVLS AllReduce
    否则检查是否所有任务完成
```

FA task 队列在 kernel 启动时天然存在（以 FA tile = 128 行为单位）：

```text
num_fa_row_tiles = cu_fa_m_blocks[num_batch]
total_fa_tasks   = num_fa_row_tiles * H_local
```

一维 task id 先拆成压平 FA row tile 和 head：

```text
fa_task_id  = fa_row_tile * H_local + head

fa_row_tile = fa_task_id / H_local
head        = fa_task_id % H_local
```

然后通过预生成的 FA row descriptor 映射到 varlen 坐标：

```text
fa_row_key = fa_row_desc[fa_row_tile]
           = (batch_idx, fa_m_block)
```

`fa_row_desc[fa_row_tile]` 是 host 侧生成的固定元数据表，只保存压平 FA row tile 到
varlen 逻辑坐标的最小映射：

```text
fa_row_desc[fa_row_tile] = {
    batch_idx : int32,
    fa_m_block: int32,
}
```

主 persistent kernel 不在热路径里按 `cu_seqlens_q` 反复做 prefix-sum 查找。CTA 领取到 `fa_task_id` 后，只需要一次读取 `fa_row_desc[fa_row_tile]`，就能得到该 tile 属于哪个 sequence、以及是 sequence 内第几个 128 行 Q tile。其它派生信息仍从 `cu_seqlens_q/cu_seqlens_k` 计算。第一版要求 host 或 wrapper 在 launch 前保证每个 sequence 的 `q_len == k_len`：

```text
q_start      = cu_seqlens_q[batch_idx]
q_len        = cu_seqlens_q[batch_idx + 1] - q_start
k_start      = cu_seqlens_k[batch_idx]
k_len        = cu_seqlens_k[batch_idx + 1] - k_start

assert q_len == k_len    # 第一版完整 prompt prefill 前置条件

q_tile_start = q_start + fa_m_block * FA_M_TILE
valid_fa_m   = min(FA_M_TILE, q_len - fa_m_block * FA_M_TILE)
```

这样 `fa_row_desc` 只负责调度 id 到 varlen 坐标的还原，不缓存 q/k start、length、
`valid_fa_m` 等可由 `cu_seqlens` 直接推出的信息。由于第一版强制 `q_len == k_len`，
FA block range 和 causal mask 都按 prompt 下三角处理，不引入 `seqlen_k - seqlen_q`
尾部对齐 offset。

CTA 通过 atomic counter 动态领取 FA task：

```text
fa_task_id = atomicAdd(fa_task_counter, 1)
```

这不是让某个 SM 顺序做所有 task。所有常驻 CTA 都在竞争同一个 counter，返回值天然不同。谁先完成，谁继续领下一个 task，有利于缓解 causal prefill 中前轻后重的负载不均衡。

O_proj task 不是一开始全部生成。某个 `fa_row_key` 的所有 heads 完成后，最后完成的
FA CTA 将该 FA tile 对应的所有 O_proj super group task 写入 ready queue：

```text
if fa_m_block * FA_M_TILE < q_len:
    publish (fa_row_tile_id, n_super_group = 0 .. num_super_groups - 1)
```

这样 O_proj CTA 不需要自旋等待 `row_ready`。只要某个 task 的 queue entry 已经通过
`oproj_publish_tail` release 发布，就说明对应 FA row tile 的 `O_scratch` 已经可以读取。

### Runtime task descriptor 与动态 varlen payload

最终 fused persistent kernel 是面向整个 varlen batch 的单次 kernel launch，而不是为每个
sequence、row tile 或 hidden super group 生成独立 kernel。CTA 每次从调度器领取的 task
可能来自不同 sequence、不同 row tile 和不同 hidden super group，因此 `q_start`、
`valid_m`、`k_len`、`base_out_n_tile` 等 task 形状与边界必须由 runtime descriptor 描述，
不能作为 task 级 compile-time constant 固化在 kernel 中。

第一版 fused kernel 的编译期 specialization 只覆盖有限 kernel variant：

```text
compile-time:
    arch = SM90
    dtype = fp16/bf16
    causal = True
    FA_M_TILE = OPROJ_M_TILE = 128
    N_TILE = 128
    D = Dv = 128
    kv_stages = 2
    K_CHUNK = 64
    oproj_stages = 4
    H_local / K_local / hidden tile 参数按 launch variant 固定
```

每个 task 的 varlen 坐标、有效行数、KV block 范围、O_proj tail 范围在 runtime 解码：

```text
runtime:
    batch_idx
    fa_m_block
    q_start / k_start
    q_len / k_len
    valid_m
    n_block_min / n_block_max
    slot_id / n_super_group / base_out_n_tile / valid_n_tiles
```

这与 FA4 的 persistent/runtime varlen 路径一致：kernel variant 固定 tile/stage/head_dim，
但每个 work tile 在 loop 内根据 `batch_idx` 构造 seqlen/offset 信息，再由 block-info 逻辑
计算当前 tile 的 K block 范围，并用 dynamic loop 处理不同长度的 sequence。

FA task 的 runtime descriptor：

```text
fa_task_id = atomicAdd(fa_task_counter, 1)

fa_row_tile_id = fa_task_id / H_local
head           = fa_task_id % H_local

row_desc       = fa_row_desc[fa_row_tile_id]
batch_idx      = row_desc.batch_idx
fa_m_block     = row_desc.fa_m_block

q_start        = cu_seqlens_q[batch_idx]
k_start        = cu_seqlens_k[batch_idx]
q_len          = cu_seqlens_q[batch_idx + 1] - q_start
k_len          = cu_seqlens_k[batch_idx + 1] - k_start

assert q_len == k_len    # 第一版完整 prompt prefill 前置条件

q_tile_start   = q_start + fa_m_block * FA_M_TILE
valid_m        = min(FA_M_TILE, q_len - fa_m_block * FA_M_TILE)
```

第一版只支持完整 prompt prefill，因此 causal block range 可以按 `q_len == k_len` 简化：

```text
num_k_blocks = ceil_div(k_len, N_TILE)

# prompt causal: 当前 Q tile 可见到自己右边界对应的 K block
n_block_min = 0
n_block_max = min(num_k_blocks, ceil_div((fa_m_block + 1) * FA_M_TILE, N_TILE))
```

为了保留与 FA4 通用语义的一致性，实现中仍建议把公式写成 `seqlen_k - seqlen_q` offset
形式，再由第一版前置条件让 offset 等于 0：

```text
m_idx_max   = (fa_m_block + 1) * FA_M_TILE
n_idx_right = m_idx_max + (k_len - q_len)
n_block_max = min(ceil_div(k_len, N_TILE), ceil_div(n_idx_right, N_TILE))
```

FA 的 KV block loop 使用 runtime 计数，不使用 `range_constexpr(nblk)`。causal prompt 下推荐
从最右侧可见 K block 向左处理：

```text
if n_block_min < n_block_max:
    first = n_block_max - 1
    load K(first)
    load Q current tile
    load V(first)

    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
        n_block = n_block_max - 2 - i
        load/consume K(n_block), V(n_block)
```

从右向左不是数学必需条件；online softmax 可以按任意 block 顺序处理。这里采用右到左，
是因为 causal prompt 的最右侧 block 通常靠近对角线，需要 causal mask；更左侧的历史
K block 往往整块可见，可以走更简单的 no-mask 主路径：

```text
Q rows 384..511:
    K block 384..511  # 右侧对角线 block，需要 causal mask
    K block 256..383  # 整块可见，通常不需要 causal mask
    K block 128..255  # 整块可见
    K block 0..127    # 整块可见
```

O_proj task 的 runtime descriptor：

```text
slot_id = pop(oproj_ready_queue)

fa_row_tile_id   = slot_id / num_super_groups
n_super_group    = slot_id % num_super_groups
base_out_n_tile  = n_super_group * super_group_n_tiles

row_desc         = fa_row_desc[fa_row_tile_id]
batch_idx        = row_desc.batch_idx
fa_m_block       = row_desc.fa_m_block
q_start          = cu_seqlens_q[batch_idx]
q_len            = cu_seqlens_q[batch_idx + 1] - q_start
valid_m          = min(OPROJ_M_TILE, q_len - fa_m_block * OPROJ_M_TILE)

num_out_n_tiles  = ceil_div(hidden, N_TILE)
valid_n_tiles    = min(super_group_n_tiles, num_out_n_tiles - base_out_n_tile)
```

O_proj 的 K loop 仍可保持 compile-time，因为第一版 `K_local = H_local * D` 和
`K_CHUNK` 由 kernel variant 固定：

```text
for k_chunk in range_constexpr(K_local / K_CHUNK):
    load A_chunk + W_o_chunk
    WGMMA accumulate
```

super group 内的 N tile 循环建议使用固定上限加 runtime predicate，而不是为 tail
重新生成 kernel：

```text
for sg_tile in range_constexpr(super_group_n_tiles):
    if sg_tile < valid_n_tiles:
        out_n_tile = base_out_n_tile + sg_tile
        valid_n = min(N_TILE, hidden - out_n_tile * N_TILE)
        compute/store with (m < valid_m and n < valid_n)
```

runtime descriptor 的解码可以由各 WG 根据同一个 `(mode, arg)` 独立完成。第一版不要求
leader 把完整 descriptor 写入 shared memory 再广播；`fa_row_desc` 和 `cu_seqlens` 的读取量
很小，重复读取比引入新的 CTA-local descriptor 协议更容易验证。后续如果 profiling 证明
descriptor 读取成为瓶颈，再考虑由 leader 解码并通过 shared memory 广播。

### O_proj 规模估算

一个 O_proj `row_tile` 最多包含 `OPROJ_M_TILE` 个 token。第一版按：

```text
OPROJ_M_TILE = 128
N_TILE       = 128
```

估算（`num_fa_row_tiles = sum ceil(q_len / 128) ≈ total_tokens / 128`）：

```text
B=1,  L=32k  -> total tokens=32k  -> row_tiles=256
B=4,  L=16k  -> total tokens=64k  -> row_tiles=512
B=8,  L=8k   -> total tokens=64k  -> row_tiles=512
B=16, L=4k   -> total tokens=64k  -> row_tiles=512
```

O_proj 的 N 维 tile 数：

```text
num_out_n_tiles = ceil_div(hidden, N_TILE)

hidden=4096   -> num_out_n_tiles=32
hidden=8192   -> num_out_n_tiles=64
hidden=12288  -> num_out_n_tiles=96
hidden=16384  -> num_out_n_tiles=128
```

如果每个 task 只覆盖单个 `out_n_tile`，每个 row 产生：

```text
hidden=4096   -> 32 个 O_proj task / AR tile
hidden=8192   -> 64 个 O_proj task / AR tile
hidden=16384  -> 128 个 O_proj task / AR tile
```

这会带来较多 O_proj 调度任务和 push-to-owner AR ready atomic。反过来，如果让一个 CTA
负责整个 row 的完整 hidden，任务和同步数量最少，但会把 N 维并行度砍得太狠：

```text
hidden=4096   -> 一个 CTA 串行处理 32 个 N tile
hidden=8192   -> 一个 CTA 串行处理 64 个 N tile
hidden=16384  -> 一个 CTA 串行处理 128 个 N tile
```

长 prefill 不一定 batch 很大，`B=1, L=4k` 时只有 32 个 row tiles。整行 CTA 最多只有 32 个 O_proj tasks，可能连 SM 都喂不满，而且 AR 要等整行 hidden 都写完才开始。第一版不采用整行 CTA。

折中采用 super group：

```text
super_group_n_tiles = 2 或 4
num_super_groups    = ceil_div(num_out_n_tiles, super_group_n_tiles)
total_oproj_tasks   = num_fa_row_tiles * num_super_groups
```

例如 hidden=4096：

```text
super_group_n_tiles=2 -> 每 row_tile 16 个 O_proj task / AR tile
super_group_n_tiles=4 -> 每 row_tile 8 个 O_proj task / AR tile
```

这样可以把调度和 AR 同步数量降低 2x 到 4x，同时保留 N 维并行度和 O_proj/AR overlap。

### O_proj task identity

O_proj task 空间是规则的，task identity 使用一个 `slot_id` 表示，不需要完整 descriptor。
下标空间直接使用 FA row tile（128 行）：

```text
slot_id = fa_row_tile_id * num_super_groups + n_super_group

fa_row_tile_id = slot_id / num_super_groups
n_super_group  = slot_id % num_super_groups
base_out_n_tile = n_super_group * super_group_n_tiles
valid_n_tiles  = min(super_group_n_tiles, num_out_n_tiles - base_out_n_tile)

batch_idx  = fa_row_desc[fa_row_tile_id].batch_idx
fa_m_block = fa_row_desc[fa_row_tile_id].fa_m_block
q_len      = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]
m_start    = fa_m_block * FA_M_TILE
valid_m    = min(OPROJ_M_TILE, q_len - m_start)

# A 矩阵 = O_scratch 的完整 128 行 FA tile：
A = O_scratch[fa_row_tile_id, :, :, :]   # [128, H_local*D]

for sg_tile in 0 .. valid_n_tiles - 1:
    out_n_tile = base_out_n_tile + sg_tile
    valid_n    = min(N_TILE, hidden - out_n_tile * N_TILE)
```

`valid_m` 和 `valid_n` 只影响实际数据读写谓词，不改变调度和 ready 粒度：

```text
O_proj:
    只对 m < valid_m 且 n < valid_n 的元素写 symmetric partial buffer

NVLS final store:
    只对 m < valid_m 且 n < valid_n 的元素写 Y_final

ready_count:
    仍然按 ar_slot_id = (fa_row_tile_id, n_super_group) 每 rank 加一次
```

最后一个 partial row tile 不要求清零无效 `m` 行；O_proj store 和 NVLS final store 必须用谓词避免写出无效 token。最后一个 hidden tile 不要求 `hidden` 被 `N_TILE` 整除；尾部 `valid_n < N_TILE` 时同样用 store 谓词处理。

### 方案 A: 64-bit ready bitset 评估

方案 A 使用 `uint64_t` bitset 表示 O_proj task ready 状态：

```text
oproj_ready_bits[num_oproj_words]  # uint64_t bitset

num_oproj_words = ceil_div(total_oproj_tasks, 64)
```

其中：

```text
num_super_groups  = ceil_div(num_out_n_tiles, super_group_n_tiles)
total_oproj_tasks = num_fa_row_tiles * num_super_groups
num_oproj_words   = ceil_div(total_oproj_tasks, 64)
```

一般 prefill 下的 ready words 数量大致如下。假设：

```text
OPROJ_M_TILE = 128
N_TILE = 128
total_tokens = 64k
num_fa_row_tiles = total_tokens / OPROJ_M_TILE = 512
super_group_n_tiles = 4
```

则：

```text
hidden=4096,  super_group=4:
num_out_n_tiles=32,  num_super_groups=8
total_oproj_tasks=4096
num_ready_words=64

hidden=8192,  super_group=4:
num_super_groups=16
total_oproj_tasks=8192
num_ready_words=128

hidden=16384, super_group=4:
num_super_groups=32
total_oproj_tasks=16384
num_ready_words=256
```

如果 `super_group_n_tiles = 2`，`num_super_groups` 和 `num_ready_words` 会翻倍：

```text
hidden=4096   -> num_ready_words=128
hidden=8192   -> num_ready_words=256
hidden=16384  -> num_ready_words=512
```

bitset 方案的 producer 很便宜。某个 FA row tile ready 后，producer 通常只需要一次 `atomicOr_release` 就能发布该 row 的所有 O_proj tasks：

```text
base_slot = fa_row_tile_id * num_super_groups
mask      = 该 O_proj row 在 64-bit word 内对应的 n_super_group bits

atomicOr_release(oproj_ready_bits[word_id], mask)
```

问题在 consumer。consumer 必须轮询 ready words 才能找到可执行 task：

```text
word = acquire_load(oproj_ready_bits[word_id])

if word != 0:
    bit = ffs(word)
    old = atomicAnd_acq_rel(oproj_ready_bits[word_id], ~bit)
    if old & bit:
        slot_id = word_id * 64 + bit
        执行 O_proj(slot_id)
```

如果朴素地从 word 0 开始扫描，执行到一半时前面的 ready words 大多已经被清空。例如 `num_ready_words = 512` 时，如果前 256 个 words 已经空了，每个 consumer 尝试 O_proj 都可能先做 256 次无效 acquire load。这会浪费 L2 带宽和 CTA 调度周期。

可以用 global circular scan cursor、chunk 分片、frontier hint 等方法降低空轮询，但这些方法会引入额外策略问题：

```text
1. cursor 如果用 atomicAdd 分配 chunk，早期会很快扫到后面尚未 ready 的空区域。
2. cursor 如果作为 low-watermark，只有 chunk 全空才推进，则要处理多个 CTA 挤在同一 chunk 的竞争。
3. causal prefill 下 row 大体前面先 ready、后面后 ready，但动态 FA 调度、不同 sequence 长度和 head 完成时间会打乱严格顺序。
4. active 阶段不能空扫太多，drain 阶段又必须兜底扫完所有迟到 ready bits。
```

因此方案 A 的 producer 成本最低，但 consumer 侧需要较复杂的 bounded probing 策略才能避免无效轮询。

### 方案 B: O_proj ready queue

第一版最终采用方案 B：ready queue。FA producer 不再设置 `oproj_ready_bits`，而是把 ready 的 O_proj task 写入一个队列。队列 entry 只存 `slot_id`：

```text
oproj_queue[total_oproj_tasks]  # uint32 slot_id

oproj_reserve_tail              # producer 预留写入区间
oproj_publish_tail              # 连续已写完、consumer 可见的尾指针
oproj_consume_head              # consumer 已领取到的位置
oproj_done_count                # 已完成 O_proj task 数
```

队列区间语义：

```text
[0, consume_head)              已被 consumer 领取
[consume_head, publish_tail)   已发布，可被 consumer 领取
[publish_tail, reserve_tail)   已预留，但不保证都写完，不可消费
[reserve_tail, total_tasks)    未预留
```

ASCII 示意：

```text
0                                                        total_tasks
|----------------|-------------------|-------------------|
 consumed         published-ready     reserved-unpublished
                  可消费              不可消费，可能有洞

^                ^                   ^
consume_head     publish_tail         reserve_tail
```

#### Producer reserve / write / ordered publish

某个 `fa_row_key` 的所有 heads 完成后，最后完成的 FA CTA 作为 O_proj task producer。
它一次性发布该 FA row tile 的所有 `num_super_groups` 个 O_proj tasks：

```text
publish_oproj_tasks(fa_row_tile_id):
    n = num_super_groups
    start = atomicAdd(oproj_reserve_tail, n)
    end = start + n

    for i in 0 .. n - 1:
        slot_id = fa_row_tile_id * num_super_groups + i
        oproj_queue[start + i] = slot_id

    threadfence_release()

    while acquire_load(oproj_publish_tail) != start:
        backoff

    release_store(oproj_publish_tail, end)
```

`oproj_reserve_tail` 允许多个 producer 并发预留不重叠区间；`oproj_publish_tail` 只允许按 reservation 顺序推进，保证 consumer 可见区间连续无洞。

并发写入示意。初始状态：

```text
consume_head = 80
publish_tail = 100
reserve_tail = 100
num_super_groups = 8

idx:     80              100
         | ready tasks   | free ...
         ^               ^
         consume_head    publish_tail/reserve_tail
```

Producer A 和 B 并发 reserve：

```text
A_start = atomicAdd(reserve_tail, 8)  # 100
A_end   = 108

B_start = atomicAdd(reserve_tail, 8)  # 108
B_end   = 116
```

reserve 后：

```text
idx:     80              100        108        116
         | ready tasks   | A reserved | B reserved | free
         ^               ^                        ^
         consume_head    publish_tail              reserve_tail
```

如果 B 先写完，而 A 还没写完：

```text
idx:     100        108        116
         | A ?????? | B done   |
         ^          ^          ^
         publish    B_start    reserve_tail
```

B 不能发布到 116，因为 `[100,108)` 还可能没有写完。B 必须等待：

```text
while acquire_load(oproj_publish_tail) != B_start:
    wait
```

此时 `publish_tail = 100`，`B_start = 108`，所以 B 等待。A 写完后：

```text
idx:     100        108        116
         | A done   | B done   |
         ^          ^          ^
         publish    B_start    reserve_tail
```

A 看到 `publish_tail == A_start == 100`，执行：

```text
release_store(oproj_publish_tail, A_end)  # publish_tail = 108
```

B 随后看到 `publish_tail == B_start == 108`，执行：

```text
release_store(oproj_publish_tail, B_end)  # publish_tail = 116
```

最终：

```text
idx:     80              100        108        116
         | old ready     | A ready  | B ready  | free
         ^                                      ^
         consume_head                            publish_tail/reserve_tail
```

这样 reserve/write 可以并发，但 publish 必须按 reservation 顺序前进。consumer 只读取 `[consume_head, publish_tail)`，因此永远不会读到未初始化 descriptor。

#### Consumer pop

consumer 不使用 `atomicAdd(oproj_consume_head, 1)` 直接领取，因为如果抢早了发现 `consume_head >= publish_tail`，已经前移的 `consume_head` 会造成任务丢失。第一版使用 CAS：

```text
try_pop_oproj_task():
    while true:
        head = relaxed_load(oproj_consume_head)
        tail = acquire_load(oproj_publish_tail)

        if head >= tail:
            return EMPTY

        if atomicCAS(oproj_consume_head, head, head + 1) == head:
            slot_id = acquire_load(oproj_queue[head])
            return slot_id
```

多个 consumer 并发时：

```text
idx:     80        81        82                 116
         | ready   | ready   | ready ...        |
         ^                                      ^
         consume_head=80                        publish_tail=116

C0 CAS 80->81 成功，领取 idx 80
C1 同时 CAS 80->81 失败，重读 head=81
C1 CAS 81->82 成功，领取 idx 81
```

因此每个 O_proj task 只会被一个 CTA 领取。

#### 方案 B 取舍

方案 B 的 producer 成本高于 bitset：

```text
每个 ready row:
    1 次 atomicAdd reserve_tail
    num_super_groups 次 queue entry store
    可能短暂等待 publish_tail
    1 次 release_store publish_tail
```

例如 `hidden=16384, super_group_n_tiles=4` 时，`num_super_groups=32`，每个 ready row 要写 32 个 queue entries。方案 A 通常只需要 1 次 `atomicOr_release` 发布这个 row 的 ready bits。

但方案 B 避免了 consumer 轮询 ready words：

```text
每个 O_proj task:
    1 次 publish_tail acquire load
    1 次 consume_head CAS
    1 次 queue entry load
```

考虑到一般 prefill 下 `num_ready_words` 可能达到 64、128、256，`super_group_n_tiles=2`
时还会再翻倍，bitset consumer 如果设计不好会反复读取大量空 word。ready queue 的
consumer 成本更确定，也更容易写对。因此第一版采用方案 B。方案 A 保留为后续性能优化方向：
如果 profiling 显示 queue producer 的 entry 写入和 ordered publish 成为瓶颈，再回到 bitset + 更精细的 cursor/frontier probe 方案。

完整 happens-before 链路变为：

```text
FA head 写 O_scratch
    -> atomicAdd_acq_rel(head_ready_count)
    -> 最后一个 head 观察到计数达到 H_local
    -> 写 oproj_queue entries
    -> release_store(oproj_publish_tail)
    -> O_proj consumer acquire 看到 publish_tail
    -> atomicCAS claim consume_head
    -> 读取 queue entry，解码 slot_id
    -> 读取 O_scratch 并执行 O_proj
```

## Task 调度策略

每个 CTA 都支持三类 work source：

```text
FA task:
    来源: fa_task_counter
    任务: (fa_row_tile_id, head)

O_proj task:
    来源: oproj_ready_queue
    任务: slot_id -> (fa_row_tile_id, n_super_group)

AR owner task:
    来源: ar_owner_probe_bits
    任务: ar_slot_id -> (fa_row_tile_id, n_super_group)
```

每个 CTA 完成当前任务并回到全局调度循环后，按静态偏好表尝试取下一类任务。偏好只决定尝试顺序，不把 CTA 固定到某一类任务；当前优先级没有任务时，CTA 立即尝试下一类任务。

使用 `cta_id % 6` 的静态偏好表：

```text
class = cta_id % 6

class 0, 1, 2, 3:
    FA -> O_proj -> AR

class 4:
    O_proj -> AR -> FA

class 5:
    AR -> FA -> O_proj
```

调度循环：

```text
while not done:
    order = preference_table[cta_id % 6]

    for source in order:
        if source == FA:
            如果 try_get_fa_task() 成功:
                进入 FA mode
                break

        if source == O_proj:
            如果 try_pop_oproj_queue() 成功:
                进入 O_proj mode
                break

        if source == AR:
            如果 try_claim_ar_owner_task() 成功:
                进入 AR owner reduce mode
                break

    如果三类 source 都没有可执行任务:
        检查全局完成条件，或者短暂 backoff 后继续调度
```

这样：

- `4/6` CTA 优先推进 FA，适合 prefill 早期 FA 任务最多的阶段。
- `1/6` CTA 优先处理 O_proj，使 row ready 后能尽早启动后接 GEMM。
- `1/6` CTA 优先处理 AR owner task，使已经完成 local partial 的 tile 能及时尝试 NVLS AllReduce。
- 所有 CTA 都有 fallback 顺序，所以某一类任务暂时为空时不会原地空转。

AR owner task 的产生和领取规则：

```text
O_proj CTA 完成某个 ar_tile 的本 rank partial 后:
    写 symmetric partial buffer
    等待 partial store 真正完成
    如果使用 TMA S2G store，则 cp_async_bulk_wait_group(0, read=False)
    cute.arch.fence_proxy("alias")
    system-scope release fence
    old = atomicAdd_acq_rel_system(owner.ready_count_owner[ar_slot_id], 1)

    if old + 1 == tp_size:
        atomicOr_release_system(owner.ar_owner_probe_bits[word_id], bit)
```

`ar_owner_probe_bits` 表示某个 `ar_slot_id` 已经由最后一个完成 partial 的 rank 确认 ready，可以由 owner rank 执行 NVLS reduce/store。它不是“某个 rank 刚完成 partial”的通知位，因此 owner 不需要对未 ready 的 tile 做 retry。CTA claim 到 AR owner task 后：

```text
ar_slot_id = fa_row_tile_id * num_super_groups + n_super_group

old_probe = atomicAnd_acq_rel(ar_owner_probe_bits[word_id], ~bit)

if old_probe & bit == 0:
    没有成功 claim，直接返回调度循环

old_done = atomicOr_acq_rel(ar_done_bits[word_id], bit)

if old_done & bit == 0:
    执行 multimem.ld_reduce + multimem.st
    等待 final store 完成
    atomicAdd(ar_done_count, 1)
else:
    该 ar_slot_id 已经完成，直接返回调度循环
```

全局完成条件：

```text
fa_done_count    == total_fa_tasks
oproj_done_count == total_oproj_tasks
ar_done_count    == local_owned_ar_tasks
```

每个 O_proj task 对应一个 `ar_tile`，但每个 rank 只负责自己 owner 的 AR tile。`total_oproj_tasks` 是全局 O_proj partial 数量，`local_owned_ar_tasks` 是本 rank 要执行 NVLS reduce/store 的 owner task 数量。

done counter 的递增点必须晚于对应数据和任务发布：

```text
fa_done_count:
    O_scratch store completion
    + head_ready_count 更新完成
    + 如果是最后一个 head，则 O_proj queue publish 完成
    之后递增

oproj_done_count:
    super group 内所有 valid partial store completion
    + 如果使用 TMA S2G store，则等待 bulk store completion
    + alias proxy fence
    + system-scope release fence
    + ready_count_owner 更新完成
    + 如果是 last-arriver，则 ar_owner_probe_bits 投递完成
    之后递增

ar_done_count:
    ar_owner_probe_bits claim 成功
    + ar_done_bits 首次置位成功
    + multimem.ld_reduce / multimem.st 完成
    之后递增
```

每次 CTA 从一种 mode 切换到另一种 mode 前，必须保证：

```text
TMA copy 已完成
WGMMA 已 wait 完
当前 mode 使用的 mbarrier / pipeline stage 已收尾
所有 warp 回到 CTA 内一致的同步点
```

这里的“收尾”是 drain，不是 reset。drain 表示当前 task 不再有正在飞行的 TMA/WGMMA，
所有已经消费过的 pipeline stage 都完成了 release，后续 mode 可以安全覆盖 tensor shared
memory。drain 不表示 SMEM mbarrier 对象回到 kernel 刚启动时的初始 phase。

persistent kernel 内同一个 CTA 会在一个 kernel launch 中连续执行多个 task。mbarrier 位于
SMEM，kernel start 初始化一次，后续 task 复用同一组 mbarrier。CuTe/CUTLASS 的
`PipelineState` 是寄存器/SSA 中的软件游标，包含当前 circular stage 的 `index` 和
`phase`；`PipelineTmaAsync` 的 wait/arrive 操作会用这个 `index/phase` 去操作 SMEM
mbarrier。两者必须同步推进：

```text
PipelineState:
    软件游标，存在寄存器/SSA 值里
    make_pipeline_state 初始化
    state.advance() 推进 index，绕回 stage 0 时翻转 phase

SMEM mbarrier:
    硬件同步对象，存在 shared memory 里
    PipelineTmaAsync.create 初始化
    producer_acquire / consumer_wait / consumer_release 推进内部 phase/arrival 状态
```

因此不能在每个 FA 或 O_proj task 内重新 `make_pipeline_state`，除非同时重新初始化对应
SMEM mbarrier。重新创建 `PipelineState` 只会把软件游标恢复到初始 `index/phase`，不会重置
mbarrier 内部状态；下一次 `wait(state.index, state.phase)` 可能等待错误的 phase 并导致
hang。第一版设计采用长寿命 pipeline state：

```text
kernel start:
    初始化 FA mbarrier、O_proj mbarrier
    创建 PipelineTmaAsync 对象

dispatch loop 外:
    创建每个 mode 的 PipelineState 软件游标

每个 task:
    使用传入的 PipelineState
    payload 内正常 acquire/wait/release/advance
    task 结束时 drain，但不重建 state，也不重置 mbarrier
```

每个 CTA 至少需要下列长寿命 state。它们不放 SMEM，也不放 global memory；它们是 kernel
局部变量，在 dispatch loop 外创建，在 payload 内推进：

```text
FA K pipeline:
    fa_k_prod  # WG0 下一次写 K stage 的 producer state
    fa_k_cons  # WG1/WG2 下一次读 K stage 的 consumer state

FA V pipeline:
    fa_v_prod  # WG0 下一次写 V stage 的 producer state
    fa_v_cons  # WG1/WG2 下一次读 V stage 的 consumer state

O_proj A/Wo pipeline:
    oproj_ab_prod  # WG0 下一次写 A/Wo shared stage 的 producer state
    oproj_ab_cons  # WG1/WG2 下一次读 A/Wo shared stage 的 consumer state
```

伪代码结构：

```python
# kernel start: mbarrier / pipeline object 初始化一次
pipeline_k  = PipelineTmaAsync.create(barrier_storage=mbar_k,  ...)
pipeline_v  = PipelineTmaAsync.create(barrier_storage=mbar_v,  ...)
pipeline_ab = PipelineTmaAsync.create(barrier_storage=mbar_ab, ...)

# dispatch loop 外：长寿命软件游标
fa_k_prod = make_pipeline_state(Producer, kv_stages)
fa_v_prod = make_pipeline_state(Producer, kv_stages)
fa_k_cons = make_pipeline_state(Consumer, kv_stages)
fa_v_cons = make_pipeline_state(Consumer, kv_stages)

oproj_ab_prod = make_pipeline_state(Producer, oproj_stages)
oproj_ab_cons = make_pipeline_state(Consumer, oproj_stages)

while not done:
    mode, arg = schedule_and_broadcast()

    if mode == FA:
        run_fa_payload(..., pipeline_k, pipeline_v,
                       fa_k_prod, fa_v_prod, fa_k_cons, fa_v_cons)

    if mode == OPROJ:
        run_oproj_payload(..., pipeline_ab, oproj_ab_prod, oproj_ab_cons)
```

## FA Mode Warp Specialization

FA mode 使用 12 warps，也就是 3 个 warp group：

```text
WG0 = warp 0..3
WG1 = warp 4..7
WG2 = warp 8..11
```

分工（FA_M_TILE=128，两个 consumer WG 沿 M 维各吃 64 行）：

```text
WG0:
    TMA load Q/K/V
    管 Q pipeline、K pipeline、V pipeline
    计算当前 task 的 Q/K/V 地址和 varlen metadata

WG1:
    QK WGMMA / causal mask / online softmax / PV WGMMA，处理 FA tile rows 0..63
    维护自己 64 行的 row_max、row_sum、acc_O

WG2:
    同 WG1，但处理 FA tile rows 64..127
```

WG1/WG2 各自用一个 64 行 WGMMA atom（`atom_layout_mnk=(2,1,1)`，
`tiler_mn=(64, N_TILE)`，对应 `tile_m=128`）。softmax 规约只在各自 owned 的 64 行内完成，
两个 WG 之间不交换 row_max/row_sum，也不合并 acc_O。

FA 主流程：

```text
取 FA task: (fa_row_key, head)        # fa_row_key = (batch_idx, fa_m_block)

WG0:
    load Q[fa_row_key, head]          # 128 行
    按 causal 允许范围逐块 load K/V，K 和 V 使用独立 pipeline

WG1 (rows 0..63), WG2 (rows 64..127):
    使用当前 block 的 K 做 QK
    使用上一 block 已生成的 P 和 V 做 PV
    通过 intra-wg overlap 让 QK(current) 和 PV(previous) 重叠
```

### FA K/V pipeline 与 intra-wg overlap

第一版 FA mode 对齐 FA4 Hopper 的默认思路：Q 使用 1-stage pipeline，K 和 V 使用两个独立的
2-stage pipeline。不要把 K/V 合成一个 `pipeline_kv`，因为 K 和 V 的生命周期不同：

```text
K block:
    QK WGMMA 完成后就不再需要
    可以尽早 release K stage

V block:
    必须等该 block 的 P 已经生成后，才能参与 PV WGMMA
    release 比对应的 K stage 晚一个节拍
```

第一版固定：

```text
pipeline_q : 1 stage
pipeline_k : 2 stages
pipeline_v : 2 stages
mma_pv_is_rs = True     # P 作为 PV 的 register A operand，不写 sP
```

SMEM 概念布局：

```text
SharedStorage for FA mode

+-------------------------------------------------------------+
| mbar_ptr_Q : Q full/empty barrier, 1 stage                  |
| mbar_ptr_K : K full/empty barrier, 2 stages                 |
| mbar_ptr_V : V full/empty barrier, 2 stages                 |
+-------------------------------------------------------------+
| sQ : [FA_M_TILE=128, D, stage=1]                            |
+-------------------------------------------------------------+
| sK : [N_TILE, D, stage=2]                                   |
|      +---------------------------+------------------------+ |
|      | K.stage0                  | K.stage1               | |
|      +---------------------------+------------------------+ |
+-------------------------------------------------------------+
| sV : [N_TILE, Dv, stage=2]                                  |
|      +---------------------------+------------------------+ |
|      | V.stage0                  | V.stage1               | |
|      +---------------------------+------------------------+ |
+-------------------------------------------------------------+
```

每个 stage 有两个同步语义：

```text
empty: producer 等它，表示该 stage 可以写
full : consumer 等它，表示 TMA 已经写完，该 stage 可以读
```

`M=128` 时，WG1/WG2 都是 K/V pipeline 的 consumer。某个 stage 只有在 WG1 和 WG2
都 release 后才真正 empty，producer 才能在下一 phase 复用这个 stage：

```text
WG1 release K.stage0
WG2 release K.stage0
    => K.stage0 empty，producer 可以加载下一块 K 到 K.stage0
```

以 4 个 KV block 为例，causal prefill 通常从右向左处理：

```text
B3, B2, B1, B0
```

2-stage 的 producer 写入顺序可以理解为：

```text
time ->

K.stage0 <- K(B3)
Q stage  <- Q tile

K.stage1 <- K(B2)
V.stage0 <- V(B3)

K.stage0 <- K(B1)    # 复用 K.stage0，必须等 K(B3) 已被两个 WG release
V.stage1 <- V(B2)

K.stage1 <- K(B0)
V.stage0 <- V(B1)    # 复用 V.stage0，必须等 PV(B3) 完成并被两个 WG release

V.stage1 <- V(B0)
```

consumer 侧采用“当前 block 做 QK，上一个 block 做 PV”的软件流水：

```text
Step A: first_half(B3)
    wait K.stage0 full
    QK(B3) -> acc_S
    release K.stage0
    mask / online_softmax(acc_S)
    tOrP = P(B3)

Step B: current=B2, previous=B3
    wait K.stage1 full
    issue QK(B2) -> acc_S

    wait V.stage0 full
    issue PV(B3): tOrP=P(B3) @ V(B3) -> acc_O

    wait_group(1)       # 等较老的 QK(B2) 完成，允许 PV(B3) 继续飞
    release K.stage1
    mask / online_softmax(acc_S)      # acc_S 原地变成 P(B2)

    wait_group(0)       # 等 PV(B3) 完成
    release V.stage0
    tOrP = P(B2)        # 覆盖旧 P(B3)，给下一轮 PV(B2) 用

Step C: current=B1, previous=B2
    QK(B1) overlaps PV(B2)
    end: tOrP = P(B1)

Step D: current=B0, previous=B1
    QK(B0) overlaps PV(B1)
    end: tOrP = P(B0)

Step E: last_half(B0)
    wait V.stage1 full
    PV(B0): tOrP=P(B0) @ V(B0) -> acc_O
    release V.stage1
```

这里 `P` 不在 K/V stage 中。QK 的 WGMMA 输出 `acc_S` 是当前 WG 的寄存器 accumulator；
`online_softmax(acc_S)` 会把 scores 原地改写成未归一化概率 `P`，再通过
`reshape_acc_to_frgA / cvt_f16` 变成 PV 的 register A operand `tOrP`。默认
`mma_pv_is_rs=True`，因此第一版不分配 `sP`。

`wait_group(1)` 和 `wait_group(0)` 的含义必须严格区分：

```text
wait_group(1):
    当前有 QK(current) 和 PV(previous) 两个 WGMMA group outstanding 时，
    等到最多只剩 1 个 outstanding group。
    因此可以安全使用 QK(current) 的 acc_S，但不能认为 PV(previous) 已完成。

wait_group(0):
    等所有 outstanding WGMMA 完成。
    之后才能 release V(previous) stage，也才能覆盖 tOrP。
```

因此 FA mode 切换或 tail 阶段必须保证所有 K/V stage 都完成 wait/release，不能留下正在飞的
WGMMA 或未 release 的 pipeline stage。

FA 尾声：

```text
WG1/WG2:
    finalize online softmax
    rescale acc_O
    将各自 64 行 acc_O 写入 O_scratch[fa_row_tile_id, wg_row_base : wg_row_base+64, head, :]
    (wg_row_base = 0 for WG1, 64 for WG2; 超出 valid_fa_m 的行用 store 谓词跳过)

一个 elected lane (两个 WG 都写完后):
    等待 O_scratch store 真正完成
    old = atomicAdd_acq_rel(head_ready_count[fa_row_tile_id], 1)
    如果 old + 1 == H_local:
        acquire 已完成所有 head 的 O_scratch 发布
        reserve/write/publish 该 fa_row_tile_id 的 O_proj queue entries
```

这里的“尾声”分两部分：

- 计算尾声由 WG1/WG2 完成，因为 accumulator 在 WGMMA warp group 的寄存器里。
- 控制尾声由 elected lane 完成，负责 ready 计数和 O_proj queue 发布。

### FA 到 O_proj 的内存序

`head_ready_count[fa_row_tile_id]` 不只是普通计数器，也是 FA producer 和 O_proj task producer 之间的同步点。粒度是 128 行 FA/O_proj row tile：一个 FA tile 的全部 H_local heads 写完后，该 row tile 的 O_proj super group task 才能发布。它必须使用 device-scope release/acquire 语义的 atomic；不能把普通 `atomicAdd` 默认当作数据发布协议。

每个 FA CTA 对一个 `(fa_row_key, head)` 完成后：

```text
1. 等待所有 WGMMA 完成。
2. 完成 O_scratch[fa_row_tile_id, :, head, :] 写出（128 行，两个 WG 各 64）。
3. 如果使用 TMA S2G store，必须 wait 到 TMA store completion。
   注意：TMA 发起不等于 TMA 写完。
4. 执行 atomicAdd_acq_rel(head_ready_count[fa_row_tile_id], 1)。
```

这里的 release 语义保证：

```text
O_scratch[fa_row_tile_id, :, head, :] 的写入
    happens-before
head_ready_count[fa_row_tile_id] 的递增被后续 acquire 观察到
```

最后一个 head 的 FA CTA 看到 `old + 1 == H_local` 时，它的 acq_rel atomic 同时承担两件事：

```text
release : 发布自己这个 head 的 O_scratch。
acquire : 获取之前其他 head 对 O_scratch 的发布。
```

因此它可以安全地发布该 FA row tile 的 O_proj queue entries。O_proj consumer 通过 acquire 读取 `oproj_publish_tail` 后，再用 `consume_head` CAS 领取 queue entry，才能读取对应的 `O_scratch`。

完整 happens-before 链路是：

```text
FA head 写 O_scratch
    -> atomicAdd_acq_rel(head_ready_count)
    -> 最后一个 head 观察到计数达到 H_local
    -> 写 oproj_queue entries
    -> release_store(oproj_publish_tail)
    -> O_proj consumer acquire 看到 publish_tail
    -> atomicCAS claim consume_head
    -> 读取 queue entry，解码 slot_id
    -> 读取 O_scratch 并执行 O_proj
```

## O_proj/AR Mode Warp Specialization

O_proj/AR mode 同样使用 3 个 warp group：

```text
WG0 = warp 0..3
WG1 = warp 4..7
WG2 = warp 8..11
```

采用 TensorRT-LLM SM90 GemmAllReduce 风格：

```text
WG0:
    producer warp group
    TMA load O_scratch 和 W_o
    管多 stage pipeline

WG1/WG2:
    consumer warp group
    共同负责同一个 out_n_tile 的不同行
    WGMMA accumulate
    store partial
    完成自己负责的 64 行 partial store
```

关键约束：

```text
一个 out_n_tile 的 M 维由两个 consumer WG 分片：
    WG1 -> rows 0..63
    WG2 -> rows 64..127
两个 WG 都执行完整 K_local loop，但各自只维护自己 64 行的 accumulator。
```

这样每个 output tile 的不同行 accumulator 始终归不同 warp group 所有；由于 M 行互不重叠，
不需要跨 WG 做 accumulator 合并。

O_proj task 内部以单个 `out_n_tile` 为一轮计算单位：

```text
WG1 -> out_n_tile_i, rows 0..63
WG2 -> out_n_tile_i, rows 64..127
```

第一版推荐以 `n_super_group` 为 O_proj task 粒度：

```text
O_proj task         = (fa_row_tile_id, n_super_group)
n_super_group       = 一组连续 out_n_tile
super_group_n_tiles = 2 或 4
```

即一个 CTA 领取一个 `n_super_group`，并在 CTA 内按每轮一个 `out_n_tile` 推进：

```text
第 0 轮: WG1/WG2 -> 第 0 个 out_n_tile 的 rows 0..127
第 1 轮: WG1/WG2 -> 第 1 个 out_n_tile 的 rows 0..127
...
```

最后一个 row tile 如果 `valid_m <= 64`，WG2 的 store 全部被谓词屏蔽；如果
`64 < valid_m < 128`，WG2 只写有效行。最后一个 hidden tile 如果 `valid_n < N_TILE`，
WG1/WG2 都按 N 维谓词写出。

## O_proj/AR Pipeline

O_proj 的 K 维是：

```text
K_local = H_local * D
```

每个 consumer WG 对同一个 `out_n_tile` 的不同行执行完整 K loop：

```text
A = O_scratch[fa_row_tile_id, :, :, :]   # [128, H_local * D]

for k_chunk in K_local:
    A_chunk = view(A)[:, k_chunk]
    B_chunk = W_o_local[k_chunk, out_n_tile]
    WG1: acc_Y_rows_0_63    += A_chunk[0:64, :] @ B_chunk
    WG2: acc_Y_rows_64_127  += A_chunk[64:128, :] @ B_chunk
```

WG0 负责向 shared memory stage 搬运数据。A/B stage 对两个 consumer WG 共享：

```text
sA[num_stages]   # [OPROJ_M_TILE, K_CHUNK] = [128, K_CHUNK]
sB[num_stages]   # [K_CHUNK, N_TILE]
```

每个 stage 有一个 full/empty 状态。`empty` 必须等 WG1/WG2 都完成该 stage 的 WGMMA
并且不再读取该 stage 后才能 release：

```text
full[stage]
empty[stage]
consumer_done_mask[stage]  # bit0 = WG1 done, bit1 = WG2 done
```

流程：

```text
WG0:
    wait empty[stage]
    TMA load A_chunk + B_chunk
    release full[stage]

WG1:
    wait full[stage]
    issue WGMMA accumulate rows 0..63
    wait_group，确认本 WG 对该 stage 的 WGMMA 已经不再读取 sA/sB
    mark consumer_done_mask[stage].WG1
    如果 WG1/WG2 都 done:
        release empty[stage]

WG2:
    wait full[stage]
    issue WGMMA accumulate rows 64..127
    wait_group，确认本 WG 对该 stage 的 WGMMA 已经不再读取 sA/sB
    mark consumer_done_mask[stage].WG2
    如果 WG1/WG2 都 done:
        release empty[stage]
```

WG1/WG2 可以在同一 stage 上并行 issue WGMMA，但 stage 复用必须等待两个 WG 都完成。
第一版采用保守 release 规则：**不能在 WGMMA issue 后立刻 release empty**。某个 WG
只有在 `wgmma.wait_group` 确认该 stage 对应的 WGMMA 已完成、不会再读 `sA/sB` 后，
才能设置自己的 done bit。最后一个设置 done bit 的 WG 才 release `empty[stage]`。

第一版可以先用最容易验证的策略：

```text
每处理一个 k_chunk:
    wait full[stage]
    issue WGMMA
    wait_group(0)
    mark consumer_done_mask
    两个 WG 都 done 后 release empty[stage]
```

这个策略牺牲一部分 O_proj 内部 overlap，但 release 点清晰，不会发生 producer 复用 stage
覆盖仍被 WGMMA 读取的 shared memory。功能跑通后再优化为多 stage outstanding：
只在 stage 即将被复用前 wait 到该 stage 相关 WGMMA 完成。

整个 `n_super_group` 的 local partial 写完后，CTA 再对该 `ar_slot_id` 发布一次 owner ready 信号。

## Shared Memory 预算

H200 基于 Hopper 架构。第一版按 SM90/Hopper 的 shared memory 上限估算：

```text
per SM shared memory capacity       ~= 228 KB
per thread block usable upper bound ~= 227 KB
```

设计上不能把 FA mode 和 O_proj/AR mode 的 shared storage 简单相加。一个 CTA 同一时刻只处于一种 mode，并且 mode 切换前要求 TMA/WGMMA/pipeline 全部收尾，因此 shared storage 必须按 union/overlay 复用：

```text
shared_storage = max(FA_mode_smem, O_proj_AR_mode_smem)
                 + mbarrier / padding / alignment overhead
```

overlay 只用于 tensor payload buffer，不用于 mbarrier。mbarrier 是 pipeline 的同步对象，
有独立 phase/arrival 状态，必须按 pipeline 类型单独分配并在 kernel start 初始化一次。
FA mode 的 K/V mbarrier 和 O_proj mode 的 A/Wo mbarrier 不能互相覆盖。tensor shared
memory 则可以覆盖，因为任一 CTA 同一时刻只执行一种 mode，且 mode 切换前要求当前 mode
完全 drain。

第一版 shared storage 采用如下概念结构：

```text
SharedStorage

+-------------------------------------------------------------+
| mode broadcast / CTA-local small scalars                    |
+-------------------------------------------------------------+
| FA mbarriers                                                |
|   mbar_k : K pipeline full/empty barriers                   |
|   mbar_v : V pipeline full/empty barriers                   |
|   mbar_q : 可选，若 Q 后续改成 TMA/cp.async pipeline         |
+-------------------------------------------------------------+
| O_proj mbarriers                                            |
|   mbar_ab : shared A/Wo pipeline full/empty barriers         |
+-------------------------------------------------------------+
| tensor_overlay : MemRange[dtype, max(FA_tensor, OPROJ_tensor)] |
|   同一块字节在不同 mode 下按不同 layout 和 offset 解释        |
+-------------------------------------------------------------+
```

FA mode 进入时，`tensor_overlay` 被解释为：

```text
FA tensor view:
    sQ = overlay[off_fa_sQ : off_fa_sQ + cosize(sQ_layout)]
    sK = overlay[off_fa_sK : off_fa_sK + cosize(sK_layout)]
    sV = overlay[off_fa_sV : off_fa_sV + cosize(sV_layout)]

FA_tensor = aligned_cosize(sQ_layout)
          + aligned_cosize(sK_layout)
          + aligned_cosize(sV_layout)
```

O_proj mode 进入时，同一个 `tensor_overlay` 被解释为：

```text
O_proj tensor view:
    sA  = overlay[off_op_sA  : off_op_sA  + cosize(sA_layout)]
    sWo = overlay[off_op_sWo : off_op_sWo + cosize(sWo_layout)]

OPROJ_tensor = aligned_cosize(sA_layout)
             + aligned_cosize(sWo_layout)
```

最终分配：

```text
tensor_overlay_cosize = max(FA_tensor, OPROJ_tensor)
```

CuTe DSL 中没有 C/C++ union 语法，但 FA4 Hopper 已经使用过同类 reinterpret 思路：
`Q_in_regs=True` 时只分配一块 `MemRange[max(cosize(sQ), cosize(sV))]`，然后同一个
storage 字段既可以用 `sQ_layout` 取 `sQ` view，也可以用 `sV_layout` 取 `sV` view。
本设计在此基础上多一步：同一个大 `tensor_overlay` 内部按 offset carving 出多段，再分别
用 FA 或 O_proj 的 layout 取 tensor view。实现时必须以 `cute.cosize(layout)` 和 alignment
后的 offset 为准，不能只按手算 KB 数硬编码。

第一版目标：

```text
hard limit      : <= 227 KB / CTA
engineering goal: <= 200 KB / CTA
reserve         : 20~30 KB 给 mbarrier、padding、layout alignment、编译器额外开销
```

### FA mode 粗估

下面是第一版 FA mode 的默认 shared memory 假设。这个估算是 logical tensor size，
真实实现还要以 CuTe `cute.cosize(layout)`、mbarrier 数组、alignment padding 为准：

```text
FA_M_TILE    = 128
N_TILE       = 128
D            = 128
Dv           = 128
dtype        = fp16/bf16
kv_stages    = 2
Q_in_regs    = False       # sQ 单独分配，不和 sV overlay
mma_pv_is_rs = True        # P/tOrP 在寄存器中，不分配 sP
```

第一版采用 FA4 Hopper 风格的独立 K/V pipeline：

```text
pipeline_q : 1 stage
pipeline_k : kv_stages
pipeline_v : kv_stages
```

默认 2-stage K/V pipeline 下：

```text
sQ = FA_M_TILE * D  * 2 bytes             = 128 * 128 * 2 = 32 KB
sK = 2 * N_TILE * D  * 2 bytes            = 2 * 128 * 128 * 2 = 64 KB
sV = 2 * N_TILE * Dv * 2 bytes            = 2 * 128 * 128 * 2 = 64 KB
sP = 0                                    # mma_pv_is_rs=True
----------------------------------------------------------------
FA tensor smem lower bound                                  = 160 KB
```

如果 `Dv != D`，`sV` 必须按 `Dv` 单独估算：

```text
sV = kv_stages * N_TILE * Dv * sizeof(dtype)
```

如果关闭 register-source PV，也就是 `mma_pv_is_rs=False`，需要额外分配 `sP`：

```text
sP = FA_M_TILE * N_TILE * 2 bytes = 128 * 128 * 2 = 32 KB

2-stage FA tensor smem lower bound:
    sQ + sK + sV + sP = 32 + 64 + 64 + 32 = 192 KB
```

192 KB 还没超过硬上限，但留给 O_scratch store staging、mbarrier、padding、调试余量的空间明显变小。
第一版因此固定 `mma_pv_is_rs=True`，让 P 作为寄存器 `tOrP` 直接进入 PV WGMMA。

如果 K/V pipeline 取 3-stage，并保持 `mma_pv_is_rs=True`：

```text
sQ = 32 KB
sK = 3 * 128 * 128 * 2 = 96 KB
sV = 3 * 128 * 128 * 2 = 96 KB
sP = 0
-----------------------------------------
FA tensor smem lower bound = 224 KB
```

224 KB 已经贴近 227 KB/block 上限，实际加上 mbarrier 和 alignment 后基本不可作为第一版路径。
如果 3-stage 同时 `mma_pv_is_rs=False`，还要再加 32 KB `sP`，直接不可行。

官方 FA4 Hopper 还支持 `Q_in_regs=True` 时让 `sQ/sV` 共享一块 storage（取两者最大值），
但第一版 persistent fused kernel 暂不把这个作为默认路径。原因是本设计还要处理 FA/O_proj/AR
mode overlay 和 mode 切换收尾，先选择更直观的独立 `sQ/sK/sV` 布局便于验证。

因此第一版 FA shared memory 结论是：

```text
默认路径:
    2-stage K/V pipeline
    mma_pv_is_rs=True
    Q_in_regs=False
    D=Dv=128
    FA tensor smem lower bound ~= 160 KB

工程判断:
    2-stage FA 是第一版唯一默认配置。
    3-stage FA 不进入第一版。
    如果后续要启用 sP 或 Q_in_regs，需要重新计算 shared storage overlay 和 occupancy。
```

### O_proj/AR mode 粗估

第一版 O_proj 使用共享 A/B stage。WG1/WG2 读取同一个 `sA/sB` stage，
分别计算同一个 `out_n_tile` 的上下两个 64 行半块：

```text
sA[num_stages]   # [OPROJ_M_TILE, K_CHUNK]
sB[num_stages]   # [K_CHUNK, N_TILE]
```

按 `OPROJ_M_TILE=128, N_TILE=128, dtype=fp16/bf16`，第一版默认：

```text
K_CHUNK = 64
num_stages = 4
```

则：

```text
A per stage = 128 * 64  * 2 = 16 KB
B per stage = 64  * 128 * 2 = 16 KB
per stage                        = 32 KB
four stages                      = 128 KB
```

这个配置让 O_proj mode 的 tensor smem 低于 FA mode 的 160 KB，同时 stage 数量足够多，
后续可以逐步恢复 producer/consumer overlap。不要为了和 FA mode 完全等大而硬凑到
160 KB；剩余空间留给 mbarrier、alignment、调试和后续尾声 staging 更稳。

其它备选配置：

```text
K_CHUNK=64,  stages=2:
    tensor smem = 64 KB
    正确性最简单，但 pipeline 余量偏小，第一版不作为默认。

K_CHUNK=128, stages=2:
    A per stage = 128 * 128 * 2 = 32 KB
    B per stage = 128 * 128 * 2 = 32 KB
    tensor smem = 128 KB
    smem 与默认相同，但 K 粒度更粗，调度和 tail 处理不如 K_CHUNK=64 细。

K_CHUNK=128, stages=3:
    tensor smem = 192 KB
    已接近 FA 2-stage + overhead 的压力区，不进入第一版。
```

因此第一版 O_proj/AR mode 约束为：

```text
K_CHUNK    = 64
num_stages = 4
stage type = shared A/B stage
estimated tensor smem ~= 128 KB
```

这样 O_proj/AR mode 的 shared memory 小于 2-stage FA mode，整个 kernel 的 shared memory
主要由 FA mode 决定。stage 复用时必须等 WG1/WG2 都完成该 stage 的 WGMMA 读取后再 release empty。

## O_proj/AR 尾声

O_proj compute 和 NVLS AllReduce 分成两个阶段：

```text
阶段 1: O_proj compute
    计算 local partial
    写入本 rank 的 symmetric partial buffer
    递增 owner ready_count，last-arriver 投递 AR owner task

阶段 2: AR owner reduce
    只有 owner rank claim last-arriver 投递的 ready task
    claim 到 task 后执行 multimem reduce/store
    如果 ready_count 尚未到达 tp_size，则不会产生 owner task
```

### 跨 rank ready 方式

第一版采用 push-to-owner ready count，而不是每个 rank 去通知所有 rank，也不是反复对 flag 做 `multimem.ld_reduce` 轮询。

对某个 AR tile：

```text
ar_tile    = (fa_row_tile_id, n_super_group)
ar_slot_id = fa_row_tile_id * num_super_groups + n_super_group
owner      = hash(fa_row_tile_id, n_super_group) % tp_size
```

一个 `ar_tile` 覆盖该 super group 内的所有 `out_n_tile`。因此 super group 放大后，push-to-owner ready atomic 和 owner reduce task 数量同步下降，但 owner 对单个 `ar_tile` 的 reduce/store 工作量也会变大。

每个 rank 只写自己的 partial：

```text
O_proj CTA:
    对 super group 内每个 valid out_n_tile:
        consumer WG 将 acc_Y 写入 symmetric partial buffer
    wait super group 内所有 partial store 完成
    如果使用 TMA S2G store，则 cp_async_bulk_wait_group(0, read=False)
    cute.arch.fence_proxy("alias")
    system-scope release fence
    old = atomicAdd_acq_rel_system(owner.ready_count_owner[ar_slot_id], 1)

    if old + 1 == tp_size:
        atomicOr_release_system(owner.ar_owner_probe_bits[word_id], bit)
```

也就是说，ready count 的粒度是 `ar_slot_id = (fa_row_tile_id, n_super_group)`，不是单个 `out_n_tile`。一个 rank 对同一个 `ar_slot_id` 只加一次 ready count；只有这个 super group 内所有 `valid_n_tiles` 都已经写入 symmetric partial buffer 后，才能参与 ready count。

`ar_owner_probe_bits` 只由最后一个完成 partial 的 rank 设置。`old + 1 < tp_size` 的 rank 只递增 ready count，不投递 owner task；`old + 1 == tp_size` 的 rank 负责把该 `ar_slot_id` 放入 owner rank 的 AR work source。这样 owner 不会看到未 ready 的 AR task，也不需要重试队列。

这里的 ready count 存在 owner rank 的本地 symmetric/control buffer 中。第一版不把 ready count 压缩成 owner-local 稠密数组，而是在每个 rank 上都按全局 `ar_slot_id` 全量分配：

```text
ar_slot_id = fa_row_tile_id * num_super_groups + n_super_group

ready_count_owner[total_oproj_tasks]
ready_count_owner[ar_slot_id]
```

其中只有满足：

```text
owner(ar_slot_id) == this_rank
```

的槽位会被远端 rank 写入，并由本 rank owner reduce task 读取；其它槽位虽然分配了空间，但不会作为有效 ready count 使用。这样每个 rank 发起 push-to-owner atomic 时，只需要用全局 `ar_slot_id` 作为下标，不需要额外的 `ar_slot_id -> owner_local_index` 压缩映射。第一版用这点 workspace 开销换协议简单性和可验证性。

例如 `tp_size = 4`，`ar_tile = T` 的 owner 是 rank2：

```text
rank0: write C_partial0[T] ----\
rank1: write C_partial1[T] -----\
rank2: write C_partial2[T] ------> rank2.ready_count_owner[ar_slot_id(T)]
rank3: write C_partial3[T] -----/

rank2 本地看到:
    ready_count_owner[ar_slot_id(T)] == 4
        => 所有 rank 的 partial 都已经发布
```

不推荐使用“每个 rank 通知所有 rank”的方式：

```text
rank0 -> ready_count_on_rank0/1/2/3
rank1 -> ready_count_on_rank0/1/2/3
...
```

这种方式每个 AR tile 需要 `tp_size * tp_size` 次远端 atomic。push-to-owner 只需要 `tp_size` 次 atomic，owner 读取本地 `ready_count` 也更便宜。

注意，ready count 不是 relaxed 计数器。它表示 partial 已经可以被跨 rank 的 multicast view 读取，
因此 producer 必须在 partial store 完成后执行 alias proxy fence，再用 system-scope acq_rel
语义递增 ready count。如果 partial 通过 TMA S2G 写入 symmetric buffer，TMA 发起不等于写完，
必须先等待 bulk store completion，再执行 `cute.arch.fence_proxy("alias")` 和 release atomic。
最后一个 rank 的 `atomicAdd_acq_rel_system` 同时承担两件事：

```text
release : 发布自己这个 rank 的 partial。
acquire : 获取之前其它 rank 通过 ready_count 发布的 partial。
```

最后一个 rank 随后用 `atomicOr_release_system(owner.ar_owner_probe_bits, bit)` 发布 AR owner task。owner rank acquire claim 到该 bit 后，就能安全执行 `multimem.ld_reduce`。

### 非阻塞 AR owner reduce task

每个 O_proj task 完成 local partial 后，都会递增 owner rank 上的 `ready_count_owner[ar_slot_id]`。只有最后一个让 ready count 达到 `tp_size` 的 rank，才把该 tile 加入 owner rank 的 AR owner work source。第一版可以复用 slot id 表示，不需要完整 descriptor：

```text
ar_slot_id = fa_row_tile_id * num_super_groups + n_super_group

ar_owner_probe_bits[num_oproj_words]  # owner rank 本地 uint64_t bitset
ar_done_bits[num_oproj_words]         # owner rank 本地 uint64_t bitset
```

O_proj CTA 写完本 rank partial 后：

```text
old = atomicAdd_acq_rel_system(owner.ready_count_owner[ar_slot_id], 1)

if old + 1 == tp_size:
    atomicOr_release_system(owner.ar_owner_probe_bits[word_id], bit)
```

AR owner task 不再 probe 未 ready 状态。`ar_owner_probe_bits` 里的 bit 出现时，该 `ar_slot_id` 已经 ready。claim 顺序是先清 ready work bit，再抢 done bit：

```text
old_probe = atomicAnd_acq_rel(ar_owner_probe_bits[word_id], ~bit)

if old_probe & bit == 0:
    没有成功 claim，直接返回调度循环

解码 ar_tile = (fa_row_tile_id, n_super_group)

old_done = atomicOr_acq_rel(ar_done_bits[word_id], bit)

if old_done & bit == 0:
    执行 multimem.ld_reduce + multimem.st
    等待 final store 完成
    atomicAdd(ar_done_count, 1)
else:
    该 ar_slot_id 已经完成，直接返回调度循环
```

`ar_done_bits` 是终态保护。正常情况下 last-arriver 只会投递一次 `probe_bit`，但保留 done bit 可以防御重复投递、调试阶段协议 bug、或者后续优化引入的重复 claim；它保证同一个 `ar_slot_id` 只执行一次 reduce/store，`ar_done_count` 也只递增一次。

时间线示例：

```text
tp_size = 4, owner = rank2

time ->
rank2: partial done, ready_count 0->1, not last, no probe
rank0: partial done, ready_count 1->2, not last, no probe
rank1: partial done, ready_count 2->3, not last, no probe
rank3: partial done, ready_count 3->4, last, set owner.probe_bit

owner rank2:
    claim probe_bit
    set ar_done_bits
    multimem.ld_reduce + multimem.st
    ar_done_count++
```

这样 persistent worker 不会因为跨 rank 等待而长期卡住，也不会周期性检查未 ready 的 tile。跨 rank 等待只体现在 ready count 尚未到达 `tp_size` 时没有 AR owner task 被投递。

### NVLS AllReduce 执行方式

采用 in-place symmetric buffer 的思路。FA 的 attention 输出 `O_scratch` 写到本 rank
普通本地显存，不放在 symmetric buffer 中；O_proj 的 local partial 写入 symmetric
buffer，然后 owner 直接在这块 symmetric buffer 的 multicast view 上做 AllReduce。

```text
FA O:
    O_scratch_local，普通本地显存，形状约为 [total_tokens, H_local * D]

O_proj partial / final:
    C_sym，symmetric buffer，按 FA row tile 和 output N tile 分块存储

AR:
    multimem.ld_reduce 从 C_sym_mc 读取 partial 并求和
    multimem.st 将 final Y 写回 C_sym_mc 的同一 offset
```

第一版把 symmetric partial buffer 和 final activation buffer 的物理布局固定为同一块
in-place buffer。`C_sym` 的真实物理维度按 tile padding 后分配：

```text
C_sym[fa_row_tile_id, m, out_n_tile, n]
    fa_row_tile_id : 0 .. num_fa_row_tiles - 1
    m              : 0 .. OPROJ_M_TILE - 1
    out_n_tile     : 0 .. num_out_n_tiles - 1
    n              : 0 .. N_TILE - 1

num_out_n_tiles = ceil_div(hidden, N_TILE)
num_super_groups = ceil_div(num_out_n_tiles, super_group_n_tiles)
```

O_proj/AR task identity 仍使用 `ar_slot_id`，但它只用于调度和 ready count，不单独作为
activation buffer 的主物理维度：

```text
ar_slot_id = fa_row_tile_id * num_super_groups + n_super_group
out_n_tile = n_super_group * super_group_n_tiles + sg_tile
```

因此某个 super group 内第 `sg_tile` 个 output tile 的 element 地址由
`(fa_row_tile_id, m, out_n_tile, n)` 唯一确定。每个 TP rank 在自己的 symmetric
allocation 中按这个相同 offset 写本 rank partial。owner rank 使用该 symmetric allocation
的 multicast view `C_sym_mc` 对同一 offset 做 `multimem.ld_reduce.add`，再通过
`multimem.st` 写回同一 offset。写回后，同一块 `C_sym` 上该位置的语义从 local partial
变成最终 activation，下一层可以直接把这块 symmetric allocation 作为输入。

因为 final store 覆盖的是本 tile 的 partial 位置，所以同一个 `ar_slot_id` 必须只有一个
owner 执行 reduce/store；owner 在写回前要先完成该元素的 `multimem.ld_reduce`。第一版不把
partial 和 final 分成两块 buffer，否则 activation workspace 会翻倍。

所有 partial/final store 都必须带谓词：

```text
m < valid_m
sg_tile < valid_n_tiles
n < valid_n(out_n_tile)
```

无效 `m/n` 元素不要求清零，也不参与最终输出。

 owner rank 对整个 `ar_tile` 执行 reduce/store，而不是把同一个 tile 再按 M 维切给所有 rank。owner 按 tile 轮转，长期负载均衡：

```text
owner = hash(fa_row_tile_id, n_super_group) % tp_size
```

对 owner rank：

```text
for 每个元素 e in ar_tile:
    y = multimem.ld_reduce.add(C_sym_mc[fa_row_tile_id, m, out_n_tile, n])
    multimem.st(C_sym_mc[fa_row_tile_id, m, out_n_tile, n], y)
```

图示：

```text
              symmetric partials for ar_tile T

              rank0       rank1       rank2       rank3
elem e:       p0     +    p1     +    p2     +    p3
                \          |          |          /
                 \         |          |         /
                  multimem.ld_reduce.add by owner
                              |
                              v
                    y = p0 + p1 + p2 + p3
                              |
                              v
                    multimem.st multicast final Y[e]
```

后续优化可以考虑 M-shard reduce：

```text
rank0 reduce rows 0..15
rank1 reduce rows 16..31
...
```

但这会要求每个 reducer rank 都知道该 tile 已 ready，通常会引入 `tp_size * tp_size` ready 通知或额外 coordinator 协议。第一版先采用单 owner reduce 整个 tile，降低同步复杂度。

第一版文档层面要求：

```text
同一个 ar_tile 的所有 rank partial 写完并通过 owner ready_count 发布后，owner 才能进行 multimem reduce/store。
```

## Workspace 生命周期与全局同步

TP rank 之间使用一块长期复用的 symmetric/control workspace。多层 Transformer 中不为每层重新分配 symmetric buffer；同一块 workspace 会被每层的 fused FA + O_proj + AR kernel 复用。

### Workspace 容量估算

第一版 workspace 按下面的语义划分：

```text
O_scratch_local:
    普通本地显存，非 symmetric buffer。
    保存 FA 输出，供本 rank 的 O_proj 读取。

C_sym:
    symmetric buffer。
    先保存本 rank 的 O_proj partial。
    owner reduce 后，final Y in-place 写回同一 tile-padded offset。

control workspace:
    symmetric/control 状态，包括 ready_count、bitset、queue counter、barrier 等。
```

设：

```text
T      = total_tokens
hidden = 全局 hidden size
tp     = tensor parallel size
dtype  = fp16/bf16 = 2 bytes
R      = OPROJ_M_TILE = 128
N      = N_TILE
row_tiles = num_fa_row_tiles = sum_b ceil(seqlen_q[b] / R)
n_tiles   = num_out_n_tiles  = ceil_div(hidden, N)
```

则主 workspace 容量按物理 tile padding 估算为：

```text
O_scratch_local:
    row_tiles * R * (hidden / tp) * 2 bytes

C_sym partial/final in-place:
    row_tiles * R * n_tiles * N * 2 bytes

总 activation workspace / rank:
    row_tiles * R * (hidden / tp) * 2
  + row_tiles * R * n_tiles * N * 2
```

如果把 O_proj partial 和 final Y 分成两块 symmetric buffer，则需要：

```text
O_scratch_local + C_partial_sym + Y_final_sym
    = row_tiles * R * (hidden / tp) * 2
    + 2 * row_tiles * R * n_tiles * N * 2
```

这会让 symmetric activation workspace 翻倍，第一版不采用。

典型容量估算。若所有 sequence 长度刚好对齐 128，且 `hidden` 对齐 `N_TILE`，则
`row_tiles * R == T`、`n_tiles * N == hidden`，会退化成常见的近似公式：

```text
T=64k, hidden=4096, tp=4:
    O_scratch_local = 64k * 1024 * 2 = 128 MiB
    C_sym in-place  = 64k * 4096 * 2 = 512 MiB
    activation workspace ~= 640 MiB / rank

    如果 partial/final 分开:
        128 + 512 + 512 = 1152 MiB / rank
```

只看 `C_sym` in-place 的大小：

```text
T=16k, hidden=4096   -> 128 MiB / rank
T=32k, hidden=4096   -> 256 MiB / rank
T=64k, hidden=4096   -> 512 MiB / rank

T=64k, hidden=8192   -> 1024 MiB / rank
T=64k, hidden=16384  -> 2048 MiB / rank
```

如果 varlen batch 中最后一个 FA row tile 不满 128 行，或 hidden 不是 `N_TILE` 的整数倍，
实际分配应按上面的 tile-padded 公式计算，而不是按 `T * hidden` 的逻辑有效元素数计算。
无效 `m/n` 元素不要求清零，但它们占用 symmetric buffer 地址空间，且 store/reduce/store-back
必须始终用 `valid_m/valid_n` 谓词屏蔽。

control workspace 相比 activation buffer 小很多。以 `T=64k, ROW_M_TILE=128,
hidden=4096, super_group_n_tiles=4` 为例：

```text
num_fa_row_tiles = 512
num_out_n_tiles  = 32
num_super_groups = 8
total_oproj_tasks = 4096
num_oproj_words = 64

head_ready_count[num_fa_row_tiles]       ~= 2 KB   # int32
oproj_queue[total_oproj_tasks]           ~= 16 KB  # uint32
ready_count_owner[total_oproj_tasks]     ~= 16 KB  # int32
ar_owner_probe_bits/ar_done_bits          ~= 1 KB   # uint64 bitsets
其它 counter/barrier                     ~= KB 级
```

因此第一版主要容量压力来自 activation workspace，而不是 control state。`O_scratch`
放普通本地显存、`C_sym` partial/final in-place 复用，是当前设计下最省空间且协议清晰的方案。

workspace 分成两类状态：

```text
长期 barrier 状态:
    grid_sync_count[]
    nvl_barrier_counter
    nvl_barrier_signal[2]

每次 kernel/layer 的 control 状态:
    fa_task_counter
    fa_done_count
    head_ready_count[num_fa_row_tiles]      # 按 FA tile（128 行）粒度

    oproj_reserve_tail
    oproj_publish_tail
    oproj_consume_head
    oproj_done_count

    ar_owner_probe_bits[num_oproj_words]
    ar_done_bits[num_oproj_words]
    ready_count_owner[total_oproj_tasks]    # 全量稀疏索引，只使用本 rank owner 的 ar_slot_id
    ar_done_count
```

barrier 状态使用 phase/sign 协议复用，不依赖每次清零。每次 kernel/layer 的 control 状态必须在本 kernel 开始时初始化为 0。`O_scratch` 和 symmetric partial buffer 不要求整体清零，因为有效 tile 会覆盖写；控制计数器、队列指针和 bitset 必须清零。

每层 kernel 的生命周期：

```text
1. kernel start init:
    并行清理本 rank 的 per-kernel control state
    local grid_sync
    nvl_barrier(init)

2. persistent task loop:
    FA task
    O_proj ready queue task
    AR owner reduce task

3. local_done:
    fa_done_count    == total_fa_tasks
    oproj_done_count == total_oproj_tasks
    ar_done_count    == local_owned_ar_tasks

4. kernel exit:
    local grid_sync
    nvl_barrier(exit)
    local grid_sync
    return
```

`nvl_barrier(init)` 保证所有 rank 都完成本地 control state 初始化后，才允许任何 rank 开始远端写 `ready_count_owner`、symmetric partial buffer 或其它跨 rank control state。否则可能出现某个 rank 已经开始写远端 ready count，而目标 rank 仍在清理同一位置。

`nvl_barrier(exit)` 保证所有 rank 都完成本地 owner AR 工作后，任何 rank 才能退出 kernel。由于 owner 按 tile 分散，某个 rank 自己的 owner AR 完成，并不表示其它 rank 不会继续读取它的 symmetric partial buffer；exit barrier 防止本 rank 先退出后，下一层复用 workspace 时与其它 rank 的未完成读写冲突。

`local_owned_ar_tasks` 是本 rank 作为 owner 的 AR tile 数量。它只作为 `ar_done_count` 的完成目标，不作为 `ready_count_owner` 的数组长度：

```text
local_owned_ar_tasks =
    count((fa_row_tile_id, n_super_group)
          where owner(fa_row_tile_id, n_super_group) == my_rank)
```

`ready_count_owner` 仍按 `total_oproj_tasks` 全量分配。这样 owner reduce、远端 ready atomic、AR bitset 都共享同一个全局 `ar_slot_id`，避免第一版实现中引入压缩索引表和额外一致性风险。

每个 rank 都执行完整的本地 FA 和 O_proj partial，因此 `fa_done_count` 和 `oproj_done_count` 都以 `total_*` 为完成目标；每个 rank 只 owner 一部分 AR tile，因此 `ar_done_count` 以 `local_owned_ar_tasks` 为完成目标。

### local grid_sync

`grid_sync` 是本 rank 内、本 kernel launch 内所有 persistent CTA 的 device-side barrier。它的计数器位于本 rank 的 workspace，只同步本 rank 的 CTA，不同步其它 rank。

第一版为 kernel-level init/exit barrier 预留两个 grid sync counter：

```text
kInitGridSyncIndex = 0
kExitGridSyncIndex = 1
```

也可以复用同一个 counter，因为 `grid_sync` 自身使用最高位翻转作为 phase；第一版分开两个 index，便于调试和避免不同 barrier 点复用时混淆。

`grid_sync` 的计数方式参考 Mega MoE：

```text
FINISH_TAG = 0x80000000

sync_scope()

if thread_idx == 0:
    if cta_idx == 0:
        old = atomicAdd_release(grid_sync_count,
                                FINISH_TAG - (num_persistent_ctas - 1))
    else:
        old = atomicAdd_release(grid_sync_count, 1)

    while ((load_acquire(grid_sync_count) ^ old) & FINISH_TAG) == 0:
        wait

sync_scope()
```

所有 persistent CTA 的 `thread_idx == 0` 都参与计数。最后一个到达者会让 `grid_sync_count` 的最高位翻转；所有 CTA 看到最高位相对自己的 `old` 发生变化，就说明本 rank 内所有 CTA 都到达了 barrier。

这里的 `num_persistent_ctas` 必须等于实际参与 persistent kernel 的 CTA 数量。不能有 CTA 在其它 CTA 进入 `grid_sync` 前提前 return，否则会死锁。

### nvl_barrier

`nvl_barrier` 是跨 TP rank 的 device-side barrier。它由三部分组成：

```text
local grid_sync
    -> SM0 发跨 rank NVLink signal
    -> local grid_sync
```

其中前后两个 `grid_sync` 可以通过参数控制是否执行。exit barrier 使用前后两个 local grid sync；init barrier 也使用前后两个 local grid sync。

跨 rank signal 只由本 rank 的 CTA/SM0 参与：

```text
if cta_idx == 0:
    status = nvl_barrier_counter
    phase  = status & 1
    sign   = (status >> 1) & 1

    signal_ptr = nvl_barrier_signal[phase]
    delta      = sign ? -1 : +1
    target     = sign ? 0 : tp_size

    if thread_idx < tp_size:
        red_add_release_system(sym_map(signal_ptr, dst_rank=thread_idx), delta)

    sync_scope()

    if thread_idx == 0:
        atomicAdd(nvl_barrier_counter, 1)
        while load_acquire_system(signal_ptr) != target:
            wait
```

每个 rank 的 CTA/SM0 都向所有 rank 的 `signal_ptr` 做一次 system-scope reduce add，包括写给自己。因此当本 rank 的 `signal_ptr` 达到 `target` 时，说明所有 TP ranks 都已经到达该次 barrier。

朴素方案只用一个 `signal`，第一轮从 `0` 加到 `tp_size`。它的问题是下一次 barrier 开始时 `signal` 已经等于 `tp_size`，会被误判为已经同步完成。

加入 `sign` 后，可以在后续 barrier 中从 `tp_size` 减回 `0`，避免每次清零：

```text
sign = 0: delta = +1, target = tp_size
sign = 1: delta = -1, target = 0
```

但只有 sign 仍然不够。假设某个 rank A 已经完成当前 barrier 并进入下一次 barrier，开始把同一个 signal 从 `tp_size` 减回去；另一个 rank B 虽然它本地 signal 已经到达 `tp_size`，但还没完成读取和退出当前 barrier。A 的下一轮减法可能让 B 错过 `tp_size`，造成死锁。

因此还需要 `phase`，也就是两套 signal 槽位交替使用：

```text
barrier 0: signal[0] 从 0       加到 tp_size
barrier 1: signal[1] 从 0       加到 tp_size
barrier 2: signal[0] 从 tp_size 减到 0
barrier 3: signal[1] 从 tp_size 减到 0
```

一个 rank 只有完成 barrier 1 后才可能进入 barrier 2；而完成 barrier 1 说明所有 rank 都已经进入并完成了 barrier 1，也就不会再有人停留在 barrier 0 等 `signal[0] == tp_size`。因此 `signal[0]` 可以安全地在 barrier 2 中复用。

即使第一版 kernel 只有 init 和 exit 两次跨 rank barrier，也采用 phase/sign 版本。这样 workspace 可以跨层复用，不依赖 host 每层清 barrier signal，也方便后续加入更多全局 barrier。

## 第一版不做的事情

第一版只验证完整 prompt prefill 的基本结构，暂不处理：

- decode。
- append prefill。
- chunked prefill。
- KV split。
- paged KV。
- block sparsity。
- pack_gqa 优化。
- 独立 comm warp group。
- 跨 WG 合并同一行或同一元素的 accumulator。

## 设计判断

这个设计的核心取舍是：

```text
FA 以 (fa_row_key, head) 为粒度，FA_M_TILE=128，WG1/WG2 沿 M 各吃 64 行，保留 head 维并行度。
O_proj/AR 使用同一个 128 行 row tile，直接消费 FA 写出的 O_scratch。
O_proj CTA task 以 (fa_row_tile_id, n_super_group) 为粒度；super group 内每个 out_n_tile 由 WG1/WG2 沿 M 分片共同完成。
一个 persistent kernel 内，CTA 可以动态切换 FA mode 和 O_proj/AR mode。
```

它的主要优点：

- 避免 attention 被多个 O_proj tile 重算。
- 避免固定 FA CTA / O_proj CTA 比例带来的尾部空闲。
- O_proj 和 AllReduce 可以在 row key ready 后尽早启动。
- super group 把 O_proj task 和 AR ready atomic 数量降低到单个 out_n_tile 粒度的 `1 / super_group_n_tiles`。
- O_proj ready queue 避免 consumer 轮询大量空 ready words，task identity 仍由 slot id 隐式解码。
- O_proj compute 和 AR owner reduce 解耦，AR 未 ready 时不会投递 owner task，也不会阻塞 persistent worker。
- AR owner 按 tile 轮转，避免每个 tile 使用 `tp_size * tp_size` ready 通知。
- kernel start init + NVLink init barrier 允许 symmetric/control workspace 在多层 Transformer 中安全复用。
- DeepGEMM 风格的 device-side NVLink exit barrier 保证所有 rank 都完成本层 symmetric buffer 读写后才退出 kernel。

主要风险：

- 一个 kernel 内包含两套 mode，shared memory layout 和 pipeline 状态更复杂。
- fused payload 必须以 runtime task descriptor 为准；FA 的
  `q_start/k_len/valid_m/n_block_min/n_block_max` 和 O_proj 的
  `slot_id/base_out_n_tile/valid_m/valid_n_tiles` 都是运行时值，不能把 task 级
  `nblk` 当成 compile-time constant 并用 `range_constexpr(nblk)` 固化。
- FA mode 不能把 K/V 当作一个联合 stage 生命周期处理；第一版必须验证独立 `pipeline_k`
  和 `pipeline_v` 的 acquire/release 配对，以及 `tOrP` 寄存器覆盖时机。
- CTA mode 切换前必须严格收尾所有异步操作。
- O_proj/AR 的 owner ready_count、symmetric buffer、NVLS completion 协议必须独立验证。
- O_proj ready queue 的 ordered publish 会引入短暂 head-of-line blocking，后 reserve 的 producer 即使先写完也必须等待前面的区间发布。
- AR owner ready bitset 只由 last-arriver 投递；需要保证 ready_count acq_rel 链路和 ar_done_bits 终态保护正确，避免重复 reduce/store。
- `grid_sync` 要求所有 persistent CTA 都参与；任何 CTA 提前 return 或错误进入 barrier 都会导致死锁。
- workspace init 和 exit barrier 必须覆盖所有会被远端 rank 读写的 control state，避免跨层复用时清零和远端写冲突。
- super group 过大时会降低 N 维并行度并推迟 AR 启动；第一版不使用整行 CTA。
- shared memory 必须按 mode overlay 复用；第一版目标控制在 200 KB/CTA 内，优先采用 2-stage FA 和 `K_CHUNK=64, num_stages=4` 的 O_proj。
