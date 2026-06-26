# Causal Varlen Prefill FA + O_proj + NVLS AR Fused Kernel 设计

本文是 `megaAttention` fused persistent kernel 的主设计文档。它同时描述两部分内容：

```text
1. 当前已经实现并由测试覆盖的 contiguous-KV 主线。
2. 当前已经实现并由测试覆盖的 paged-KV TMA-128 路径。
```

两条路径共享同一条下游数据流：

```text
causal varlen prefill FlashAttention
    -> O_proj
    -> tensor-parallel NVLS AllReduce
```

Paged KV 已经作为 compile-time variant 进入 `src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py`
（构造参数 `paged=True, page_size=128`）。其设计以本文 paged TMA-128 章节为准；旧的补充文档
`docs/design/paged_kv_prefill_tma_128_design_zh.md` 只作为来源记录。

## 1. 范围

当前 kernel 固定服务于一个目标：在 Hopper SM90 上把 causal varlen prefill 的
FA、O_proj 和 TP NVLS AllReduce 串进同一个 persistent kernel。

支持范围：

- Hopper SM90。
- causal attention。
- contiguous-KV varlen prefill。
- paged-KV varlen prefill TMA-128。
- 完整 prompt prefill：`q_len == k_len`。
- chunked / append prefill：`0 < q_len < k_len`，Q chunk 对齐到 KV 前缀尾部。
- MHA 与标准 GQA。
- FA 后接 O_proj。
- O_proj partial 后接 TP NVLS AllReduce。
- Python + CuTe DSL 实现。

不在本文主线中展开：

- `page_size != 128` 的 paged cp.async gather。
- decode。
- SplitKV。
- partial O / LSE combine。
- non-causal attention。
- block sparsity。
- pack_gqa 优化。
- 独立 comm warp group。
- SM90 以外架构。

## 2. 端到端数据流

设计中的 fused kernel 数据流固定为：

```text
Q/K/V 或 Q + K_cache/V_cache
  -> FA task:      (fa_row_tile_id, q_head)
  -> O_scratch:    [row_tile_id, 128, H_local, D]
  -> O_proj task:  slot_id = row_tile_id * num_super_groups + n_super_group
  -> C_sym:        [row_tile_id, 128, out_n_tile, N_TILE]
  -> AR owner:     ar_slot_id == slot_id
  -> C_sym final:  in-place NVLS reduce/store result
```

`row_tile_id` 是全链路的 row identity。FA、O_proj 和 AR 都以同一个 128 行 row tile
为基本单位：

```text
FA_M_TILE = OPROJ_M_TILE = AR_M_TILE = 128
```

核心 buffer：

```text
O_scratch_local:
    普通本地显存。
    保存本 rank FA 输出，供本 rank O_proj 读取。

C_sym:
    symmetric buffer。
    先保存本 rank O_proj partial。
    AR owner 完成 NVLS reduce 后，把 final output 写回同一物理位置。

control workspace:
    task_ctrl、head_ready、oproj_queue、ready_count_owner、ar_ready_bits、
    ar_done_bits、sync_ctrl 等调度和同步状态。
```

contiguous 和 paged TMA-128 只影响 FA 读取 K/V 的地址映射。FA 输出不直接进入
symmetric buffer。只有 O_proj partial / final activation 使用 `C_sym` 的 symmetric
allocation。

## 3. Shape、Tile 与 Head 约定

contiguous-KV 输入：

```text
Q : [tot_q, H_local,    D]
K : [tot_k, H_kv_local, D]
V : [tot_k, H_kv_local, D]
W_o_local : [H_local * D, hidden]
```

paged-KV TMA-128 输入：

```text
Q       : [tot_q, H_local,    D]
K_cache : [num_pages, 128, H_kv_local, D]
V_cache : [num_pages, 128, H_kv_local, D]
page_table    : [batch, max_num_pages_per_seq] int32
cache_seqlens : [batch] int32
W_o_local     : [H_local * D, hidden]
```

Paged KV 的 page size 必须等于 FA 的 N tile：

```text
page_size == FA_N_TILE == 128
```

当 `page_size == 128` 时，一个 logical K block 正好对应一个 page。kernel 对每个
logical `n_block` 只查一次 page table，然后用 TMA 加载完整 `[128, D]` K/V tile。
当 `page_size != 128` 时，一个 128 行 K/V tile 会跨多个 physical pages，需要 row gather
和 cp.async 路径；这不属于本文第一阶段。

其中 `H_local` 是当前 TP rank 的 local Q head 数，`H_kv_local` 是当前 rank 暴露给
kernel 的 local K/V head 数。

标准 GQA 约束：

```text
H_kv_local >= 1
H_local % H_kv_local == 0
q_per_kv = H_local / H_kv_local
kv_head = q_head / q_per_kv
```

GQA 只改变 K/V head 寻址，不改变 FA task 数、O_scratch layout、O_proj K 维或 AR
task identity。O_proj 的 K 维始终按 Q head 展开：

```text
K_local = H_local * D
```

默认 tile 参数：

```text
ROW_M_TILE = 128
N_TILE = 128
D = 128
K_CHUNK = 64
kv_stages = 2
oproj_stages = 4
super_group_n_tiles = 4
```

`super_group_n_tiles` 表示一个 O_proj task 覆盖几个连续 output N tile。它降低 O_proj
task 数和 AR ready 粒度，但仍保留 N 维并行度。

## 4. Varlen Row Tile Metadata

host 侧 metadata 由 `src/mega_attention/metadata/row_desc.py` 生成。它只保存从压平
row tile 到 varlen 坐标的最小映射：

```text
cu_m_blocks[b] = sum_{i < b} ceil(seqlens_q[i] / 128)
row_desc[t] = {
    batch_idx : int32,
    m_block   : int32,
}
num_row_tiles = cu_m_blocks[num_batch]
```

FA task id 和 O_proj / AR slot 都基于同一个 `row_tile_id`：

```text
fa_task_id = row_tile_id * H_local + q_head

slot_id = row_tile_id * num_super_groups + n_super_group
ar_slot_id = slot_id
```

contiguous-KV 的 shape 信息从 `cu_seqlens_q` / `cu_seqlens_k` 动态派生：

```text
b = row_desc[row_tile_id].batch_idx
m_block = row_desc[row_tile_id].m_block

q_start = cu_seqlens_q[b]
q_len   = cu_seqlens_q[b + 1] - q_start
k_start = cu_seqlens_k[b]
k_len   = cu_seqlens_k[b + 1] - k_start

valid_m = min(128, q_len - m_block * 128)
```

paged-KV TMA-128 不传 `cu_seqlens_k`。它用 `cache_seqlens` 表示每条 sequence 的有效
KV 前缀长度：

```text
b = row_desc[row_tile_id].batch_idx
m_block = row_desc[row_tile_id].m_block

q_start = cu_seqlens_q[b]
q_len   = cu_seqlens_q[b + 1] - q_start
k_len   = cache_seqlens[b]

valid_m = min(128, q_len - m_block * 128)
```

两种 KV layout 的共同前置条件：

```text
0 < q_len <= k_len
```

paged-KV TMA-128 使用相同前置条件。`cache_seqlens[b]` 表示本次 attention 可以读取的
完整有效 KV 前缀长度，单位是 token。进入 fused kernel 前，框架或前置 kernel 必须已经：

```text
hidden_states @ Wk/Wv
写入 K_cache/V_cache 的 physical page 和 page offset
填好 page_table
填好 cache_seqlens
```

如果当前 chunk 的新 K/V 还没有写入 paged cache，就不能把这些 token 计入
`cache_seqlens[b]`。

`q_len == k_len` 是完整 prompt prefill。`q_len < k_len` 是 chunked / append prefill，
当前 Q chunk 关注同一 sequence 的完整 KV 前缀，并采用 bottom-right aligned causal 语义：

```text
offset = k_len - q_len
kv_local <= q_local + offset
```

row tile 数和 workspace 容量只按 Q token 计，不按 K token 计。`k_len` 只影响 FA 的
K/V block loop 与 mask。

## 5. FlashAttention Task

FA task identity：

```text
fa_task_id -> (row_tile_id, q_head)
row_tile_id -> (batch_idx, m_block)
kv_head = q_head / q_per_kv
```

FA mode 中：

```text
Q tile = Q[q_start + m_block * 128 : ..., q_head, :]

contiguous-KV:
    K/V tile = K/V[k_start + n_block * 128 : ..., kv_head, :]

paged-KV TMA-128:
    physical_page = page_table[batch_idx, n_block]
    K tile = K_cache[physical_page, :, kv_head, :]
    V tile = V_cache[physical_page, :, kv_head, :]
```

pure causal、non-local、non-SplitKV 的 logical K block range：

```text
m_idx_min = m_block * 128
m_idx_max = (m_block + 1) * 128
offset = k_len - q_len

n_block_min = 0
n_block_max = min(ceil_div(k_len, 128),
                  ceil_div(m_idx_max + offset, 128))

n_block_min_causal_mask =
    max(n_block_min, (m_idx_min + offset) / 128)
```

元素级可见性：

```text
q_local  = m_block * 128 + row
kv_local = n_block * 128 + col

valid = (row < valid_m)
     && (kv_local < k_len)
     && (kv_local <= q_local + offset)
```

current implementation 从右向左处理 K/V blocks。最右侧 block 必须做 residual mask，
中间靠右 block 可能需要 causal mask，左侧完整可见 block 走 no-mask 快路径。

对 paged-KV TMA-128，`n_block` 仍然是 sequence-local logical K block id。`physical_page`
只参与 K/V load 地址映射，不参与 causal 坐标计算。最后一个不满 page 的 block 仍然整页
TMA load，未使用列由 `kv_local < k_len` mask 屏蔽。

FA 输出写入：

```text
O_scratch[row_tile_id, m, q_head, d]
```

最后一个 Q tile 的无效行在 FA 尾声中置零。`warp_reduction_sum(row_sum)` 是 warp 级
collective，必须对每一行无条件执行，不能把 collective 放进 `row < valid_m` 的发散分支。

## 6. FA 到 O_proj 的 Handoff

`O_scratch` 物理 layout 固定为：

```text
O_scratch[row_tile_id, 128, H_local, D]
```

O_proj 读取时把 `(H_local, D)` flatten 成 GEMM K 维：

```text
A = O_scratch[row_tile_id, :, :, :]
  = [128, H_local * D]
```

每个 FA task 只写一个 Q head。一个 row tile 的所有 local Q heads 都完成后，才能发布该
row tile 的 O_proj tasks。同步点是：

```text
head_ready[row_tile_id]
```

FA task 完成顺序：

```text
1. 等待 FA mode 内所有 WGMMA / TMA 读完成。
2. 写 O_scratch[row_tile_id, :, q_head, :]。
3. device-scope fence。
4. atomicAdd_acq_rel(head_ready[row_tile_id], 1)。
5. 如果 old + 1 == H_local：
       publish_oproj(row_tile_id)
6. atomicAdd_release(C_FA_DONE, 1)。
```

最后一个 head 的 `atomicAdd_acq_rel` 同时承担 release 和 acquire：它发布自己的
O_scratch 写入，也获取其它 heads 已发布的 O_scratch 写入。随后它才能把 O_proj task
放入 ready queue。

完整 happens-before 链：

```text
FA head 写 O_scratch
  -> head_ready atomicAdd_acq_rel
  -> 最后一个 head 观察到 H_local
  -> 写 oproj_queue entries
  -> release publish_tail
  -> O_proj consumer acquire publish_tail
  -> O_proj 读取 O_scratch
```

## 7. O_proj Task

O_proj task identity：

```text
slot_id = row_tile_id * num_super_groups + n_super_group

row_tile_id = slot_id / num_super_groups
n_super_group = slot_id % num_super_groups
base_out_n_tile = n_super_group * super_group_n_tiles
valid_n_tiles = min(super_group_n_tiles,
                    num_out_n_tiles - base_out_n_tile)
```

每个 O_proj task 处理一个 row tile 和一组连续 output N tile：

```text
A = O_scratch[row_tile_id, :, :, :]     # [128, H_local * D]
B = W_o_local[:, out_n_tile]            # [H_local * D, 128]
C = C_sym[row_tile_id, :, out_n_tile, :] # [128, 128]
```

WG1/WG2 沿 M 维分工：

```text
WG1 -> rows 0..63
WG2 -> rows 64..127
```

两个 consumer WG 都执行完整 K loop，但 accumulator 行互不重叠，不需要跨 WG 合并。

store predicate：

```text
m < valid_m
n < valid_n(out_n_tile)
```

O_proj 只写有效 `m/n` 元素。`C_sym` padding 区不要求清零，测试用 sentinel 验证
O_proj 不会覆盖 masked tail。AR 阶段可能覆盖 valid super group 内的完整物理 tile；
padding 区不属于语义输出，reference 和 consumer 都必须按 `valid_m/valid_n` 忽略。

## 8. O_proj Ready Queue

O_proj 使用 ready queue，不使用 ready bitset。

相关状态位于 `task_ctrl` 和 `oproj_queue`：

```text
C_OP_RESERVE
C_OP_PUBLISH
C_OP_CONSUME
oproj_queue[total_oproj_tasks_capacity]  # uint32 slot_id
```

队列区间语义：

```text
[0, consume_head)              已被 consumer 领取
[consume_head, publish_tail)   已发布，可被 consumer 领取
[publish_tail, reserve_tail)   已预留，但还不能消费
[reserve_tail, capacity)       未预留
```

producer 是最后完成某个 row tile 所有 heads 的 FA CTA。它一次发布该 row tile 的所有
`num_super_groups` 个 O_proj tasks：

```text
publish_oproj(row_tile_id):
    n = num_super_groups
    start = atomicAdd(C_OP_RESERVE, n)
    base = row_tile_id * n

    for i in 0 .. n - 1:
        oproj_queue[start + i] = base + i

    fence

    while acquire_load(C_OP_PUBLISH) != start:
        wait

    atomicAdd_release(C_OP_PUBLISH, n)
```

`C_OP_RESERVE` 允许多个 producer 并发预留不重叠区间。`C_OP_PUBLISH` 按 reservation
顺序推进，保证 consumer 只能看到连续、无洞、已写完的 queue entries。

consumer 领取：

```text
try_pop_oproj():
    head = load(C_OP_CONSUME)
    tail = acquire_load(C_OP_PUBLISH)

    if head < tail:
        old = atomicCAS(C_OP_CONSUME, head, head + 1)
        if old == head:
            slot_id = oproj_queue[head]
            return FOUND(slot_id)
```

这里 `publish_tail acquire` 保证 queue entry 对 consumer 可见；`consume_head CAS`
保证一个 queue entry 只被一个 CTA 领取。

## 9. AR Owner Readiness 与 Ready Bitset

AR owner task 使用 ready bitset。不要把它和 O_proj ready queue 混淆。

O_proj task 完成后，`slot_id` 同时作为 AR slot：

```text
ar_slot_id = slot_id
owner_rank = ar_slot_id % tp_size
owner_idx  = ar_slot_id / tp_size
```

每个 rank 对同一个 `ar_slot_id` 写完本 rank partial 后，向 owner rank 的
`ready_count_owner[owner_idx]` 加一。最后一个到达的 rank 设置 owner-local ready bit：

```text
publish_ar_ready(ar_slot_id):
    owner_rank = ar_slot_id % tp_size
    owner_idx  = ar_slot_id / tp_size

    old = atomicAdd_acq_rel(owner.ready_count_owner[owner_idx], 1)

    if old + 1 == tp_size:
        word = owner_idx / 64
        bit  = 1 << (owner_idx % 64)
        atomicOr_release(owner.ar_ready_bits[word], bit)
```

`ready_count_owner` 的粒度是 `(row_tile_id, n_super_group)`，不是单个 `out_n_tile`。
一个 rank 只有在该 super group 内所有 `valid_n_tiles` partial 都写入 `C_sym` 后，才能加
ready count。`ready_count_owner` 不是普通进度计数器，它也是 O_proj partial 对 AR owner
可见的发布点。发布顺序必须是：

```text
1. super group 内所有 valid partial store 写入 C_sym。
2. 等待 store 对后续跨 rank multicast 读取可见。
   当前代码使用普通 global store + device fence；如果改成 TMA S2G store，
   必须先等待 bulk store completion，再做 alias proxy fence。
3. 对 owner_rank.ready_count_owner[owner_idx] 做 acq_rel atomicAdd。
4. last-arriver 再 release 设置 owner_rank.ar_ready_bits[word]。
```

跨 rank路径使用 system-scope peer atomic；最后一个 rank 的 acq_rel `atomicAdd` 同时发布
自己的 partial，并 acquire 之前其它 rank 通过 ready count 发布的 partial。owner rank
acquire claim 到 ready bit 后，才能执行 `multimem.ld_reduce`。

owner CTA 领取 AR task：

```text
try_claim_ar():
    start = C_AR_CURSOR
    for i in 0 .. owner_words_active - 1:
        word_id = (start + i) % owner_words_active
        word = acquire_load(ar_ready_bits[word_id])
        if word == 0:
            continue

        bit = lowest_set_bit(word)
        owner_idx = word_id * 64 + bit_index
        old = atomicAnd_acq_rel(ar_ready_bits[word_id], ~bit)

        if old & bit:
            if owner_idx < local_owned_ar_tasks_active:
                ar_slot_id = owner_idx * tp_size + rank
                C_AR_CURSOR = word_id
                return FOUND(ar_slot_id)
            else:
                # tail-invalid bit: cleared and skipped
                continue
```

清掉 ready bit 就是完成 owner task 认领。Exactly-once reduce 由 `ar_ready_bits` 上的
`atomicAnd` 保证。`ar_done_bits` 不用于 claim，它只保护 done count：

```text
run_ar_owner(ar_slot_id):
    执行 NVLS reduce/store
    old_done = atomicOr_acq_rel(ar_done_bits[word], bit)
    if old_done & bit == 0:
        atomicAdd_release(C_AR_DONE, 1)
```

`ar_done_bits` / `C_AR_DONE` 的语义是 reduce 已经落地，不是 task 已认领。

`C_AR_CURSOR` 只是扫描 hint，不参与 correctness。当前实现每次从 cursor 开始全扫描
`owner_words_active` 个 active words，不使用 bounded probing，也不区分 active/drain
扫描窗口。这样 owner words 很小时逻辑更直接：只要 ready bit 已经置位，本轮全扫描一定有机会
claim 到；空扫成本只是几十次 acquire load。

## 10. NVLS AllReduce

`tp_size == 1` 时，AR 是 identity path。`C_sym` 中的 O_proj partial 已经是 final，
kernel 仍走 AR done 记账，保持调度终止协议一致。

`tp_size > 1` 时，owner rank 在 `C_sym` multicast view 上执行 in-place reduce/store：

```text
for sg_tile in valid_n_tiles:
    out_n_tile = base_out_n_tile + sg_tile
    for m in 0 .. 127:
        for n in 0 .. N_TILE step 8:
            y = multimem.ld_reduce.add(C_sym_mc[row_tile_id, m, out_n_tile, n:n+8])
            multimem.st(C_sym_mc[row_tile_id, m, out_n_tile, n:n+8], y)
```

所有 rank 的 symmetric allocation 使用相同 physical offset。owner 的 `multimem.ld_reduce`
读取所有 rank 对应 offset 的 partial 并求和，再 `multimem.st` 写回同一 offset。

AR reduce/store 覆盖 `valid_n_tiles` 内的完整物理 tile，包括 padding rows/cols。padding
不属于语义输出。最终输出和 reference compare 只读取：

```text
m < valid_m
n < valid_n(out_n_tile)
```

## 11. Persistent Scheduler

每个 CTA 是一个 persistent worker。CTA leader 领取任务，然后把 `(mode, arg)` 通过
shared memory broadcast 给整个 CTA：

```text
MODE_FA
MODE_OPROJ
MODE_AR
MODE_DONE
```

三类 work source：

```text
FA:
    source = C_FA_COUNTER
    arg = fa_task_id

O_proj:
    source = O_proj ready queue
    arg = slot_id

AR:
    source = ar_ready_bits
    arg = ar_slot_id
```

role 只决定 probe order，不是静态分区：

```text
role 0: FA -> O_proj -> AR
role 1: O_proj -> AR -> FA
role 2: AR -> FA -> O_proj
```

role 来自 `bidx % (w_fa + w_oproj + w_ar)`。所有 role 都会 fallback 到其它 work source，
所以某类 task 暂时为空不会让 CTA 固定闲置。

done 条件：

```text
C_FA_DONE >= num_fa_tasks_active
C_OP_DONE >= total_oproj_tasks_active
C_AR_DONE >= local_owned_ar_tasks_active
```

`actv` 是 per-launch active counts，layout 与 `row_desc.active_counts()` 保持一致：

```text
actv[0] = num_fa_tasks_active
actv[1] = total_oproj_tasks_active
actv[2] = num_row_tiles_active
actv[3] = owner_slots_active
actv[4] = owner_words_active
actv[5] = local_owned_ar_tasks_active
```

active count 公式：

```text
total_oproj_tasks_active = num_row_tiles_active * num_super_groups
owner_slots_active = ceil_div(total_oproj_tasks_active, tp_size)
owner_words_active = ceil_div(owner_slots_active, 64)
local_owned_ar_tasks_active =
    ceil_div(max(total_oproj_tasks_active - rank, 0), tp_size)
```

kernel 按 bucket capacity 编译和分配 workspace，但调度、done 目标、AR claim 扫描上界
都使用 active counts。

done counter 的递增点固定如下：

```text
C_FA_DONE:
    O_scratch store 完成
    + head_ready 更新完成
    + 如果是最后一个 head，则 O_proj queue publish 完成
    之后递增。

C_OP_DONE:
    C_sym partial store 完成
    + ready_count_owner 更新完成
    + 如果是 last-arriver，则 ar_ready_bits 发布完成
    之后递增。

C_AR_DONE:
    ar_ready_bits claim 成功
    + NVLS reduce/store 完成
    + ar_done_bits 首次置位成功
    之后递增。
```

## 12. Warp Group 与 Pipeline

CTA 使用 384 threads，也就是 3 个 warp groups：

```text
WG0 = producer / TMA
WG1 = consumer rows 0..63
WG2 = consumer rows 64..127
```

FA mode：

```text
pipeline_q: 1 stage
pipeline_k: kv_stages
pipeline_v: kv_stages

sQ: [128, D, 1]
sK: [128, D, kv_stages]
sV: [128, D, kv_stages]
```

K 和 V 使用独立 pipeline，因为 K 在 QK 完成后即可 release，V 要等对应 P 生成并完成
PV 后才能 release。PV 使用 register-source P，不分配 `sP`。

FA consumer 的 release 规则：

```text
K stage:
    QK WGMMA 完成并且不再读取该 K stage 后 release。

V stage:
    对应 block 的 P 已生成，并且 PV WGMMA 完成后 release。
```

当前 FA 右到左处理 K/V blocks，并用“当前 block 做 QK、上一 block 做 PV”的软件流水。
`wait_group(1)` 只表示 outstanding WGMMA group 数降到最多 1 个，可安全使用刚完成的
QK accumulator；它不表示上一轮 PV 已完成。`wait_group(0)` 才表示所有 outstanding WGMMA
完成，之后才能 release V stage 或覆盖保存给 PV 的 `tOrP`。

O_proj mode：

```text
pipeline_ab: oproj_stages

sA : [128, K_CHUNK, oproj_stages]
sWo: [K_CHUNK, N_TILE, oproj_stages]
```

O_proj stage 复用规则：

```text
producer 只能在 empty stage 上写 sA/sWo。
WG1/WG2 都会读取同一个 stage。
某个 stage 只有在 WG1 和 WG2 对该 stage 的 WGMMA 都完成后才能 release empty。
```

当前实现采用保守策略：每个 `K_CHUNK` 的 WGMMA issue 后等待完成，再 release 对应 stage。
后续如果增加多 stage outstanding，仍必须保证 producer 复用 stage 前，所有会读取该 stage
的 WGMMA 都已经完成。

Pipeline objects 和 PipelineState cursor 在 dispatch loop 外创建，并跨多个 task 长寿命复用。
不能在每个 mode branch 内重新 `make_pipeline_state`，除非同时重新初始化对应 SMEM mbarrier。

mode 切换前必须 drain 当前 mode：

```text
所有 TMA copy 已完成
所有 WGMMA 已 wait 完
已消费 stage 已 release
CTA 内所有 warp 回到同步点
```

drain 不是 reset。mbarrier phase 和 PipelineState phase 会随着 task 持续推进。

## 13. Shared Memory 与 Register

FA tensor 和 O_proj tensor 共享一个 SMEM overlay。mbarrier 不 overlay。

当前代码的概念结构：

```text
SharedStorage:
    bc[4]       # mode/arg broadcast
    mbar_q
    mbar_k
    mbar_v
    mbar_ab
    overlay
```

FA mode 把 overlay 解释为：

```text
sQ, sK, sV
```

O_proj mode 把 overlay 解释为：

```text
sA, sWo
```

overlay 正确性的前提是同一个 CTA 同一时刻只执行一种 payload mode，且 mode 切换前完成
drain。

默认 smem 下界：

```text
FA:
    sQ = 128 * 128 * 2 = 32 KB
    sK = 2 * 128 * 128 * 2 = 64 KB
    sV = 2 * 128 * 128 * 2 = 64 KB
    total ~= 160 KB

O_proj:
    per stage = 128 * 64 * 2 + 64 * 128 * 2 = 32 KB
    stages = 4
    total ~= 128 KB
```

大 accumulator 和 fragment 必须放在 payload branch 内，且只在 consumer WG 创建。跨 task
长寿命的只能是协议状态、pipeline 对象和 PipelineState cursor。

## 14. Workspace 与生命周期

control state 分两类：

```text
task_ctrl / per-layer control:
    每层结束由 kernel exit cleaner 清零。

sync_ctrl:
    长寿命 barrier state。
    不被 exit cleaner 清零。
```

`task_ctrl` scalar slots：

```text
C_FA_COUNTER
C_FA_DONE
C_OP_RESERVE
C_OP_PUBLISH
C_OP_CONSUME
C_OP_DONE
C_AR_DONE
C_AR_CURSOR
```

per-layer arrays：

```text
head_ready[max_num_row_tiles]
oproj_queue[max_total_oproj_tasks]
ready_count_owner[max_owner_slots]
ar_ready_bits[max_owner_words]
ar_done_bits[max_owner_words]
```

`sync_ctrl` slots：

```text
S_GS_QUIESCE
S_GS_CLEANDONE
S_NVL_COUNTER
```

每层 kernel exit 流程：

```text
1. 所有 CTA 达到 local done。
2. grid_sync(S_GS_QUIESCE)：
       保证本 rank 没有 CTA 仍在读写 task_ctrl。
3. 所有 CTA 并行清理 full capacity control state：
       head_ready
       ready_count_owner
       ar_ready_bits
       ar_done_bits
       task_ctrl scalars
4. grid_sync(S_GS_CLEANDONE)：
       保证本 rank 清理对 CTA0 可见。
5. tp_size > 1 时执行 nvl_barrier(exit_clean)：
       所有 rank 都完成旧工作和清理。
6. return。
```

exit cleaner 清 full capacity，不只清 active prefix。control state 很小，全清能保证下一层
任意 active range 都可以直接启动，不需要 kernel-start cleaner 或 init barrier。

以下状态不按层清零：

```text
O_scratch_local:
    本层有效 row/head 会被 FA 覆盖。

C_sym:
    本层有效 partial/final 会被 O_proj/AR 覆盖。
    padding 区允许任意值。

oproj_queue entries:
    entry 本身不清，只清 reserve/publish/consume counters。

sync_ctrl:
    使用 phase 复用协议，不能被 cleaner 清零。
```

activation workspace 按 tile-padded 物理容量分配：

```text
row_tiles = num_row_tiles_capacity
num_out_n_tiles = ceil_div(hidden, N_TILE)

O_scratch_local elements =
    row_tiles * 128 * H_local * D

C_sym elements =
    row_tiles * 128 * num_out_n_tiles * N_TILE
```

`tot_k` 只影响 FA 读取 K/V 的范围和工作量，不扩大 `O_scratch_local` 或 `C_sym`。chunked
prefill 下 `max_tot_k` 需要独立作为 K/V input buffer capacity 管理。

paged-KV TMA-128 下，K/V input capacity 按 page 管理：

```text
K_cache/V_cache capacity =
    max_num_pages * 128 * H_kv_local * D

page_table capacity =
    max_num_batch * max_num_pages_per_seq

cache_seqlens capacity =
    max_num_batch
```

Paged K/V cache、page table 和 cache seqlens 是 FA 输入 metadata，不改变
`O_scratch_local`、`C_sym`、O_proj queue 或 AR owner control 的容量公式。

control workspace 按 bucket capacity 分配：

```text
max_total_oproj_tasks = max_num_row_tiles * num_super_groups
max_owner_slots = ceil_div(max_total_oproj_tasks, tp_size)
max_owner_words = ceil_div(max_owner_slots, 64)
```

## 15. grid_sync 与 nvl_barrier

`grid_sync` 是本 rank 内所有 persistent CTA 的 barrier。当前 exit path 使用两个独立
counter：

```text
S_GS_QUIESCE
S_GS_CLEANDONE
```

它采用 `FINISH_TAG = 0x80000000` 的 high-bit phase 翻转协议。所有 persistent CTA 都必须
参与；任何 CTA 提前 return 都会死锁。

`nvl_barrier` 是跨 TP rank 的 exit barrier，只在 `tp_size > 1` 时使用。它使用两槽
signal 和 phase/sign 复用协议：

```text
phase = counter & 1
sign  = (counter >> 1) & 1

sign = 0: signal[phase] 从 0 加到 tp_size
sign = 1: signal[phase] 从 tp_size 减到 0
```

两槽 signal 避免一个 rank 进入下一次 barrier 时修改另一个 rank 仍在等待的 signal。

每层只有一次 cross-rank barrier：kernel exit 的 `exit_clean`。它同时承担下一层 init
边界：

```text
所有 rank 的本层 AR 已完成
所有 rank 的本层 control cleaner 已完成
下一层可以复用 C_sym 和 control workspace
```

首层由 workspace create 的 full zero + rendezvous 兜底。

跨 rank signal 由 CTA0 的前 `tp_size` 个线程逐 peer 执行 `+1` 或 `-1` system-scope
atomic add，包括写给本 rank 自己。barrier 后面只有 `return`，不需要 trailing local
`grid_sync`；下一层 kernel launch 边界会等待本层所有 CTA 返回。

## 16. Host Wrapper 与 Launch Metadata

host wrapper 可以提供统一入口，但底层 kernel variant 必须按 KV layout 在 JIT 编译期区分：

```text
contiguous_varlen:
    K/V 使用 contiguous tensor。
    使用 cu_seqlens_k。
    mPageTable is None。

paged_tma_varlen_128:
    K/V 使用 K_cache/V_cache。
    使用 page_table + cache_seqlens。
    page_size constexpr == 128。
    mPageTable is not None。
```

`mPageTable is None / is not None` 可以作为 CuTe/JIT compile-time specialization 条件，
类似 FA4 的 Optional tensor 分支。未选分支必须被编译期消除；不能在同一个已编译 device
kernel 里用 runtime `if page_table is None` 混跑两种 layout。

contiguous-KV host 侧需要准备：

```text
row_desc:
    batch_idx[row_tile_id]
    m_block[row_tile_id]
    cu_seqlens_q
    cu_seqlens_k
```

paged-KV TMA-128 host 侧需要准备：

```text
row_desc:
    batch_idx[row_tile_id]
    m_block[row_tile_id]
    cu_seqlens_q
    cache_seqlens
    page_table

wrapper checks:
    page_size == 128
    page_table.dtype == int32
    page_table.shape[0] == batch
    page_table.shape[1] >= max_b ceil_div(cache_seqlens[b], 128)
    K_cache/V_cache shape == [num_pages, 128, H_kv_local, D]
```

两种 KV layout 共用 task counts：

```text
task counts:
    num_fa_tasks = num_row_tiles * H_local
    num_out_n_tiles = ceil_div(hidden, N_TILE)
    num_super_groups = ceil_div(num_out_n_tiles, super_group_n_tiles)
    total_oproj_tasks = num_row_tiles * num_super_groups

active counts:
    active_counts(num_row_tiles, H_local, num_super_groups, tp_size, rank)
```

workspace capacity 按 bucket 最大 shape 分配；每次 launch 通过 `actv` 给 kernel 当前 active
范围。`FusedFaOprojAr.__init__` 中的 `num_row_tiles`、`total_oproj`、owner slots/words
是 capacity / compile-time sizing；`actv` 是 runtime active range。

runtime workspace 契约由 `src/mega_attention/runtime/workspace.py` 实现。它不是通用
attention API，而是 fused kernel 的长期资源对象：

```text
create:
    按 bucket capacity 分配 Q/K/V、O_scratch、C_sym 和 control arrays。
    paged variant 用 K_cache/V_cache/page_table/cache_seqlens 替代 contiguous K/V capacity。
    tp_size > 1 时对 C_sym、ready_count_owner、ar_ready_bits、nvl signal 做 symmetric
    allocation + rendezvous。
    首层前完成 full zero、cuda synchronize 和跨 rank barrier。

compile:
    在 bucket capacity 上编译一次，并把 multicast / peer pointers bake 进 kernel。

set_layer:
    只填 active prefix、row_desc metadata 和 actv。
    不做 host reset。

launch:
    直接启动 kernel。跨层 control 清理由上一层 kernel exit cleaner 保证。
```

`debug_zero_all()` 只用于故障定位。生产路径、benchmark 路径和默认测试路径不得依赖 host
reset，否则会绕开 workspace reuse 协议。

multi-rank NVLS 需要把以下指针作为 closure constants baked into kernel：

```text
C_sym multicast VA
nvl signal local / peer VAs
ready_count_owner peer VAs
ar_ready_bits peer VAs
```

不要把这些 64-bit VA 放进普通 CuTe compile args，以免被截断或错误处理。

## 17. Correctness Invariants

必须保持的不变量：

- `row_tile_id` 是 FA、O_proj、AR 的共享 row identity。
- `O_scratch` layout 固定为 `[row_tile_id, 128, H_local, D]`。
- GQA 只改变 K/V head 寻址，不改变 O_proj K 维和下游 task identity。
- KV layout 只改变 FA K/V load 地址映射。contiguous 和 paged TMA-128 不改变
  O_scratch、O_proj ready queue、AR owner bitset、scheduler 或 workspace lifecycle。
- paged-KV TMA-128 中 `n_block` 是 logical K block，`physical_page` 只用于 TMA source index。
- O_proj 使用 ready queue：`reserve -> write entries -> ordered publish -> CAS consume`。
- AR owner 使用 ready bitset：`ready_count_owner -> ar_ready_bits -> atomicAnd claim`。
- O_proj queue 和 AR bitset 是两套不同协议，不能混用。
- `ready_count_owner` 必须晚于 C_sym partial store 可见性发布。
- `q_len <= k_len` 使用 bottom-right aligned causal mask。
- paged-KV TMA-128 的真实 `k_len` 来自 `cache_seqlens`；最后一个 page 的无效 token 必须
  由 `kv_local < k_len` 屏蔽。
- FA tail row 的 collective 不能放进 divergent branch。
- O_proj 只写 valid `m/n`；padding 不属于语义输出。
- AR done count 只能在 reduce/store 完成后递增。
- PipelineState 和 SMEM mbarrier 长寿命配对复用，不能单独 reset state。
- tensor SMEM 可以 overlay，mbarrier 不能 overlay。
- exit cleaner 清 per-layer control full capacity，不清 sync_ctrl。
- 所有 persistent CTA 必须参与 exit grid_sync。
- runtime workspace 不依赖 host 每层 reset；首层靠 create full zero，后续层靠上一层
  exit cleaner + nvl_barrier。

## 18. Verification Matrix

当前主要验证入口：

```text
tests/fused/test_fused_single_card.py
tests/fused/test_fused_paged_single_card.py
tests/fused/test_fused_production.py
tests/fused/test_fused_paged_production.py
tests/fused/test_role_weights.py
benchmarks/bench_fused_fa_oproj_ar.py
tests/metadata/test_row_desc.py
tests/kernels/test_fa_varlen.py
```

单卡 fused 测试覆盖（contiguous: `test_fused_single_card.py`；
paged-KV TMA-128: `test_fused_paged_single_card.py`）：

- full prompt prefill。
- `q_len < k_len` contiguous chunked / append prefill。
- paged-KV TMA-128 full prompt prefill。
- paged-KV TMA-128 `q_len < k_len` chunked / append prefill。
- paged cache 最后一个 K page 不满 128。
- page_table 使用非连续、乱序 physical pages。
- wrapper 对 `page_size != 128` 拒绝进入 paged TMA variant。
- MHA。
- GQA。
- 最后一个 Q tile 不满 128。
- hidden tail / ragged super group。
- O_scratch 数值。
- C_sym O_proj 数值。
- sentinel leak，验证 O_proj 不写 masked tail。
- exit cleaner，验证 task_ctrl 退出后被清零。
- multi-layer reuse，不依赖 host reset。
- 大 active range 后接小 active range。
- 长 `k_len` 后接短 `k_len`。
- over-capacity `set_layer` 拒绝后 workspace 仍可复用。

多卡 production 测试覆盖 TP/NVLS 路径（contiguous: `test_fused_production.py`；
paged-KV TMA-128: `test_fused_paged_production.py`，已在 TP=4 H200 验证通过）。benchmark
用于评估 role weights、super group 粒度和 end-to-end 性能。

## 19. Paged KV TMA-128 实现设计

Paged KV 已经作为 compile-time variant 进入 fused kernel（`FusedFaOprojAr(paged=True,
page_size=128)`、`FusedFaOprojArWorkspace.create(paged=True, ...)`）。本节定义该实现遵守的设计。

第一阶段只实现：

```text
page_size == FA_N_TILE == 128
paged TMA load
```

不实现：

```text
page_size != 128 的 PagedKVManager / cp.async row gather
decode
SplitKV
partial O/LSE combine
```

### 19.1 为什么固定 page_size = 128

FA K/V tile 的 N 维是 128。当 `page_size == 128` 时，一个 logical K block 正好对应一个
logical page：

```text
logical n_block -> physical_page = page_table[batch_idx, n_block]
K tile = K_cache[physical_page, :, kv_head, :]
V tile = V_cache[physical_page, :, kv_head, :]
```

这可以用 TMA 一次加载完整 `[128, D]` tile。最后一个 page 即使未满 128 token，也仍然整页
加载；无效列由 `k_len` mask 屏蔽。

当 `page_size != 128` 时，一个 128 行 logical K/V tile 会跨多个 physical pages：

```text
row_idx = n_block * 128 + row
page_idx, page_offset = divmod(row_idx, page_size)
physical_page = page_table[batch_idx, page_idx]
```

physical pages 不保证连续，不能用一次 TMA 表达完整 tile。这需要类似 FA4
`PagedKVManager` 的 row gather 和 cp.async 路径。该路径会改变 K/V producer 的线程参与、
pointer 计算、predicate 和 pipeline 类型，必须作为后续独立阶段。

### 19.2 Runtime Metadata

paged variant 的 FA descriptor 由下面信息组成：

```text
row_tile_id -> (batch_idx, m_block)

q_start = cu_seqlens_q[batch_idx]
q_len   = cu_seqlens_q[batch_idx + 1] - q_start
k_len   = cache_seqlens[batch_idx]
offset  = k_len - q_len
valid_m = min(128, q_len - m_block * 128)
```

`cache_seqlens[batch_idx]` 必须是当前 request 已写入 paged cache 的完整有效 KV 前缀长度。
`page_table` 的第二维必须覆盖：

```text
ceil_div(cache_seqlens[batch_idx], 128)
```

个 logical pages。page table 中超出 `ceil_div(k_len, 128)` 的 entries 对该 sequence 无语义；
它们必须被 block range 和 mask 排除，不能参与 attention。

### 19.3 K/V Load

Paged TMA-128 的 block range 与 contiguous 路径相同：

```text
n_block_min = 0
n_block_max = min(ceil_div(k_len, 128),
                  ceil_div((m_block + 1) * 128 + offset, 128))
```

进入范围的每个 `n_block` 先查 logical page：

```text
physical_page = page_table[batch_idx, n_block]
```

然后把 `physical_page` 作为 TMA source index：

```text
TMA K: K_cache[physical_page, :, kv_head, :]
TMA V: V_cache[physical_page, :, kv_head, :]
```

不在 tile 内计算 `page_offset`。`page_offset` 恒等于 tile row，因为 `page_size == 128`。

mask 使用 logical token 坐标：

```text
q_local  = m_block * 128 + row
kv_local = n_block * 128 + col

valid = (row < valid_m)
     && (kv_local < k_len)
     && (kv_local <= q_local + offset)
```

`physical_page` 不参与 causal 计算。

### 19.4 GQA 与 TP 责任边界

Paged KV 不改变 GQA 语义：

```text
q_per_kv = H_local / H_kv_local
kv_head = q_head / q_per_kv
```

kernel 只消费当前 rank 的 local paged cache view：

```text
K_cache : [num_pages, 128, H_kv_local, D]
V_cache : [num_pages, 128, H_kv_local, D]
```

kernel 不负责 global KV head replication，也不要求 global `H_kv` 能被 TP 整除。框架层必须
保证传入的 local view 满足：

```text
H_kv_local >= 1
H_local % H_kv_local == 0
```

当 `global H_kv < TP` 时，框架可以采用 SGLang 风格 KV head replication，使每个 rank
暴露至少一个 `H_kv_local`。

### 19.5 SGLang 接口约束

SGLang 集成进入 paged TMA variant 前，wrapper 必须把 token pool KV cache reshape 成：

```text
key_cache.view(-1, 128, H_kv_local, D)
value_cache.view(-1, 128, H_kv_local, D)
```

并传入：

```text
page_table    : [batch, max_num_pages_per_seq] int32
cache_seqlens : [batch] int32
cu_seqlens_q  : [batch + 1] int32
```

如果运行时 page size 不是 128，wrapper 必须拒绝进入 `paged_tma_varlen_128`。

### 19.6 对下游协议的影响

Paged TMA-128 不改变：

```text
row_tile_id
FA task identity
O_scratch layout
head_ready 粒度
O_proj slot_id
O_proj ready queue
AR owner ready_count / ar_ready_bits / ar_done_bits
persistent scheduler
workspace exit cleaner
nvl_barrier
```

因此实现 paged KV 时，改动边界应集中在：

```text
kernel compile-time variant
wrapper 参数检查
FA descriptor 解码中的 k_len 来源
FA K/V TMA tensor view 与 source index
reference / tests 中 paged cache -> logical contiguous K/V 的还原
```
