# FlashAttention-4 SM90 paged KV 路径参考

本文只整理 `third_party/flash-attention/flash_attn/cute/` 中 SM90 forward 对
contiguous KV、paged TMA KV 和 paged cp.async KV 的代码组织。它是
`megaAttention` paged KV 方案的外部参考，不能覆盖
`causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md` 中的 fused
FA -> O_proj -> NVLS AR 协议。

## 结论

FA4 在源码组织上使用同一个 Python interface、同一个 `FlashAttentionForwardSm90`
类和同一个 `kernel` 函数覆盖 paged / non-paged。但它不是在同一个已编译 kernel
里用 runtime `if page_table is None` 混跑所有路径。

关键分支进入 JIT compile key：

```text
page_table is not None
page_size not in [None, tile_n]   # paged KV non-TMA
```

因此对 SM90 来说会生成不同 specialization：

```text
non-paged contiguous/batched KV:
    page_table is None
    page_size is None
    paged_kv_non_tma = False
    use_tma_KV = True

paged KV, page_size == tile_n:
    page_table is not None
    page_size == tile_n
    paged_kv_non_tma = False
    use_tma_KV = True

paged KV, page_size != tile_n:
    page_table is not None
    page_size != tile_n
    paged_kv_non_tma = True
    use_tma_KV = False
```

所以“看起来一个 kernel”是源码模板层面的事实；从编译结果和寄存器/SMEM/pipeline
形态看，它仍然是按 compile-time 条件分开的路径。

## Interface 层

入口在：

```text
third_party/flash-attention/flash_attn/cute/interface.py
```

当 `page_table is not None` 时，FA4 要求：

```text
cu_seqlens_k is None
page_table.dtype == int32
page_table.shape == [batch_size, max_num_pages_per_seq]
K/V shape == [num_pages, page_size, num_head_kv, head_dim]
```

这说明 FA4 paged KV 路径不使用 `cu_seqlens_k` 描述 K 的 batch prefix。每个 batch
的有效 KV 长度通过 `seqused_k` 可选表达；如果没有 `seqused_k`，kernel 内的静态
`seqlen_k` 在 SM90 `SeqlenInfoQK` 中会按 CuTe view 的 page size 维和 page table
宽度计算：

```text
seqlen_k_static = mK.shape[0] * mPageTable.shape[1]
                = page_size * max_num_pages_per_seq   # 按 FA4 paged K/V CuTe view 语义
```

处理。Python interface 入口检查的原始 PyTorch shape 仍是
`[num_pages, page_size, H_kv, D]`；进入 CuTe 后，paged K/V load 代码按 page-size 维、
head_dim 维和 physical page 维来构造 TMA/cp.async 访问。

### 具体例子

假设一个 SM90 paged prefill launch：

```text
B = 2
page_size = tile_n = 128
num_pages = 10
max_num_pages_per_seq = 4
H_kv = 1
D = 128

K_cache / V_cache 原始 PyTorch shape:
    [num_pages=10, page_size=128, H_kv=1, D=128]

page_table:
    shape = [B=2, max_num_pages_per_seq=4]
    batch 0: [7, 2, 5, 0]
    batch 1: [3, 9, 1, 4]

seqused_k:
    batch 0: 300 tokens
    batch 1: 448 tokens
```

Python interface 层只检查两件事：

```text
1. page_table 是 [2, 4] 的 int32 表。
2. K/V 是 [10, 128, 1, 128] 的 paged cache。
```

它不再需要 `cu_seqlens_k=[0, ..., ...]`，因为 K/V 不是按 batch 连续拼起来的。
batch 0 的 logical KV 前缀不是 `K_cache` 里某段连续 `[0:300]`，而是由
`page_table[0]` 映射出来：

```text
logical tokens 0..127    -> physical page 7
logical tokens 128..255  -> physical page 2
logical tokens 256..299  -> physical page 5 的前 44 行
logical tokens 300..511  -> padding / 不属于有效 KV，由 seqused_k=300 mask 掉
```

对应到 FA tile 的 K block：

```text
n_block = 0 -> page_table[0, 0] = 7
n_block = 1 -> page_table[0, 1] = 2
n_block = 2 -> page_table[0, 2] = 5
n_block = 3 -> page_table[0, 3] = 0   # 对 batch 0 无语义，因为 seqused_k=300
```

在 SM90 paged TMA 路径中，源码把 paged K view 组织成可用 physical page 作为 TMA
source index 的形式。可以把 kernel 内访问理解成下面这个等价关系：

```text
K_view[page_offset, d, kv_head, physical_page]
    等价于 Python 原始 K_cache[physical_page, page_offset, kv_head, d]
```

所以 `mK.shape[0]` 在这段 SM90 paged K/V view 里表达的是 `page_size=128`，
`mPageTable.shape[1]` 表达的是每条 sequence 最多 4 个 logical pages。若没有
`seqused_k`，kernel 只能把每条 sequence 的静态 K 长度看成：

```text
seqlen_k_static = page_size * max_num_pages_per_seq
                = 128 * 4
                = 512
```

但本例 batch 0 实际只有 300 个有效 KV token。因此需要 `seqused_k[0]=300`
（在 megaAttention 设计里对应 `cache_seqlens[0]=300`）来让 block range 和
`kv_pos < seqlen_k` mask 使用真实长度。否则 batch 0 的 logical block 2 后半和
block 3 会被当成有效 KV 参与 attention。

对 `page_size == tile_n == 128`，paged TMA 的加载粒度刚好和 logical K block 对齐：

```text
处理 batch 0, n_block = 1:
    physical_page = page_table[0, 1] = 2
    TMA load K_cache[2, :, kv_head, :]
    TMA load V_cache[2, :, kv_head, :]
```

最后一个不满 page 的 block 不改变 TMA tile 形状。例如 batch 0 的 `n_block=2`
仍然整页 TMA load `K_cache[5, :, kv_head, :]`，但 softmax mask 只允许
`kv_pos < 300`，也就是这个 block 内只有 local rows `0..43` 有效。

SM90 forward 对象创建时传入：

```text
paged_kv_non_tma = page_size not in [None, tile_n]
```

`FlashAttentionForwardSm90.__init__` 内部再转换成：

```text
self.use_tma_KV = not paged_kv_non_tma
```

这就是 TMA 与 cp.async paged KV 的总开关。这个值是 constexpr，不是每个 tile 的
runtime 条件。

## SM90 Kernel 层

核心文件：

```text
third_party/flash-attention/flash_attn/cute/flash_fwd_sm90.py
```

`__call__` 里只有在 `self.use_tma_KV` 为真时才创建 K/V TMA descriptor：

```text
if self.use_tma_KV:
    tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(...)
    tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(...)
```

然后传给 kernel 的 K/V 参数也不同：

```text
mK_arg = tma_tensor_K if self.use_tma_KV else mK
mV_arg = tma_tensor_V if self.use_tma_KV else mV
```

kernel 内 pipeline 也跟着分开：

```text
if self.use_tma_KV:
    pipeline_k = PipelineTmaAsync(...)
    pipeline_v = PipelineTmaAsync(...)
else:
    pipeline_k = PipelineCpAsync(...)
    pipeline_v = PipelineCpAsync(...)
```

producer 线程参与方式也不同：

```text
TMA KV:
    is_kv_load_warp = warp_idx_in_wg == 0
    只有 producer WG 内的 warp 0 发起 TMA

cp.async KV:
    is_kv_load_warp = True
    producer WG 的所有 warp 参与 cp.async row gather
```

## Non-Paged TMA

当 `self.use_tma_KV` 为真且 `mPageTable is None`：

```text
mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
gK = local_tile(mK_cur, [tile_n, head_dim], [None, 0])
gV = local_tile(mV_cur, [tile_n, head_dim_v], [None, 0])
```

之后加载第 `n_block` 个 K/V tile 时：

```text
src_idx = n_block
tma_load_fn(src_idx=src_idx, ...)
```

这里 `n_block` 是当前 sequence 内的 logical K block，`seqlen.offset_batch_K`
已经把 tensor view 移到当前 batch / 当前 varlen K prefix 的起点。

## Paged TMA

当 `self.use_tma_KV` 为真且 `mPageTable is not None`：

```text
mK_cur = mK[None, None, head_idx_kv, None]
mV_cur = mV[None, None, head_idx_kv, None]
gK = local_tile(mK_cur, [tile_n, head_dim], [0, 0, None])
gV = local_tile(mV_cur, [tile_n, head_dim_v], [0, 0, None])
```

这个 view 刻意保留 page 维。每个 logical `n_block` 先查表：

```text
page_idx = mPageTable[batch_idx, n_block]
```

然后 TMA load 使用 physical page 作为 source index：

```text
src_idx = page_idx
tma_load_fn(src_idx=src_idx, ...)
```

因此 `page_size == tile_n` 的 paged TMA 路径仍然是一块 logical K block 对应一次完整
TMA tile load。它不在 tile 内逐行计算 page offset。

## Paged cp.async

当 `self.use_tma_KV` 为假时，SM90 进入 paged cp.async 路径。FA4 的这个分支是为
`page_size != tile_n` 设计的。

producer 侧创建：

```text
paged_kv_manager = PagedKVManager.create(
    mPageTable,
    mK,
    mV,
    FastDivmodDivisor(mK.shape[0]),  # FA4 paged K/V CuTe view 中的 page_size
    batch_idx,
    head_idx_kv,
    tidx,
    seqlen.seqlen_k,
    leftpad_k = 0,
    n_block_size = tile_n,
    ...
)
```

每个 logical `n_block` 会先：

```text
paged_kv_manager.load_page_table(n_block)
```

`load_page_table` 不是只查一次 block-level page。它按当前线程负责的若干 row 计算：

```text
row_idx = n_block * n_block_size + row
page_idx, page_offset = divmod(row_idx + leftpad_k, page_size)
page = page_table[page_idx] if row_idx < seqlen_k else 0
```

然后 `load_KV` 对每个 row 用保存下来的 `(physical page, page_offset)` 计算 K 或 V
的实际 gmem pointer，再用 `cp.async` 把该 row 的 head_dim 分片搬到 shared memory：

```text
x_ptr = elem_pointer(K_or_V_cache, [page_offset, d_offset, physical_page])
cp.async.global.shared(...)
```

所以 cp.async 路径的本质是 row gather：一个 `tile_n=128` 的 logical K/V tile 可以跨
多个不连续 physical pages。它不能用一次 TMA 完整表达。

## Block Range 与 Mask

paged / non-paged 不改变 FA4 的 block range 计算。consumer 和 producer 都通过
`SeqlenInfoQK` 和 `BlockInfo` 得到：

```text
n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
```

纯 causal 情况下核心公式是：

```text
n_block_max = ceil_div(seqlen_k, tile_n)
m_idx_max   = (m_block + 1) * tile_m
n_idx_right = m_idx_max + seqlen_k - seqlen_q
n_block_max = min(n_block_max, ceil_div(n_idx_right, tile_n))
n_block_min = 0
```

这就是 bottom-right aligned causal 语义。paged KV 只改变 K/V load 的地址映射，不改变
Q/K 的 logical token 坐标和 mask 公式。

## 对 megaAttention 的含义

`megaAttention` 如果只实现第一阶段 `page_size == FA_N_TILE == 128`，应参考 FA4 的
paged TMA 分支，而不是 `PagedKVManager`：

```text
logical n_block -> physical_page = page_table[batch_idx, n_block]
TMA load K_cache[physical_page, :, kv_head, :]
TMA load V_cache[physical_page, :, kv_head, :]
```

`megaAttention` 可以采用与 FA4 类似的公共源码模板，但必须让 KV layout 成为编译期
specialization，而不是 device runtime 分支。第一阶段可以用 Optional `page_table`
触发 JIT 分支：

```text
page_table is None:
    K/V 使用 cu_seqlens_k + contiguous TMA view

page_table is not None:
    K/V 使用 page_table + cache_seqlens + paged TMA view
    host wrapper 必须保证 page_size == 128

paged_cpasync_varlen:
    page_size != 128 时另行设计；需要类似 FA4 PagedKVManager 的 row gather
```

也就是说，公共 fused kernel 模板可以覆盖 contiguous 和 paged TMA 128；最终编译结果
仍是不同 specialization，未选分支由 `const_expr` 消除。不要把 `page_size != 128`
的 cp.async 路径提前混进第一阶段。

这里 `cache_seqlens` 在语义上对应 FA4 的 `seqused_k`：它提供每个 batch 的实际 KV
长度。若不传等价信息，paged cache 的最后有效 token 边界会退化成静态 page-table 容量，
这不符合 serving prefill 的 request-level KV 前缀语义。
