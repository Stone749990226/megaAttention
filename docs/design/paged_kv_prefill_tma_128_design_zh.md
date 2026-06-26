# Paged KV Prefill TMA 128 设计补充

本文记录在 `causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md` 之上增加
paged KV prefill / chunked prefill 的第一阶段设计。目标是适配 SGLang 中
Qwen3-235B-A22B 这类 MHA/GQA 模型的 paged KV cache，同时保持 megaAttention 当前
causal varlen prefill FA + O_proj + NVLS AllReduce fused persistent kernel 的主线不变。

第一阶段只实现 `page_size == FA_N_TILE == 128` 的 paged TMA 路径。不实现 decode、
SplitKV、partial O/LSE combine，也不实现 `page_size != 128` 的 cp.async gather 路径。

## 范围

当前 fused kernel 需要支持两种 KV layout：

```text
contiguous-KV prefill:
    Q  : [tot_q, H_local,    D]
    K  : [tot_k, H_kv_local, D]
    V  : [tot_k, H_kv_local, D]
    cu_seqlens_q : [B + 1]
    cu_seqlens_k : [B + 1]

paged-KV prefill, first stage:
    Q       : [tot_q, H_local,    D]
    K_cache : [num_pages, 128, H_kv_local, D]
    V_cache : [num_pages, 128, H_kv_local, D]
    cu_seqlens_q : [B + 1]
    cache_seqlens: [B]
    page_table   : [B, max_num_pages_per_seq] int32
```

Paged KV 只改变 FA 读取 K/V 的寻址方式。以下设计不变量保持不变：

```text
FA row tile id 仍是 (batch_idx, fa_m_block) 压平后的 fa_row_tile_id
FA task 仍是 (fa_row_tile_id, q_head)
O_scratch layout 仍是 [fa_row_tile_id, 128, H_local, D]
head_ready_count 粒度不变
O_proj task identity 不变
AR owner/count/ready 协议不变
persistent scheduler 的 FA -> O_proj -> AR 数据流不变
```

## 为什么第一阶段固定 page_size = 128

FlashAttention-4 SM90 paged KV 路径按 `page_size` 和 FA 的 `tile_n` 关系分成两类：

```text
page_size == tile_n:
    paged TMA path

page_size != tile_n:
    PagedKVManager + cp.async gather path
```

Qwen3-235B-A22B 的 attention 参数为：

```text
global H_q  = 64
global H_kv = 4
D           = 128
```

megaAttention 当前 FA tile 设计为：

```text
FA_M_TILE = 128
FA_N_TILE = 128
D         = 128
```

当 `page_size == 128` 时，一个 logical K/V block 正好对应一个 page。kernel 对每个
`n_block` 只需要查一次 page table：

```text
physical_page = page_table[batch_idx, n_block]
```

然后将 `physical_page` 作为 TMA source index，直接加载完整的 `[128, D]` K/V tile：

```text
K_tile = K_cache[physical_page, :, kv_head, :]
V_tile = V_cache[physical_page, :, kv_head, :]
```

当 `page_size=16` 或 `page_size=1` 时，一个 `FA_N_TILE=128` 的 K/V tile 会跨多个
physical pages。physical pages 不保证连续，不能用一次 TMA 加载完整 `[128, D]` tile，
必须在 tile 内对每一行计算：

```text
page_idx, page_offset = divmod(row_idx, page_size)
physical_page = page_table[batch_idx, page_idx]
```

这会进入 FA4 的 `PagedKVManager` + cp.async gather 路径，K/V load pipeline、pointer
计算和 predicate 都与 TMA 路径不同。该路径作为第二阶段单独设计和实现，不混入第一阶段
paged TMA kernel。

## Kernel Variant 与编译期分支

上层 Python wrapper 提供统一入口。底层可以使用一个公共 fused kernel 源码模板，
但 KV layout 必须在 JIT 编译期区分，不能在同一个已编译 device kernel 里用 runtime
`if page_table is None` 混跑 contiguous 和 paged。

```text
contiguous_varlen:
    K/V contiguous
    使用 cu_seqlens_k

paged_tma_varlen_128:
    page_size constexpr = 128
    K/V paged cache
    使用 page_table + cache_seqlens
```

第一阶段允许采用与 FA4 相同的 Optional tensor 编译期分支：

```text
mPageTable is None:
    编译为 contiguous_varlen specialization

mPageTable is not None:
    编译为 paged_tma_varlen_128 specialization
```

这里的 `mPageTable is None / is not None` 必须是 `const_expr` / JIT compile-time 条件。
未选分支必须被编译期消除。host wrapper 在进入 `mPageTable is not None` 分支前必须检查：

```text
page_size == FA_N_TILE == 128
```

后续如果需要完整对齐 FA4 的一般 page size 支持，再新增：

```text
paged_cpasync_varlen:
    page_size != 128
    移植 FA4 PagedKVManager
```

`page_size != 128` 的 cp.async gather 路径不能混入第一阶段。它需要
`PagedKVManager` 式 row gather，会改变 K/V producer 的线程参与方式、pointer 计算和
pipeline 类型；这不是 paged TMA 128 的简单 source-index 替换。

## Runtime Metadata

Paged TMA 路径不传 `cu_seqlens_k`。每个 sequence 的实际 KV 长度来自
`cache_seqlens[batch_idx]`：

```text
batch_idx = row_desc[fa_row_tile_id].batch_idx
m_block   = row_desc[fa_row_tile_id].m_block

q_start = cu_seqlens_q[batch_idx]
q_len   = cu_seqlens_q[batch_idx + 1] - q_start
k_len   = cache_seqlens[batch_idx]

assert 0 < q_len <= k_len

offset  = k_len - q_len
valid_m = min(128, q_len - m_block * 128)
```

`cache_seqlens` 表示当前 request 的完整 KV 前缀长度，单位是 token。`page_table` 的第二维
必须至少覆盖：

```text
ceil_div(k_len, 128)
```

个 logical pages。最后一个 page 可以不满 128 token；未使用 token 必须由 `k_len` mask 屏蔽。

Paged TMA fused kernel 不负责计算或写入 K/V cache。进入 kernel 前，框架或前置 kernel
必须已经完成：

```text
hidden_states @ Wk / Wv
将得到的 K/V 写入 K_cache / V_cache 的 physical page 和 page offset
填好 page_table
填好 cache_seqlens
```

`cache_seqlens[b]` 表示本次 attention 可以读取的有效 KV 前缀长度。如果当前 prefill
chunk 的新 K/V 还没有写入 paged cache，就不能把这些 token 计入 `cache_seqlens[b]`。
否则 FA 会按 page table 读取未定义或旧的 K/V。

## KV Block Loop

Paged TMA 路径的 logical K block range 与 contiguous 路径一致。`n_block` 是
sequence-local logical K block id，范围采用右开区间：

```text
n_block_min <= n_block < n_block_max
```

纯 causal、非 local、非 SplitKV 情况下，bottom-right aligned causal 语义继续使用当前
FA4 风格的 block range 和 mask：

```text
offset       = k_len - q_len
n_block_min = 0

m_idx_max   = (m_block + 1) * 128
n_idx_right = m_idx_max + offset
n_block_max = min(ceil_div(k_len, 128), ceil_div(n_idx_right, 128))
```

这个公式决定当前 Q row tile 需要加载和消费哪些 logical K/V blocks。它会剪掉
完全位于 causal 未来侧的 K/V blocks；block 内仍可能需要逐元素 mask。

对每个进入范围的 logical `n_block`：

```text
physical_page = page_table[batch_idx, n_block]

TMA load:
    K_cache[physical_page, :, kv_head, :]
    V_cache[physical_page, :, kv_head, :]
```

由于 `page_size == 128`，`n_block` 与 logical page id 一一对应，不需要在 tile 内计算
`page_offset`。最后一个 K block 的越界列仍由 `k_len` predicate 屏蔽：

```text
kv_pos = n_block * 128 + col
valid  = kv_pos < k_len
```

causal mask 不变：

```text
q_local  = m_block * 128 + row
kv_local = n_block * 128 + col

kv_local <= q_local + offset
kv_local < k_len
```

这里 causal 比较使用当前 sequence 内局部坐标；`q_start` 只用于定位 Q 的 global row，
不参与 causal 比较。

例如：

```text
page_size = 128
q_len = 256
k_len = 300
offset = 44
```

`m_block = 0` 对应 Q local rows `0..127`，最远可见 K local row `127 + 44 = 171`：

```text
n_block_max = min(ceil_div(300, 128), ceil_div(128 + 44, 128))
            = min(3, 2)
            = 2

处理 logical K blocks [0, 2): block 0 和 block 1
```

其中 block 1 是 causal 边界块，需要逐元素 causal mask。`m_block = 1` 对应 Q local
rows `128..255`，最远可见 K local row `299`，需要处理 blocks `[0, 3)`；block 2
是最后一个不满 page 的块，只允许 `kv_local < 300` 的列参与 softmax/PV。

## GQA 与 TP 责任边界

Paged KV 不改变 GQA 语义。kernel 只消费当前 rank 的 local view：

```text
Q       : [tot_q, H_local,    D]
K_cache : [num_pages, 128, H_kv_local, D]
V_cache : [num_pages, 128, H_kv_local, D]
q_per_kv = H_local / H_kv_local
```

kernel 前置条件：

```text
H_kv_local >= 1
H_local % H_kv_local == 0
q_per_kv = H_local / H_kv_local
```

kernel 内 GQA 寻址固定为：

```text
kv_head = q_head / q_per_kv
```

megaAttention kernel 不负责 KV head replication，也不感知 global `H_q` / `H_kv`。
当模型满足 `global H_kv < TP` 时，例如 Qwen3-235B-A22B 在 TP=8 下：

```text
global H_q  = 64
global H_kv = 4
local H_q   = 8
local H_kv  = 1
```

框架层负责将 Q/K/V projection、paged KV cache 和 page table 准备成当前 rank 的 local
view。框架可以采用 SGLang 风格的 KV head replication，使每个 rank 暴露至少一个
`H_kv_local`。kernel 不参与复制，只按传入的 local `H_local`、`H_kv_local` 和
`q_per_kv` 计算。

因此主设计文档中的 TP 约束需要从：

```text
全局 H_kv 需能被 TP 整除
```

调整为 kernel 层约束：

```text
kernel 不要求 global H_kv 能被 TP 整除。
kernel 只要求框架传入的 local H_local 和 local H_kv_local 满足：
    H_kv_local >= 1
    H_local % H_kv_local == 0
```

## SGLang 接口对齐

SGLang FA4 MHA/GQA prefill 路径会将 token pool 中的 KV cache reshape 为：

```text
key_cache.view(-1, page_size, H_kv_local, D)
value_cache.view(-1, page_size, H_kv_local, Dv)
```

对于本设计，`page_size` 必须为 128：

```text
key_cache   : [num_pages, 128, H_kv_local, D]
value_cache : [num_pages, 128, H_kv_local, D]
page_table  : [B, max_num_pages_per_seq] int32
cache_seqlens: [B] int32
cu_seqlens_q : [B + 1] int32
```

SGLang 的通用 CUDA 默认 `page_size` 可以是 1，但 FA4 MHA/GQA native paged KV 服务路径
要求 `page_size=128`。集成 megaAttention paged TMA 路径时，wrapper 必须在 launch 前
检查 `page_size == 128`，否则不能进入该 kernel variant。

## 与 FlashAttention-4 源码的对应关系

本设计对齐 FA4 SM90 paged TMA 路径：

```text
third_party/flash-attention/flash_attn/cute/interface.py:
    paged_kv_non_tma = page_size not in [None, tile_n]

third_party/flash-attention/flash_attn/cute/flash_fwd_sm90.py:
    self.use_tma_KV = not paged_kv_non_tma

    page_size == tile_n:
        page_idx = mPageTable[batch_idx, n_block]
        tma_load_fn(src_idx=page_idx, ...)

    page_size != tile_n:
        PagedKVManager.load_page_table(n_block)
        PagedKVManager.load_KV(...)
```

本阶段只移植 `page_size == tile_n` 的 TMA path。`PagedKVManager` 对应的一般 page size
路径不进入本阶段。

## 验证要求

第一阶段验证至少覆盖：

```text
page_size = 128
causal = True
D = 128
MHA: H_local == H_kv_local
GQA: H_local > H_kv_local, H_local % H_kv_local == 0
q_len == k_len 完整 prompt prefill
q_len <  k_len chunked prefill / append prefill
最后一个 Q tile 不满 128
最后一个 K page 不满 128
batch 内不同 sequence 长度
page_table 使用非连续、乱序 physical pages
wrapper 对 page_size != 128 拒绝进入 paged_tma_varlen_128
```

数值参考应将 paged KV cache 通过 `page_table` 还原为每条 sequence 的 logical contiguous
K/V 前缀，再与 PyTorch reference 或现有 contiguous reference 对比。

多卡 fused full-chain 验证仍以主设计文档的命令为准：

```bash
torchrun --nproc_per_node=8 tests/fused/test_fused_full_chain.py
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --cases readme --auto
```

新增 paged KV 后，需要同时保留 contiguous-KV 回归，证明新的 paged variant 没有改变
O_scratch、O_proj 和 NVLS AR 协议。
