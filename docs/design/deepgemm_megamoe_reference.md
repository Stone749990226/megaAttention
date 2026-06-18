# DeepGEMM MegaMoE 核心逻辑阅读笔记

本文面向后续实现“多机通算融合算子”的 agent。目标不是解释整个 DeepGEMM，而是把 MegaMoE 里可复用的核心路径、数据流、同步方式、workspace 管理和代码入口整理成最短阅读地图。

本文是 `third_party/DeepGEMM` 的 MegaMoE 外部参考说明。除非特别说明，下文代码路径都相对：

```text
third_party/DeepGEMM
```

MegaMoE 是 DeepGEMM 中的 SM100 MoE 通算融合实现，只能作为 symmetric memory、GPU-side
barrier、dispatch/combine、workspace 管理和通算融合调度的参考。它不能覆盖
`docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md` 中对本项目第一版
Hopper SM90 causal varlen prefill FA + O_proj + NVLS AR fused persistent kernel 的设计约束。

## 1. 总体目标

MegaMoE 把 MoE 的以下阶段融合进一个 SM100 mega-kernel：

1. EP dispatch：根据 `topk_idx` 把 token 发送到拥有目标 expert 的 rank。
2. Linear1：本 rank 的 local experts 对收到的 token 做 FP8 x FP4 GEMM，输出 `2 * intermediate_hidden`。
3. SwiGLU + topk weight：Linear1 epilogue 里做激活、乘 topk weight，并量化成 FP8，作为 Linear2 输入。
4. Linear2：本 rank 的 local experts 做第二个 FP8 x FP4 GEMM，输出 BF16 hidden。
5. EP combine：把每个 token/topk 的 Linear2 输出写回源 rank，再在源 rank 上对 topk 结果求和。

核心思想是：通信、GEMM、epilogue、combine 都在同一个 kernel 中用不同 warp/warpgroup 并行推进，尽量用 symmetric memory 和 GPU-side barrier 避免 CPU/NCCL 参与。

## 2. 关键代码路径

### Python 用户入口

- `deep_gemm/mega/__init__.py`
  - `get_symm_buffer_for_mega_moe(...)`
    - 对 `num_max_tokens_per_rank` 做 alignment。
    - 调 C++ 计算 symmetric buffer 大小和切片布局。
    - 使用 `torch.distributed._symmetric_memory.empty` 分配每 rank 同形同大小 buffer。
    - `symm_mem.rendezvous(...)` 得到所有 rank 的 buffer 指针。
    - 暴露 `buffer.x`, `buffer.x_sf`, `buffer.topk_idx`, `buffer.topk_weights`, `buffer.l1_acts`, `buffer.l1_acts_sf`, `buffer.l2_acts`, `buffer.l2_acts_sf`。
  - `transform_weights_for_mega_moe(...)`
    - L1 权重 gate/up interleave。
    - L1/L2 scale-factor 转成 UTCCP 需要的布局。
  - `fp8_fp4_mega_moe(...)`
    - 把 output、weights、symmetric buffer、rank 信息传给 C++ API。

### C++ API 和 JIT

- `csrc/apis/mega.hpp`
  - `get_symm_buffer_size_for_mega_moe(...)`
    - 构造 `layout::Workspace`。
    - 依次排布 input、L1 pool、L2 pool、combine buffer。
    - 返回总 buffer bytes 和 Python tensor view slicer。
  - `fp8_fp4_mega_moe(...)`
    - 校验 shape/layout/dtype。
    - 从 raw symmetric buffer 切出各段 tensor。
    - 只支持 `arch_major == 10`，然后调用 SM100 JIT kernel wrapper。

- `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`
  - `SM100FP8FP4MegaMoERuntime`
    - 根据 runtime shape/config 生成模板实例化代码。
    - 构造 TMA descriptors。
    - 创建 `layout::SymBuffer<>(sym_buffer_ptrs, rank_idx)`，用于 kernel 内做 rank 地址映射。
    - launch 参数：`grid_dim = num_sms`，`cluster_dim = 2`。

- `csrc/jit_kernels/heuristics/mega_moe.hpp`
  - `get_mega_moe_config(...)`
    - 按 expected tokens per expert 选择 `BLOCK_M`、`STORE_BLOCK_M`、epilogue warps。
    - 固定 `BLOCK_N = 128`, `BLOCK_K = 128`。
    - 选择 `num_experts_per_wave`，让 L1/L2 wave 有足够 blocks 填满 SM。
    - 计算 shared memory pipeline stages 和线程布局。

### Device 端核心

- `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
  - 主 kernel：`sm100_fp8_fp4_mega_moe_impl(...)`
  - 是最重要的文件。读它时按 warp role 分段看，不要从头线性硬读。

- `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`
  - `layout::Workspace`
    - 定义 workspace 内各段 buffer 和指针计算。
  - `layout::Buffer`
    - rank/token 维度的 buffer view。
  - `TokenSrcMetadata`
    - combine 写回源 rank 时需要的 `{rank_idx, token_idx, topk_idx}`。

- `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh`
  - `layout::SymBuffer`
    - 保存本 rank base 和每个 rank 相对本 rank 的 offset。
    - `map(ptr, dst_rank)` 把“本 rank 上的某个 symmetric buffer 地址”映射到目标 rank 对应地址。

- `deep_gemm/include/deep_gemm/comm/barrier.cuh`
  - `grid_sync(...)`
    - kernel 内 rank-local grid barrier。
  - `nvlink_barrier(...)`
    - rank 间 barrier。SM0 负责用 system-scope release red-add 给所有 rank signal，再 acquire poll 本地 signal。

- `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
  - `MegaMoEScheduler`
    - 根据每个 local expert 收到的 token 数，生成 Linear1/Linear2 block 序列。
    - 用 `BlockPhase::{Linear1, Linear2}` 表示当前 block 属于哪一层。

## 3. Symmetric buffer 布局

`csrc/apis/mega.hpp::get_symm_buffer_size_for_mega_moe` 和 `layout::Workspace` 要一起看。每个 rank 分配同样大小的 symmetric buffer，内部大致是：

```text
workspace
  barrier/grid-sync/nvlink signal
  expert_send_count[num_experts]
  expert_recv_count[num_experts]            // 实际按 rank/local_expert 展开
  expert_recv_count_sum[num_experts_per_rank]
  l1_arrival_count[num_max_pool_blocks]
  l2_arrival_mask[num_max_pool_blocks]
  src_token_topk_idx[local_expert][src_rank][slot]
  token_src_metadata[num_max_pool_tokens]

registered input views
  x
  x_sf
  topk_idx
  topk_weights

local pooled activations
  l1_acts
  l1_acts_sf
  l1_topk_weights
  l2_acts
  l2_acts_sf

combine buffer
  combine_token_buffer[topk_slot][token]
```

使用方式：

- Python 每次调用前把本 rank input copy 到 `buffer.x/x_sf/topk_idx/topk_weights`。
- kernel 内通过 `SymBuffer::map` 对远端 rank 的同 layout 地址做普通 global load/store/atomic。
- `src_token_topk_idx` 和 `token_src_metadata` 通常不清零，下次按有效 slot 覆盖。
- 计数器和 arrival flag/mask 会在 kernel 内定向清零，避免下次使用读到旧状态。

## 4. Kernel 线程角色

主 kernel 的线程布局由 `get_mega_moe_config` 生成。下面的默认值是线程数，实际 warp 数再除以 32：

```text
dispatch group        : kNumDispatchThreads，默认 128 threads
non-epilogue group    : kNumNonEpilogueThreads，默认 128 threads
epilogue/combine group: kNumEpilogueThreads，根据 block_m 选择，128 或 256 threads 等
cluster size          : 2 CTA
grid                  : num_sms CTAs
```

在 `sm100_fp8_fp4_mega_moe.cuh` 中按 `warp_idx` 分工：

- `warp_idx < kNumDispatchWarps`
  - 做 dispatch 统计、远端索引写入、远端 token pull、workspace 清理。

- `warp_idx == kNumDispatchWarps`
  - GEMM A operand / activation TMA load warp。
  - Linear1 等待 `l1_arrival_count`。
  - Linear2 等待 `l2_arrival_mask`。

- `warp_idx == kNumDispatchWarps + 1`
  - GEMM B operand / weight TMA load warp。

- `warp_idx == kNumDispatchWarps + 2`
  - leader CTA 上 issue SM100 UMMA。

- `warp_idx == kNumDispatchWarps + 3`
  - 保留/寄存器调整，实际主要用于保持 warp role 布局。

- `warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps`
  - epilogue warps。
  - Linear1 epilogue：TMEM -> SwiGLU/topk weight -> FP8 store 到 `l2_acts/l2_acts_sf`，并 signal `l2_arrival_mask`。
  - Linear2 epilogue：TMEM -> BF16 -> 写远端 `combine_token_buffer`。
  - Linear2 全部远端写完成后进入 combine：读本 rank combine buffer，对 topk 求和，写最终 `y`。

## 5. 执行阶段详解

### 5.1 初始化

位置：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`

主要动作：

- 预取 TMA descriptors。
- 构造 `layout::Workspace` 和各个 `layout::Buffer` view。
- 计算 shared memory 区域：dispatch shared count、TMA stage buffer、scale-factor buffer、barriers、TMEM pointer。
- `cluster_sync_with_relaxed_arrive()`，因为 2-CTA tensor memory allocation 需要 cluster 同步。
- 初始化 dispatch barriers、GEMM full/empty barriers、TMEM barriers、combine barriers。
- 创建 `MegaMoEScheduler`。

### 5.2 Dispatch 统计与远端索引发布

位置：dispatch warps 分支。

核心步骤：

1. 遍历本 rank 的 `topk_idx`。
2. 在 shared memory 中按 global expert 统计 token 数。
3. 每个 SM 用 `atomic_add(workspace.get_expert_send_count_ptr(i), send_value)` 获得 per-expert slot offset。
   - `send_value = (1ull << 32) | local_count`
   - low 32 bit 累加 token count。
   - high 32 bit 统计有多少 SM 提交了这个 expert 的计数。
4. 再遍历 topk，把 `token_topk_idx` 写到目标 expert 所在 rank 的 `src_token_topk_idx`：

```cpp
dst_rank_idx = expert_idx / kNumExpertsPerRank;
dst_ptr = workspace.get_src_token_topk_idx_ptr(local_expert, src_rank, dst_slot);
*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
```

5. rank-local `grid_sync`。
6. `sm_idx == 0` 把 finalized expert count 写到远端：
   - `expert_recv_count[src_rank][dst_local_expert]`
   - `expert_recv_count_sum[dst_local_expert]`
7. `nvlink_barrier(kBeforeDispatchPullBarrierTag)`，保证所有 rank 的 dispatch 元数据都可见。

### 5.3 Pull token 到本地 expert pool

位置：dispatch warps 分支，`Barrier before pulling` 之后。

核心逻辑：

1. `scheduler.fetch_expert_recv_count()` 等待每个 local expert 的 `expert_recv_count_sum` finalized。
   - 判断条件是 high 32 bit 等于 `kNumSMs * kNumRanks`。
2. 对本 rank local experts，按 rank/token slot 拉取远端 token：
   - 从 `workspace.get_src_token_topk_idx_ptr(local_expert, src_rank, slot)` 取 `src_token_topk_idx`。
   - 从远端 input buffer 读 token 和 SF。
   - 写入本地 `l1_token_buffer/l1_sf_buffer/l1_topk_weights_buffer`。
3. 写 `workspace.get_token_src_metadata_ptr(pool_token_idx)`，记录 combine 回写需要的源 rank、源 token、topk slot。
4. 每写完一个 token 到 L1 pool，对对应 pool block 做：

```cpp
ptx::red_add_rel(workspace.get_l1_arrival_count_ptr(pool_block), 1);
```

GEMM A loader 会等 `l1_arrival_count == valid_m` 才开始 Linear1。

### 5.4 Scheduler 的 Linear1/Linear2 block 序列

位置：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`

`MegaMoEScheduler::for_each_block` 每个参与 GEMM 的 warp role 都会独立调用，但它们根据相同的 workspace counts 和相同的 `blockIdx.x` 生成同样的 block 序列。

状态机：

```text
Linear1 wave over kNumExpertsPerWave experts
  -> Linear2 wave over same experts
  -> next Linear1 wave
  -> next Linear2 wave
  -> ...
```

每个 block 输出：

```cpp
BlockPhase block_phase;        // Linear1 or Linear2
local_expert_idx;
num_k_blocks;
m_block_idx;
n_block_idx;
```

Linear1 和 Linear2 都用同一套 persistent schedule，但等待条件不同：

- Linear1 A loader 等 `l1_arrival_count[pool_block] == valid_m`。
- Linear2 A loader 等 `l2_arrival_mask[pool_block] == expected_mask`，代表 Linear1 对该 block 的所有 N chunks 已经写入 L2 activation。

### 5.5 GEMM pipeline

位置：non-epilogue warps 和 epilogue warps。

数据流：

```text
A loader:
  l1_acts/l2_acts + SF -> smem_a/smem_sfa via TMA

B loader:
  l1_weights/l2_weights + SF -> smem_b/smem_sfb via TMA

MMA issue warp:
  wait full_barrier
  UTCCP copy scale factors to TMEM
  issue SM100 MXF8F6F4 2x1SM UMMA
  commit empty/full barriers

Epilogue warps:
  wait tmem_full_barrier
  consume TMEM
  Linear1 or Linear2 epilogue
```

Linear1 epilogue：

- 从 TMEM 读 `gate/up`。
- BF16 clamp（可选）。
- `SwiGLU(gate) * up * topk_weight`。
- 做 amax，生成 FP8 scale factor。
- 写 `l2_acts` 和 `l2_acts_sf`。
- 最后 `red_or_rel_gpu(workspace.get_l2_arrival_mask_ptr(pool_block), 1ull << n_block_idx)` 通知 Linear2。

Linear2 epilogue：

- 从 TMEM 读 BF16 output。
- 根据 `token_src_metadata` 找到源 rank、源 token、topk slot。
- 写远端 `combine_token_buffer[topk_slot][token]`：

```cpp
*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
```

### 5.6 Combine

位置：epilogue warps 在所有 Linear2 远端写完成之后。

步骤：

1. `nvlink_barrier(kBeforeCombineReduceBarrierTag)`。
   - 保证所有 rank 对远端 `combine_token_buffer` 的写入完成并可见。
2. 和 dispatch warps 在 `kDispatchWithEpilogueBarrierIdx` 会合。
   - epilogue 进入 combine。
   - dispatch 开始清理 workspace。
3. epilogue warps 遍历本 rank 的 `token_idx`。
4. 对每个 chunk，按 `topk_idx` 读取 `combine_token_buffer[topk_slot][token]`。
5. FP32 累加 topk 贡献，cast BF16，TMA store 到最终 `y[token]`。

## 6. 跨 rank 同步机制

MegaMoE 使用 `comm::nvlink_barrier` 做阶段级跨 rank barrier，不使用 megaAttention 那种 per-tile `multimem_red_add1` flag。

位置：`deep_gemm/include/deep_gemm/comm/barrier.cuh`

### grid_sync

`grid_sync` 是 rank 内所有 SM 的同步：

- 调用传入的 `sync_scope()`，通常是 named barrier。
- `thread_idx == 0` 对 workspace grid counter 做 release atomic add。
- 轮询 counter 的 high-bit phase。
- 再调用一次 `sync_scope()`。

### nvlink_barrier

执行结构：

```text
optional grid_sync
if sm_idx == 0:
  thread_idx < kNumRanks:
    red.release.sys.global.add.s32(remote_rank_signal, +1 or -1)
  thread_idx == 0:
    update local barrier counter
    poll local signal with ld.acquire.sys until target
optional grid_sync
```

关键点：

- 只有 SM0 参与跨 rank 信号，避免所有 SM 都发远端 arrival。
- `sym_buffer.map(signal_ptr, dst_rank)` 把本地 signal 地址映射成目标 rank 的同 offset 地址。
- 每个 rank 都向所有 rank signal，包括自己。
- signal 使用两个 phase slot，并交替 `+1` / `-1`：

```text
phase0 +1 -> wait num_ranks
phase1 +1 -> wait num_ranks
phase0 -1 -> wait 0
phase1 -1 -> wait 0
```

这样避免单独 reset signal 导致竞态。

MegaMoE 里主要有三个 NVLink barrier tag：

- `kBeforeDispatchPullBarrierTag`
  - dispatch 元数据写完后，pull token 前。
- `kBeforeCombineReduceBarrierTag`
  - Linear2 远端 combine buffer 写完后，combine 前。
- `kAfterWorkspaceCleanBarrierTag`
  - workspace 清理完后，kernel 结束前。

## 7. Workspace 清理

位置：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`

清理不是 kernel 末尾全量 memset，而是在 dispatch warps 完成 pull 后，与 epilogue combine 重叠执行。

触发点：

```text
dispatch pull 完成
dispatch 和 epilogue 在 kDispatchWithEpilogueBarrierIdx 会合
dispatch 清理 workspace
epilogue 同时 combine
清理完成后 nvlink_barrier(kAfterWorkspaceCleanBarrierTag)
```

清理内容：

- SM0 清 `expert_send_count`。
- 其他 SM 分摊 local experts：
  - 读取 `expert_recv_count_sum` 得到 `num_recv_tokens`。
  - 可选累加 `cumulative_local_expert_recv_stats`。
  - 清 `expert_recv_count_sum`。
  - 清每个 src rank 的 `expert_recv_count`。
  - 根据 `num_recv_m_blocks` 清 `l1_arrival_count` 和 `l2_arrival_mask`。

不清：

- `src_token_topk_idx`
- `token_src_metadata`

原因：它们按本次有效 token/pool slot 覆盖，后续逻辑由 count/arrival 控制有效范围。

## 8. 对多机通算融合算子的可借鉴点

### 8.1 适合复用的模式

- symmetric buffer + rank-relative pointer mapping
  - `SymBuffer::map` 是跨 rank 直接访存的核心抽象。
  - 后续多机版本可保留“同 layout buffer + rank offset map”的思想，但跨节点不一定能直接用 NVLink P2P，需要替换通信后端。

- 阶段级 GPU-side barrier
  - `nvlink_barrier` 适合低频阶段边界。
  - 如果新算子需要 tile/batch 级高频同步，MegaMoE 这个 barrier 可能太重；可参考 megaAttention 的 per-batch `multimem_red_add1 + acquire spin`。

- dispatch metadata 和 payload 分离
  - 先写 token slot metadata/count。
  - barrier 后再根据 metadata pull payload。
  - 这比所有 rank blindly all-to-all payload 更可控。

- arrival count/mask 作为 pipeline readiness
  - L1 用 count：每个 token 到达后 `+1`。
  - L2 用 bitmask：每个 N block 完成后 OR 一个 bit。
  - 这种方式适合 producer/consumer 在同一 kernel 内解耦。

- workspace 定向清理
  - 只清会影响下一轮同步/调度的状态。
  - 大 payload/metadata 不 memset，靠 count 限定有效范围。

### 8.2 多机扩展时需要替换或重新验证的点

- `SymBuffer::map` 假设所有 rank 的 symmetric memory 能通过同一 GPU address space/P2P offset 直接访问。
  - 跨机通常不能这样直接 store/load 远端 GPU memory。
  - 需要 RDMA/NVSHMEM/NCCL user buffer/register memory 或 proxy kernel 方案。

- `nvlink_barrier` 的 `red.release.sys.global.add` 是面向可直接访问 remote GPU memory 的节点内同步。
  - 跨机时 system-scope release/acquire 是否覆盖 NIC/RDMA 可见性，需要按具体通信栈重建语义。

- `combine_token_buffer` 远端写是普通 `*sym_buffer.map(dst_ptr, dst_rank) = packed`。
  - 多机要换成网络 write 或分层 combine。

- 当前 scheduler 假设所有 local experts 在本 rank 上执行，并且每 rank 的 `num_experts_per_rank = num_experts / num_ranks`。
  - 多机/多节点可能需要 node-local rank、global rank、expert placement 三层索引。

- 当前 barrier 是 all ranks collective。
  - 多机下最好区分 node-local barrier、inter-node barrier、global barrier，避免所有阶段都打全局同步。

## 9. 最短阅读顺序

建议 agent 按这个顺序读：

1. `README.md` 的 Mega MoE 小节，理解使用方式。
2. `deep_gemm/mega/__init__.py`，理解 Python 层 buffer 和权重转换。
3. `csrc/apis/mega.hpp`，理解 symmetric buffer 切片和 C++ API 检查。
4. `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`，理解 workspace 各段含义。
5. `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh`，理解跨 rank 地址映射。
6. `deep_gemm/include/deep_gemm/comm/barrier.cuh`，理解 grid/NVLink barrier。
7. `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`，理解 Linear1/Linear2 block 状态机。
8. `csrc/jit_kernels/heuristics/mega_moe.hpp`，理解 tiling/thread/shared-memory config。
9. `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`，理解 TMA descriptor 和 launch。
10. `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`，按 warp role 阅读 kernel 主体。
11. `tests/test_mega_moe.py`，对照 baseline 和 correctness/benchmark 调用。

## 10. 后续实现时的检查清单

- 是否明确区分 metadata、payload、arrival signal、barrier signal？
- 每个跨 rank 写入是否有匹配的 release/acquire 或更强同步？
- 每个 wait 的 target 是 count、bitmask 还是 phase？是否能被上一轮残留污染？
- workspace 哪些字段必须清零，哪些可以靠有效范围覆盖？
- scheduler 是否能在所有参与 warp role 中生成完全一致的 block 序列？
- 如果通信后端从 NVLink symmetric memory 换成跨机网络，哪些普通 load/store/atomic 必须替换？
- barrier 粒度是阶段级、batch 级还是 tile 级？当前 `nvlink_barrier` 只适合低频阶段级。
- 是否需要 node-local fast path + inter-node slow path 的分层同步？
