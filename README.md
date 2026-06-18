# megaAttention

`megaAttention` 是一个 Hopper SM90-only 的实验性 fused serving kernel 项目。

核心目标是在 decoder-only serving 的完整 prompt prefill 阶段，将下面三段计算融合进一个 persistent kernel：

```text
causal varlen FlashAttention -> O_proj -> tensor-parallel NVLS AllReduce
```

第一版聚焦：

- Hopper SM90 GPU。
- causal attention。
- varlen prefill。
- 每个 sequence 满足 `q_len == k_len`。
- Q token 数量较多的 prompt 阶段。
- FA、O_proj、AR 共用 128 行 row tile。
- 使用 Python + CuTe DSL 实现和验证。

第一版不覆盖：

- decode。
- append prefill。
- chunked prefill。
- `seqlen_q != seqlen_k` 的尾部对齐 causal mask。
- Ampere、Ada、Blackwell 或其他架构。
- 非 causal attention。

核心设计文档见：

```text
docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md
```

## 当前状态

| 模块 | 状态 |
| --- | --- |
| row tile metadata / `row_desc` | 已实现 |
| 动态 varlen FA tile | 已验证 |
| O_proj tile microkernel | 已验证 |
| standalone O_proj + NVLS AR | 已验证（8×H200） |
| persistent scheduler skeleton | 已验证 |
| real FA in fused scheduler | 已验证（含 multi_seq；原 finalize warp 发散死锁已修复） |
| real O_proj in fused kernel | 已验证（单卡 tp_size=1，C_sym partial 对 `oproj_reference`） |
| AR owner 调度协议 in fused kernel | 已验证（单卡 tp_size=1：确定性 owner 映射 + owner-local u64 bitset + exactly-once/terminate） |
| real NVLS AR（多卡 multimem reduce/store） | 已验证（8×H200：symmetric C_sym multicast reduce + 跨 rank owner 寻址 + nvl_barrier，整链对 full_chain_reference err~5e-3） |

当前 fused kernel 已在同一个 persistent 调度器内跑通 **real FA + real O_proj + real NVLS AllReduce**：
单卡 `tp_size=1` 与 8×H200 `tp_size=8` 两条路径都验证通过。tp_size=1 下 AR 退化为恒等；
tp_size=8 下 owner 用 `multimem.ld_reduce/st` 在 C_sym multicast view 上做 in-place AllReduce。

### 分阶段验证状态（单卡 H200）

| 阶段 | commit | 验证 |
| --- | --- | --- |
| P0 修 multi_seq 死锁 | `7f711d1` | `test_fused_fa_path` / `test_fa_packed` / `test_fa_varlen` 全过（含 `valid_m%8≠0` 回归） |
| P1 real O_proj 接入 | `1cd547b` | `test_fused_oproj_path` 4 用例（C_sym 对 `oproj_reference`，err~0.0018） |
| P2 AR owner 协议 | `9b028c9` | `test_scheduler_skeleton` 5 passed；fused 两路径全过；exactly-once / 正常终止 |
| P3 多卡 NVLS | 本次 | `test_fused_full_chain`（8×H200，torchrun）：整链 C_sym 对 `full_chain_reference` err~5e-3，AR per-rank owner 计数正确；单卡两路径回归不退化 |

### 性能（8×H200，`benchmarks/bench_fused_fa_oproj_ar.py`）

fused（单 persistent kernel）vs 非融合 best-of-breed 基线
（官方 **`flash_attn_varlen_func`**（flash-attn 2.8.3）+ cuBLAS GEMM + **NVLS `multimem_all_reduce_`**）：

| shape | fused | FA+GEMM+NVLS 基线 | ratio |
| --- | --- | --- | --- |
| [2048,2048] H8 hid2048 (4K) | 0.191 ms | 0.228 ms | **1.20×** |
| [1024]×4 H16 hid2048 (4K) | 0.242 ms | 0.256 ms | 1.06× |
| [4096,4096] H8 hid4096 (8K) | 0.515 ms | 0.725 ms | **1.41×** |
| [8192] H8 hid2048 (8K, 单序列) | 0.619 ms | 0.801 ms | **1.30×** |
| [2048]×8 H8 hid2048 (16K) | 0.576 ms | 0.654 ms | 1.13× |
| [8192,8192] H8 hid2048 (16K) | 1.068 ms | 1.303 ms | **1.22×** |
| [8192,8192] H8 hid4096 (16K) | 1.252 ms | 1.662 ms | **1.33×** |
| [8192,8192] H16 hid7168 (16K, DeepSeek) | 3.075 ms | 3.199 ms | 1.04× |
| [16384] H8 hid2048 (16K, 单序列) | 1.962 ms | 2.199 ms | 1.12× |
| [16384] H16 hid4096 (16K, 单序列) | 4.234 ms | 4.023 ms | 0.95× |
| [16384,16384] H8 hid2048 (32K) | 3.711 ms | 3.850 ms | 1.04× |
| [32768] H8 hid2048 (32K, 单序列) | 7.392 ms | 6.904 ms | 0.93× |

规律：
- **多序列 varlen prefill（目标场景）**：fused 普遍快 **1.1–1.4×**；hidden 越大（O_proj/AR 占比越高）优势越大。
- **长单序列（16K–32K，FA O(L²) compute-bound）**：fused 0.93–1.12×（边际/略负）——长序列下 FA 主导，
  官方 flash-attn 的 FA 比 fused 的 per-tile FA 更优化，融合省下的 launch / 独立 AR 占比变小。
- **hidden=7168（1.04×）**：当前 do_ar 是 correctness-first 的单遍 multimem reduce（无 comm/compute
  overlap），超大 hidden 下成为瓶颈。

两个吃亏区间精确对应已知优化缺口：**FA per-tile 效率** 与 **AR comm/compute overlap**
（standalone `oproj_ar.py` 的 comm warp group 方案已达 ~1.4× over un-fused AR，尚未接入 fused）。

> 备注：用 PyTorch SDPA 的 FLASH 后端（也是 FlashAttention-2）替换官方包时 ratio 更高（如
> [2048,2048] 为 1.43×），因为官方 flash-attn 比 SDPA 后端更快、基线更强；表中取官方包的权威数。
>
> 正确性：benchmark 每个 shape 还会打印 `err_abs/err_rel [OK vs baseline]`——fused 的 C_sym
> 与独立的 FA+GEMM+NVLS 基线路径逐 tile 对比，全部 shape `err_rel` 在 ~5e-4–1e-3（bf16 级），
> 即融合结果与参考路径一致、算得对。

## 目录结构

```text
src/mega_attention/
  metadata/       # host 侧 varlen row tile 描述和 workspace size 计算
  reference/      # PyTorch 参考实现
  kernels/sm90/   # Hopper SM90 CuTe DSL kernel 原型
  runtime/        # 后续放 workspace 分配、launch wrapper、参数校验

tests/
  metadata/       # CPU 可跑的元数据测试
  kernels/        # Hopper 单卡 kernel 测试
  fused/          # persistent scheduler / fused path 阶段测试

benchmarks/       # benchmark driver
scripts/          # profiling、对比、smoke 脚本
docs/             # 设计、状态、外部参考阅读笔记、profiling 文档
scratch/          # 临时执行方案和实验记录
third_party/      # 外部实现参考；flash-attention / DeepGEMM 为固定 submodule，不是运行时依赖
```

## 安装

开发安装：

```bash
pip install -e ".[dev]"
```

核心依赖包括 PyTorch、CuTe DSL、CUDA Python bindings 和 quack kernels。完整 fused / NVLS 路径还需要 Hopper GPU、NCCL NVLS、`torch.distributed._symmetric_memory` 和支持 multicast 的 symmetric memory 环境。

实测可用版本：`nvidia-cutlass-dsl 4.5.0`、`quack-kernels 0.4.1`（`pyproject.toml` 的版本下限以此为准）。

### CUDA 驱动 / forward-compat 注意

如果 PyTorch 是基于 CUDA 13 构建（如 `torch 2.x+cu130`），而机器的 NVIDIA 内核驱动只支持到 CUDA 12.8（如 driver 570.x），直接运行任何 GPU 测试都会报：

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
```

此时若机器已装 CUDA forward-compat 包（H200 等数据中心卡支持），用 compat 目录下的 userspace 驱动即可：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
```

设置后 `torch.cuda.is_available()` 应为 `True` 且识别为 `NVIDIA H200 (9, 0)`。本文下面所有 Hopper / 多卡命令都假设已设置该变量。

## 测试分级

CPU / 普通本机可跑：

```bash
pytest tests/metadata
```

Hopper 单卡可跑：

```bash
pytest tests/kernels
pytest tests/fused/test_scheduler_skeleton.py
```

`tests/fused/test_fused_fa_path.py`、`tests/fused/test_fused_oproj_path.py` 和
`tests/fused/test_fa_packed.py` 目前是 standalone 脚本（只有 `main()`，没有 `test_` 函数），
用 `pytest` 跑会 `collected 0 items`。要跑它们用脚本方式：

```bash
python tests/fused/test_fa_packed.py          # FA payload 全 PASS（含 multi_seq）
python tests/fused/test_fused_fa_path.py       # fused FA 路径全 PASS（含 multi_seq / vm44_300）
python tests/fused/test_fused_oproj_path.py    # fused FA+O_proj+AR 协议，C_sym 对 oproj_reference
```

**已解决**：早期 `multi_seq [200,64,300]` 在 fused scheduler 死锁的问题已在 P0 定位并修复
（根因是 FA finalize 把 warp-collective `warp_reduction_sum` 放在按 `valid_m` 发散的分支里，
`valid_m%8≠0` 时 warp 内发散调用 shuffle 死锁；修复为无条件调用，见 `test_fa_varlen` 的
`vm_split_m44` 与 `test_fused_fa_path` 的 `vm44_300` 回归用例）。

多卡 Hopper + NVSwitch/NVLS 环境：

```bash
torchrun --nproc_per_node=8 benchmarks/bench_oproj_ar.py --iters 30 --warmup 10
```

本机 4070 Laptop GPU 不支持 SM90 CuTe kernel。它适合跑 metadata/reference 级别测试，不适合跑 Hopper kernel 或 NVLS benchmark。

## 开发原则

- 先保持核心算法路径清楚，再考虑对外 API。
- `src/mega_attention/kernels/sm90/` 里的 kernel 是阶段性原型，不承诺稳定导入路径。
- 最终 public API 应该在 real FA + real O_proj + real NVLS AR 全部接入 fused kernel 后再暴露。
- 所有新测试应标明运行前提：CPU、Hopper 单卡或多卡 NVLS。
