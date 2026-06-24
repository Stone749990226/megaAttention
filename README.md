# megaAttention

`megaAttention` 是一个 Hopper SM90-only 的实验性 fused serving kernel 项目。

核心目标是在 decoder-only serving 的完整 prompt prefill 阶段，把下面三段计算融合进**一个 persistent kernel**：

```text
causal varlen FlashAttention -> O_proj -> tensor-parallel NVLS AllReduce
```

FA、O_proj、AR 共用同一个 128 行 row tile 作为 task identity，由同一个 persistent
scheduler 调度，用 Python + CuTe DSL 实现。

聚焦场景：Hopper SM90、causal attention、varlen prefill、Q token 数量较多。

核心设计文档：

```text
docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md
```

## 已完成

主线开发已整体完成：`real FA + real O_proj + real NVLS AllReduce` 已在同一个 persistent
调度器内跑通，单卡 `tp_size=1`（AR 退化为恒等）与 8×H200 `tp_size=8`（owner 用
`multimem.ld_reduce/st` 在 C_sym multicast view 上做 in-place AllReduce）两条路径均验证通过。

- host 侧 varlen row tile metadata 与 workspace size 计算（`row_desc`）。
- 动态 varlen FA tile、O_proj tile microkernel、standalone O_proj + NVLS AR 验证路径。
- persistent scheduler：FA task counter → O_proj ready queue → AR owner readiness，
  exactly-once + 正常终止。
- 标准 GQA（K/V 按 `kv_head = q_head // q_per_kv` 复用）。
- host 侧 launch heuristic（按 FA/O_proj MAC 比分桶选 role 权重与 super-group 数）。
- 正确性：fused 的 C_sym 对 best-of-breed 串行基线逐 tile 对比，全 shape `err_rel` 在
  bf16 级（~4e-4–2e-3）。

## Roadmap

当前范围之外、尚未实现，按需推进：

- paged KV、SplitKV、partial O/LSE combine、非 causal attention。
- FA per-tile pipeline 进一步优化（长单序列 compute-bound 场景）。
- AR comm/compute overlap 优化（大 hidden 下 AR 占比偏高）。
- 稳定 public API 封装与 `runtime/`（workspace 分配、launch wrapper、参数校验）。

## 性能

8×H200，`benchmarks/bench_fused_fa_oproj_ar.py --cases readme --auto`，fused（单 persistent
kernel）vs 非融合 best-of-breed 基线（官方 `flash_attn_varlen_func` + cuBLAS GEMM + NVLS
`multimem_all_reduce_`）。锚定大模型 TP8 per-rank 形状 + 多段不规整 varlen，kineto self
device time（30 iters / 10 warmup）：

| 模型 (per-rank @ TP8) | batch (tot tokens) | fused ms | overlap% | TFLOPS (util) | NVLink GB/s | ratio | err_rel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-235B-A22B hid4096 | varlen B8 ~22.6K | 1.262 | 64% | 340 (34%) | 263 | **1.37×** | 7.1e-4 |
| Qwen3-235B-A22B hid4096 | varlen B6 ~13.4K | 0.787 | 68% | 310 (31%) | 251 | **1.37×** | 8.2e-4 |
| Qwen3-Coder-480B hid6144 | varlen B8 ~22.6K | 2.257 | 40% | 350 (35%) | 221 | **1.19×** | 8.5e-4 |
| Qwen3-Coder-480B hid6144 | varlen B6 ~13.4K | 1.355 | 43% | 334 (34%) | 218 | **1.20×** | 7.7e-4 |
| GLM-4.6 hid5120 | varlen B8 ~22.6K | 2.006 | 44% | 357 (36%) | 207 | **1.19×** | 8.1e-4 |
| GLM-4.6 hid5120 | varlen B6 ~13.4K | 1.216 | 48% | 337 (34%) | 203 | **1.21×** | 9.3e-4 |
| Llama-3.1-405B hid16384 | varlen B8 ~22.6K | 5.165 | 34% | 392 (40%) | 257 | **1.18×** | 1.3e-3 |
| Llama-3.1-405B hid16384 | varlen B6 ~13.4K | 3.063 | 36% | 385 (39%) | 258 | **1.20×** | 1.5e-3 |
| 合成 stress hid8192 | varlen B8 ~22.6K | 2.868 | 63% | 435 (44%) | 232 | **1.31×** | 1.5e-3 |
| 合成 stress hid8192 | varlen B6 ~13.4K | 1.698 | 66% | 423 (43%) | 232 | **1.32×** | 8.7e-4 |
| Qwen3-235B q<k hid4096 | B6 k=2×q | 1.250 | 132% | 401 (41%) | 158 | **1.45×** | 7.8e-3 |
| Qwen3-235B q<k hid4096 | B6 k=4×q | 2.239 | 248% | 453 (46%) | 88 | **1.47×** | 6.1e-3 |

fused 相对 best-of-breed 串行基线全面 **1.18–1.47×**；大 hidden（如 Llama hid16384）下 AR
comm 占比高、overlap 偏低，是已知瓶颈。末两行为 q&lt;k contiguous-KV chunked/append prefill
（KV 前缀 = 2×/4× Q chunk）：FA 计算随 KV 增长而占比变大、AR/O_proj 相对下降，overlap% 可超
100%（即融合连完美重叠的最强基线都打赢，口径见 bench docstring）。完整用例集见
`bench_fused_fa_oproj_ar.py` 的 `README_CASES`。

## 目录结构

```text
src/mega_attention/
  metadata/       # host 侧 varlen row tile 描述、workspace size、launch heuristic
  reference/      # PyTorch 全链路参考实现（数值校验）
  kernels/sm90/   # Hopper SM90 CuTe DSL kernel
    fa_varlen.py / fa_ws.py       # FA tile / varlen FA payload
    oproj_tile.py                 # 单 CTA O_proj tile microkernel
    oproj_ar.py                   # standalone O_proj GEMM + NVLS AR 验证路径
    fused_fa_oproj_ar.py          # 最终 fused persistent kernel
  runtime/        # 占位：后续放 workspace 分配、launch wrapper、参数校验

tests/
  metadata/       # CPU 可跑的元数据 / launch heuristic 测试
  kernels/        # Hopper 单卡 kernel 测试
  fused/          # persistent scheduler / fused path 测试

benchmarks/       # benchmark driver、launch-config sweep
scripts/          # profiling、对比、smoke 脚本
docs/             # 设计、状态、外部参考阅读笔记、profiling 文档
third_party/      # 外部参考（flash-attention / DeepGEMM submodule，非运行时依赖）
```

## 安装

```bash
pip install -e ".[dev]"
```

核心依赖：PyTorch、CuTe DSL、CUDA Python bindings、quack kernels。完整 fused / NVLS 路径还需
Hopper GPU、NCCL NVLS、`torch.distributed._symmetric_memory` 和支持 multicast 的 symmetric
memory 环境。实测可用版本：`nvidia-cutlass-dsl 4.5.0`、`quack-kernels 0.4.1`。

### CUDA 驱动 / forward-compat

如果 PyTorch 基于 CUDA 13 构建（如 `torch 2.x+cu130`），而机器内核驱动只支持到更低版本，运行
GPU 测试会报 `The NVIDIA driver on your system is too old`。此时若已装 CUDA forward-compat 包
（H200 等数据中心卡支持），加上 compat 目录到 `LD_LIBRARY_PATH` 即可（两种布局都加上）：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/compat/lib:$LD_LIBRARY_PATH
# 若仍报错，直接定位 compat 里的 libcuda.so.1：
export LD_LIBRARY_PATH="$(dirname "$(find /usr/local/cuda*/compat -name 'libcuda.so.1' | head -1)")":$LD_LIBRARY_PATH
```

## 测试

目标环境是多卡 Hopper + NVSwitch/NVLS。核心 kernel 整链测试与 benchmark：

```bash
torchrun --nproc_per_node=8 tests/fused/test_fused_full_chain.py   # 整链对 full_chain_reference
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --cases readme --auto
```

## 开发原则

- 先保持核心算法路径清楚，再考虑对外 API。
- `kernels/sm90/` 里 `fa_varlen` / `fa_ws` / `oproj_tile` / `oproj_ar` 是验证用原型，最终路径以
  `fused_fa_oproj_ar.py` 为准；不承诺稳定导入路径。
