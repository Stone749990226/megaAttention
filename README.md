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
device time（30 iters / 10 warmup）。下表为 `--cases readme` 的原始输出（`print_table` 全列）。

shape 列即 per-rank 形状，按 `H<q_head>/kv<kv_head> hid<hidden>` 对应模型：`H8/kv1 hid4096`
= Qwen3-235B-A22B；`H12/kv1 hid6144` = Qwen3-Coder-480B；`H12/kv1 hid5120` = GLM-4.6；
`H16/kv1 hid16384` = Llama-3.1-405B；`H16/kv2 hid8192` = 合成 GQA stress；末两行 `q:... k=N×q`
= Qwen3-235B 的 q&lt;k contiguous-KV chunked/append prefill（KV 前缀 = N× Q chunk）。

| shape | tp | fused | FA | O_proj | AR | serial | ideal | overlap% | TFLOPS | NVLink | ratio | err_rel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| varlen(B=8,tot=22.6K,max=7.5K) H8/kv1 hid4096 | 8 | 1.262 | 0.755 | 0.251 | 0.716 | 1.723 | 1.006 | 64% | 340 (34%) | 263 (29%) | 1.37x | 7.1e-04 |
| varlen(B=6,tot=13.4K,max=6.8K) H8/kv1 hid4096 | 8 | 0.787 | 0.487 | 0.161 | 0.428 | 1.076 | 0.648 | 68% | 310 (31%) | 251 (28%) | 1.37x | 8.2e-04 |
| varlen(B=8,tot=22.6K,max=7.5K) H12/kv1 hid6144 | 8 | 2.257 | 1.055 | 0.567 | 1.054 | 2.675 | 1.621 | 40% | 350 (35%) | 221 (25%) | 1.19x | 8.5e-04 |
| varlen(B=6,tot=13.4K,max=6.8K) H12/kv1 hid6144 | 8 | 1.355 | 0.652 | 0.340 | 0.636 | 1.628 | 0.992 | 43% | 334 (34%) | 218 (24%) | 1.20x | 7.7e-04 |
| varlen(B=8,tot=22.6K,max=7.5K) H12/kv1 hid5120 | 8 | 2.006 | 1.061 | 0.444 | 0.889 | 2.393 | 1.505 | 44% | 357 (36%) | 207 (23%) | 1.19x | 8.1e-04 |
| varlen(B=6,tot=13.4K,max=6.8K) H12/kv1 hid5120 | 8 | 1.216 | 0.654 | 0.285 | 0.528 | 1.467 | 0.939 | 48% | 337 (34%) | 203 (23%) | 1.21x | 9.3e-04 |
| varlen(B=8,tot=22.6K,max=7.5K) H16/kv1 hid16384 | 8 | 5.165 | 1.355 | 1.981 | 2.779 | 6.115 | 3.336 | 34% | 392 (40%) | 257 (29%) | 1.18x | 1.3e-03 |
| varlen(B=6,tot=13.4K,max=6.8K) H16/kv1 hid16384 | 8 | 3.063 | 0.826 | 1.177 | 1.659 | 3.663 | 2.004 | 36% | 385 (39%) | 258 (29%) | 1.20x | 1.5e-03 |
| varlen(B=8,tot=22.6K,max=7.5K) H16/kv2 hid8192 | 8 | 2.868 | 1.354 | 0.992 | 1.398 | 3.743 | 2.345 | 63% | 435 (44%) | 232 (26%) | 1.31x | 1.5e-03 |
| varlen(B=6,tot=13.4K,max=6.8K) H16/kv2 hid8192 | 8 | 1.698 | 0.821 | 0.592 | 0.835 | 2.248 | 1.413 | 66% | 423 (43%) | 232 (26%) | 1.32x | 8.7e-04 |
| q:varlen(B=6,tot=13.4K,max=6.8K) k=2xq H8/kv1 hid4096 | 8 | 1.250 | 1.226 | 0.161 | 0.425 | 1.812 | 1.387 | 132% | 401 (41%) | 158 (18%) | 1.45x | 7.8e-03 |
| q:varlen(B=6,tot=13.4K,max=6.8K) k=4xq H8/kv1 hid4096 | 8 | 2.239 | 2.711 | 0.161 | 0.428 | 3.299 | 2.871 | 248% | 453 (46%) | 88 (10%) | 1.47x | 6.1e-03 |

时间单位 ms；`serial = FA+O_proj+AR`（best-of-breed 分段串行下界），`ideal = max(FA+O_proj, AR)`
（完美重叠下界），`overlap% = (serial-fused)/(serial-ideal)`。`TFLOPS`/`NVLink` 均除以 fused
墙钟（NVLink 为 bus BW=2(n-1)/n 口径），括号内为对 H200 峰值的 util。fused 相对串行基线
全面 **1.18–1.47×（ratio 列）**；大 hidden（Llama hid16384）下 AR comm 占比高、overlap 偏低，
是已知瓶颈。q&lt;k 两行 FA 计算随 KV 增长占比变大、AR/O_proj 相对下降，overlap% 可超 100%
（即融合连完美重叠的最强基线都打赢，口径见 bench docstring）。完整用例集见
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
third_party/      # 外部参考（flash-attention / flashinfer / DeepGEMM submodule，非运行时依赖）
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
