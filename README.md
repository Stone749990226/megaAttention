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
| O_proj tile microkernel | 已验证/开发中 |
| standalone O_proj + NVLS AR | 已验证 |
| persistent scheduler skeleton | 已验证 |
| real FA in fused scheduler | 单序列已验证；multi_seq 死锁（见测试分级"已知问题"）|
| real O_proj in fused kernel | 待完成 |
| real NVLS AR in fused kernel | 待完成/迁移 |

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
docs/             # 设计、状态、profiling 文档
scratch/          # 临时执行方案和实验记录
third_party/      # 外部实现参考；flash-attention 为固定 submodule，不是运行时依赖
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

`tests/fused/test_fused_fa_path.py` 和 `tests/fused/test_fa_packed.py` 目前是 standalone 脚本
（只有 `main()`，没有 `test_` 函数），用 `pytest` 跑会 `collected 0 items`、什么都不验证。要跑它们用脚本方式：

```bash
python tests/fused/test_fa_packed.py        # FA payload 全 PASS（含 multi_seq）
python tests/fused/test_fused_fa_path.py     # 见下方已知问题：multi_seq 会死锁
```

**已知问题（当前卡点）**：`test_fused_fa_path.py` 的单序列用例（`uniform_128` R=1、`varlen_200` R=2）
通过；但 `multi_seq [200,64,300]`（R=6，row tile 跨多个序列）在 fused persistent scheduler 里**死锁**
——kernel 正常编译、正常 launch，但 `torch.cuda.synchronize()` 永不返回、GPU 持续 100%。对照实验确认
FA payload 本身（`test_fa_packed.py` 同一输入 R=6）可正常通过，所以问题定位在 scheduler 在多 row-tile /
多序列下的 ready/同步路径。real FA 接入 fused kernel 仅在单序列下验证通过。

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
