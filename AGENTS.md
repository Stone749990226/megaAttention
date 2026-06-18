# megaAttention AI 代理系统提示词

你正在开发 `megaAttention`。这是一个尚未完成的 Hopper SM90 kernel 项目，不是通用
attention 库，也不是已经稳定发布的 Python package。

你的核心任务只有一个：**严格依据
`docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md` 开发和验证
causal varlen prefill FlashAttention + O_proj + NVLS AllReduce fused persistent
kernel**。

任何代码修改、测试设计、文档整理、性能分析和解释，都必须服务于这个核心目标。不要在执行一段
时间后脱离该设计文档，自行扩展范围、重写目标或把项目改造成别的形态。

同时，该设计文档本身也可能存在模糊、矛盾、遗漏或可以改进的地方。如果你发现：

- 设计文档和现有代码不一致。
- 某个细节没有写清楚，继续实现会引入不可逆选择。
- 某个同步、调度、layout、pipeline 或 NVLS 协议存在风险。
- 你认为有更好的设计，但会偏离原文。
- 需要在多个实现方案之间取舍。

必须及时停下来，用中文向用户说明问题、影响和可选方案，等用户敲定后再继续。不要为了推进进度
而盲目开发。

## 必须先读的设计文档

开始任何实质开发前，先阅读并对照：

```text
docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md
```

如果你要修改以下内容，必须回到该设计文档核对相关章节：

- varlen row tile metadata。
- causal mask 和 `q_len == k_len` 完整 prompt prefill 语义。
- FA task、O_scratch layout、O_proj task identity。
- persistent scheduler。
- FA -> O_proj ready queue。
- O_proj -> AR owner readiness。
- NVLS AllReduce 的 ready/count/owner 协议。
- 长寿命 pipeline state、mbarrier 复用和 mode 切换 drain 规则。

不要只凭文件名、历史记忆、通用 FlashAttention 经验或其他项目经验做判断。

## 第一版范围

第一版只做：

- Hopper SM90。
- causal attention。
- varlen prefill。
- decoder-only serving 的 prompt 阶段。
- 完整 prompt prefill：每个 sequence 满足 `q_len == k_len`。
- Q token 数量较多的场景。
- FlashAttention 后接 O_proj。
- O_proj 后接 tensor-parallel NVLS AllReduce。
- 使用一个 persistent kernel 把 FA、O_proj、AllReduce 串在同一个 kernel 内。
- Python + CuTe DSL 实现。

第一版明确不做：

- decode。
- append prefill。
- chunked prefill。
- `seqlen_q != seqlen_k` 的尾部对齐 causal mask。
- paged KV。
- SplitKV。
- partial O/LSE combine。
- 非 causal attention。
- Ampere、Ada、Blackwell 或其他非 SM90 架构。
- 通用 public API 封装。

如果用户没有明确要求，不要把这些不在第一版范围内的机制引入实现。

## 当前开发主线

项目围绕同一个 fused kernel 逐步收敛：

```text
src/mega_attention/metadata/row_desc.py
    host 侧 varlen row tile metadata，负责把 flattened row tile 映射回
    (batch_idx, fa_m_block)，并提供 O_scratch / C_sym workspace size 计算。

src/mega_attention/kernels/sm90/fa_varlen.py
src/mega_attention/kernels/sm90/fa_ws.py
    Hopper SM90 FA tile / varlen FA payload 原型。

src/mega_attention/kernels/sm90/oproj_tile.py
    单 CTA O_proj tile microkernel，用于验证 O_scratch row tile @ W_o_local。

src/mega_attention/kernels/sm90/oproj_ar.py
    standalone O_proj GEMM + NVLS AllReduce 验证路径。

src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py
    最终 fused persistent kernel 的落点。目标是按设计文档把 real FA、
    real O_proj、real NVLS AR 接入同一个 persistent scheduler。

src/mega_attention/reference/
    PyTorch 参考实现，用于阶段性数值校验。
```

不要把阶段性 kernel 误包装成稳定 API。当前重点是推进 fused kernel 主线，而不是做对外库设计。

## 实现约束

- 修改 kernel 前，先读相关源文件和核心设计文档对应章节。
- 保持小步修改、小步验证。
- 不做与核心 fused kernel 无关的大规模重构。
- 不为了“工程化”引入会掩盖算法数据流的抽象。
- 如果需要改变设计文档中的关键 invariant，必须先停下来与用户确认。

## 环境和验证

开发与验证目标环境是 Hopper SM90 GPU。

metadata/reference 级别测试可以在 CPU 环境执行，但任何 Hopper kernel 正确性、hang 修复、
性能结论都必须基于实际 Hopper 环境验证。没有实际运行对应测试时，不要声称 Hopper kernel
已经通过。

常用测试：

```bash
pytest tests/metadata
pytest tests/kernels
pytest tests/fused/test_scheduler_skeleton.py
pytest tests/fused/test_fused_fa_path.py
torchrun --nproc_per_node=8 benchmarks/bench_oproj_ar.py --iters 30 --warmup 10
```

根据上下文选择能运行的最小验证集合，并明确说明哪些测试没有运行以及原因。

## 回答和落盘语言

默认使用中文回答用户。

写入文档、计划、注释和说明时也默认使用中文。代码标识符、库名、文件名、API 名称保持英文。

解释 kernel 时优先讲：

- 算法意图。
- 数据流。
- 调度关系。
- 依赖关系和 happens-before。
- 为什么这样组织 pipeline / warp group / task queue。

不要堆砌无关底层变量名；除非用户明确要求，否则不要把解释变成代码逐行翻译。

## 防止偏航

如果你发现当前任务开始偏离核心设计文档，应主动拉回：

```text
当前改动是否推进 causal varlen prefill FA + O_proj + NVLS AR fused persistent kernel？
是否仍满足 Hopper SM90、q_len == k_len、causal、varlen prefill 的第一版范围？
是否需要回到核心设计文档确认？
```

如果答案不明确，先停下来读设计文档；如果读完后仍有模糊、矛盾、风险或设计取舍，先与用户交流，
不要盲目继续实现。
