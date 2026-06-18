# 外部实现参考索引

本文列出 `megaAttention` 当前允许参考的外部实现、对应源码位置和本仓库阅读笔记。

外部实现只能作为局部机制参考，不能覆盖核心设计文档：

```text
docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md
```

如果外部实现与核心设计文档在 scope、layout、调度、同步协议或架构假设上冲突，以核心设计文档为准；
需要改变本项目 invariant 时，先停下来和用户确认。

## FlashAttention-4

源码位置：

```text
third_party/flash-attention
```

适合参考：

- Hopper SM90 FlashAttention forward tile。
- CuTe DSL pipeline。
- TMA / WGMMA / mbarrier 协作。
- varlen block info。
- causal mask。
- online softmax。
- FA tile scheduler 的 launch shape 和 warp-group 组织。

本仓库阅读笔记：

- `docs/design/fa4_hopper_kv_stage.md`
- `docs/design/fa4_hopper_launch_shape_reference.md`

边界：

- 不引入 decode、append prefill、chunked prefill。
- 不引入 `seqlen_q != seqlen_k` 的通用尾部对齐 causal mask。
- 不引入 paged KV、SplitKV、partial O/LSE combine、非 causal attention。
- 不用 Blackwell 路径覆盖本项目 Hopper SM90 第一版设计。

## DeepGEMM MegaMoE

源码位置：

```text
third_party/DeepGEMM
```

适合参考：

- symmetric memory buffer 和 rank-relative pointer mapping。
- GPU-side rank-local grid barrier。
- NVLink / symmetric memory 阶段级跨 rank barrier。
- dispatch metadata 和 payload 分离。
- arrival count / arrival mask readiness。
- 通信、GEMM、epilogue、combine 在同一 kernel 中并行推进的调度组织。
- workspace 定向清理，避免全量 memset。

本仓库阅读笔记：

- `docs/design/deepgemm_megamoe_reference.md`

边界：

- MegaMoE 是 SM100 MoE 通算融合实现，不是本项目 Hopper SM90 attention kernel 的直接模板。
- 它的 UMMA/TMEM/FP8 x FP4 路径不能直接迁入第一版 SM90 fused FA + O_proj + AR。
- 它的阶段级 barrier 适合低频阶段边界；本项目 O_proj/AR 的 per-tile 或 per-batch ready 协议仍以核心设计文档为准。
- 如需借鉴 symmetric memory 或 barrier 语义，必须重新对照本项目 NVLS ready/count/owner 协议和 Hopper SM90 可用指令能力。
