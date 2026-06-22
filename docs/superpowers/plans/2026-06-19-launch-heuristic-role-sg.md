# Launch 启发式（A 类）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 fused FA+O_proj+NVLS AR persistent kernel 加一层 host 侧 launch 启发式，按 shape 自适应选择 FA:O_proj:AR role 软配比与 super_group_n_tiles，提升 A 类（中等/O_proj 重/多序列）shape 的加速比。

**Architecture:** 纯 host 侧配置层 + 一处最小 kernel 改动。host 侧用 `r = 2·Σ(m_block+1)/(num_row_tiles·num_out_n_tiles)` 做分桶特征，查标定表得到 `(w_fa,w_oproj,w_ar,sg)`。kernel 把固定的 `cls=bidx%6` 偏好表换成权重驱动的 role 选择，保留 fall-through，`(4,1,1)` 逐位等价旧行为。**标定结论（见 Task 5）：单切点 2 桶（r<2→(2,1,1) / r≥2→(8,1,1)）、sg 恒为 4**——初始计划的「3 桶」与「sg 预编译 {2,4,8} 变体分派」经 8×H200 sweep 后未被采用。

**Tech Stack:** Python + CuTe DSL (cutlass), PyTorch, numpy, torch.distributed + symm_mem (NVLS)，8×H200。

## Global Constraints

- 第一版范围：Hopper SM90、causal、varlen full prompt prefill、`q_len == k_len`、FA+O_proj+NVLS AR fused persistent kernel。不引入 decode/append/chunked/paged/SplitKV/非 causal/其它架构。
- 不改 kernel 任何算法 invariant、队列协议、mbarrier、drain 规则；本计划只改 role 偏好选择与 host 配置。
- `(w_fa,w_oproj,w_ar) = (4,1,1)` 必须逐位等价旧 `bidx % 6` 行为（回归基线）。
- role 权重只影响调度/性能，不影响数值结果。
- 性能结论必须 8×H200 实测；标定表数值来自 sweep，不靠闭式公式。
- 设计依据：`docs/design/launch_heuristic_role_sg_plan_zh.md`、`docs/design/causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md`。
- 回答与落盘默认中文；代码标识符英文。

---

## File Structure

- `src/mega_attention/metadata/launch_heuristic.py`（新增）：`estimate_work_ratio`、`LaunchConfig`、`choose_launch_config` + 粗桶标定表。纯 host，CPU 可测。
- `tests/metadata/test_launch_heuristic.py`（新增）：上述函数的 CPU 单测。
- `src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py`（改）：`schedule_pick` 偏好选择改 role 驱动；`FusedFaOprojAr.__init__` 与 kernel 入口加权重；skeleton 同步。
- `tests/fused/test_role_weights.py`（新增）：GPU 单测，验证非默认权重不破坏 FA 数值与调度 invariant。
- `tests/fused/test_fused_fa_path.py`（改）：`run_case` 加 `w_fa/w_oproj/w_ar` 透传参数。
- `benchmarks/bench_fused_fa_oproj_ar.py`（改）：抽出 `bench_one(...)` 可复用函数；CLI 加 `--w_fa/--w_oproj/--w_ar/--sg/--auto`。
- `benchmarks/sweep_launch_config.py`（新增）：8×H200 sweep，遍历 12 shape × (配比,sg) 网格，输出标定数据。

---

## Task 1: launch_heuristic 模块（host，CPU TDD）

**Files:**
- Create: `src/mega_attention/metadata/launch_heuristic.py`
- Test: `tests/metadata/test_launch_heuristic.py`

**Interfaces:**
- Consumes: `RowDescMeta`（含 `.m_block: np.ndarray`、`.num_row_tiles: int`）与 `cdiv` from `mega_attention.metadata.row_desc`。
- Produces:
  - `estimate_work_ratio(meta, hidden: int, N_TILE: int = 128) -> float`
  - `@dataclass LaunchConfig(w_fa: int, w_oproj: int, w_ar: int, sg: int)`
  - `choose_launch_config(meta, hidden: int, tp_size: int, N_TILE: int = 128, num_sms: int = 132) -> LaunchConfig`

- [ ] **Step 1: 写失败测试**

```python
# tests/metadata/test_launch_heuristic.py
import numpy as np
from mega_attention.metadata.row_desc import build_row_desc, cdiv
from mega_attention.metadata.launch_heuristic import (
    estimate_work_ratio, choose_launch_config, LaunchConfig)


def _bruteforce_ratio(meta, hidden, N_TILE=128):
    fa = 2 * sum(int(meta.m_block[t]) + 1 for t in range(meta.num_row_tiles))
    oproj = meta.num_row_tiles * cdiv(hidden, N_TILE)
    return fa / oproj


def test_ratio_exact_single_seq():
    meta = build_row_desc([2048])              # 16 tiles, Σ(m+1)=136
    r = estimate_work_ratio(meta, hidden=2048)
    assert abs(r - (2 * 136) / (16 * 16)) < 1e-9   # 272/256 = 1.0625


def test_ratio_matches_bruteforce_varlen():
    meta = build_row_desc([300, 1000, 128, 4096])
    assert abs(estimate_work_ratio(meta, hidden=3072)
               - _bruteforce_ratio(meta, 3072)) < 1e-9


def test_ratio_grows_with_seqlen():
    short = estimate_work_ratio(build_row_desc([2048]), hidden=2048)
    long_ = estimate_work_ratio(build_row_desc([16384]), hidden=2048)
    assert long_ > short


def test_choose_tp1_war_zero_and_sg_valid():
    meta = build_row_desc([4096])
    cfg = choose_launch_config(meta, hidden=4096, tp_size=1)
    assert isinstance(cfg, LaunchConfig)
    assert cfg.w_ar == 0
    assert cfg.sg in (1, 2, 4, 8)
    assert 1 <= cfg.sg <= cdiv(4096, 128)
    assert cfg.w_fa >= 1 and cfg.w_oproj >= 1


def test_choose_fa_heavy_biases_fa_and_coarsens_sg():
    bal = choose_launch_config(build_row_desc([4096]), hidden=4096, tp_size=1)   # r≈1
    fa = choose_launch_config(build_row_desc([32768]), hidden=2048, tp_size=1)   # r≈16
    assert fa.w_fa / fa.w_oproj >= bal.w_fa / bal.w_oproj
    assert fa.sg >= bal.sg


def test_choose_tp_gt1_allows_ar_weight():
    meta = build_row_desc([4096])
    cfg = choose_launch_config(meta, hidden=4096, tp_size=8)
    assert cfg.w_ar >= 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/metadata/test_launch_heuristic.py -q`
Expected: FAIL（ModuleNotFoundError: launch_heuristic）

- [ ] **Step 3: 写最小实现**

```python
# src/mega_attention/metadata/launch_heuristic.py
"""Host-side launch heuristic for the fused FA+O_proj+NVLS AR kernel.

设计依据: docs/design/launch_heuristic_role_sg_plan_zh.md (A 类).
r = FA_macs / OPROJ_macs 作分桶特征 (H_local 与 128^2*D 两边约掉):
    FA_macs    = 2 * Σ_t (m_block[t] + 1)        # ×2 = QK + PV
    OPROJ_macs = num_row_tiles * num_out_n_tiles
粗 3 桶查表; 表值为 H200 sweep 前的初始猜测, 由 sweep_launch_config.py 标定后覆盖.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .row_desc import cdiv

# 粗桶阈值 (按 r 分 3 档). 初始值, 待 sweep 标定.
_R_LO = 2.0
_R_HI = 6.0


def estimate_work_ratio(meta, hidden: int, N_TILE: int = 128) -> float:
    """FA/O_proj MAC 比. 单序列退化为 ~ L/hidden. 仅作分桶特征."""
    fa_macs = 2 * int((meta.m_block.astype(np.int64) + 1).sum())
    oproj_macs = meta.num_row_tiles * cdiv(hidden, N_TILE)
    return fa_macs / oproj_macs


@dataclass
class LaunchConfig:
    w_fa: int
    w_oproj: int
    w_ar: int
    sg: int


def choose_launch_config(meta, hidden: int, tp_size: int,
                         N_TILE: int = 128, num_sms: int = 132) -> LaunchConfig:
    """按 r 粗桶查表返回 (w_fa,w_oproj,w_ar,sg). tp==1 时 w_ar=0."""
    r = estimate_work_ratio(meta, hidden, N_TILE)
    num_out = cdiv(hidden, N_TILE)
    if tp_size == 1:
        # (w_fa, w_oproj, sg) — pre-calibration guesses.
        if r < _R_LO:
            wf, wo, sg = 1, 1, 2
        elif r < _R_HI:
            wf, wo, sg = 2, 1, 4
        else:
            wf, wo, sg = 4, 1, 8
        wa = 0
    else:
        # (w_fa, w_oproj, w_ar, sg) — pre-calibration guesses.
        if r < _R_LO:
            wf, wo, wa, sg = 2, 2, 1, 4
        elif r < _R_HI:
            wf, wo, wa, sg = 3, 1, 1, 4
        else:
            wf, wo, wa, sg = 5, 1, 1, 8
    sg = max(1, min(sg, num_out))
    return LaunchConfig(w_fa=wf, w_oproj=wo, w_ar=wa, sg=sg)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/metadata/test_launch_heuristic.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add src/mega_attention/metadata/launch_heuristic.py tests/metadata/test_launch_heuristic.py
git commit -m "feat(metadata): launch 启发式 r 估算 + 粗桶 choose_launch_config（初始表，待标定）"
```

---

## Task 2: kernel role 权重驱动（GPU TDD）

**Files:**
- Modify: `src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py`（`schedule_pick` 约 338-387；real kernel 入口 668、call 785；skeleton 入口 431、call 453；两个 `__init__` 395、492）
- Modify: `tests/fused/test_fused_fa_path.py`（`run_case` 约 36、115）
- Test: `tests/fused/test_role_weights.py`（新增）

**Interfaces:**
- Consumes: `FusedFaOprojAr.run_case` 形参新增 `w_fa,w_oproj,w_ar`（来自 Task 1 语义，但此处是 int 直传）。
- Produces: `FusedFaOprojAr.__init__(..., w_fa=4, w_oproj=1, w_ar=1)`；`run_case(..., w_fa=4, w_oproj=1, w_ar=1)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/fused/test_role_weights.py
"""role 权重只影响调度/性能, 不影响 FA 数值与调度 invariant (tp=1, 单序列)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_fused_fa_path import run_case, _check   # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs H200")


@pytest.mark.parametrize("w", [(4, 1, 1), (8, 2, 1), (6, 1, 0)])
def test_role_weights_preserve_correctness(w):
    r = run_case([512], H_local=4, hidden=512, num_ctas=8,
                 w_fa=w[0], w_oproj=w[1], w_ar=w[2])
    assert _check(f"w={w}", r), f"invariants violated for weights {w}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/fused/test_role_weights.py -q`
Expected: FAIL（`run_case() got an unexpected keyword argument 'w_fa'`）

- [ ] **Step 3a: 改 `schedule_pick` 为 role 驱动**

把 [fused_fa_oproj_ar.py:338-362](src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py) 形参 `cls` 改名 `role`，并替换偏好分支：

```python
def schedule_pick(ctrl, oproj_queue, ar_ready_bits,
                  role, num_fa: cutlass.Constexpr, total_oproj: cutlass.Constexpr,
                  owner_words_alloc: cutlass.Constexpr, tp_size: cutlass.Constexpr,
                  rank: cutlass.Constexpr, local_owned_ar: cutlass.Constexpr):
    """Leader-only: return (mode, arg). role 0=prefer FA, 1=OPROJ, 2=AR."""
    # ... (前面 mode/arg/atomic 读 fa_d/op_d/ar_d/all_done 保持不变) ...
    else:
        # preference order by role (0=FA,1=OPROJ,2=AR); all fall through.
        s0 = cutlass.Int32(MODE_FA); s1 = cutlass.Int32(MODE_OPROJ); s2 = cutlass.Int32(MODE_AR)
        if role == 1:
            s0 = cutlass.Int32(MODE_OPROJ); s1 = cutlass.Int32(MODE_AR); s2 = cutlass.Int32(MODE_FA)
        if role == 2:
            s0 = cutlass.Int32(MODE_AR); s1 = cutlass.Int32(MODE_FA); s2 = cutlass.Int32(MODE_OPROJ)
        # ... (后面 try s0/s1/s2 循环保持不变) ...
```

- [ ] **Step 3b: real kernel 算 role（权重驱动）**

把 [fused_fa_oproj_ar.py:668](src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py) `cls = bidx % 6` 替换为：

```python
        w_fa = cutlass.const_expr(self.w_fa)
        w_fo = cutlass.const_expr(self.w_fa + self.w_oproj)
        w_m = cutlass.const_expr(self.w_fa + self.w_oproj + self.w_ar)
        k = bidx % cutlass.Int32(w_m)
        role = cutlass.Int32(0)
        if k >= cutlass.Int32(w_fa):
            role = cutlass.Int32(1)
        if k >= cutlass.Int32(w_fo):
            role = cutlass.Int32(2)
```

并把 [line 785](src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py) 的 `schedule_pick(..., cls, num_fa, ...)` 改成传 `role`。

- [ ] **Step 3c: skeleton 同步（保持旧行为）**

把 [fused_fa_oproj_ar.py:431](src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py) `cls = bidx % 6` 替换为等价 role 映射，并把 [line 453](src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py) 传 `role`：

```python
        k6 = bidx % cutlass.Int32(6)
        role = cutlass.Int32(0)
        if k6 >= cutlass.Int32(4):
            role = cutlass.Int32(1)
        if k6 >= cutlass.Int32(5):
            role = cutlass.Int32(2)
```

- [ ] **Step 3d: 两个 `__init__` 加权重字段**

`FusedFaOprojAr.__init__`（[line 492](src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py)）签名末尾加 `w_fa=4, w_oproj=1, w_ar=1`，并在体内存：

```python
        self.w_fa = w_fa
        self.w_oproj = w_oproj
        self.w_ar = w_ar if tp_size > 1 else 0
```

`FusedFaOprojArSkeleton.__init__` 无需加字段（skeleton 固定 bidx%6）。

- [ ] **Step 3e: `run_case` 透传权重**

`tests/fused/test_fused_fa_path.py` 的 `run_case` 签名加 `w_fa=4, w_oproj=1, w_ar=1`，构造处传入：

```python
    ker = FusedFaOprojAr(num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
                         num_super_groups=num_super_groups, total_oproj=total_oproj,
                         num_ctas=num_ctas, hidden=hidden, tp_size=1, N_TILE=N_TILE,
                         super_group_n_tiles=super_group_n_tiles,
                         w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar)
```

- [ ] **Step 4: 跑回归 + 新测试**

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
pytest tests/fused/test_scheduler_skeleton.py -q          # skeleton 行为不变
python tests/fused/test_fused_fa_path.py                  # 单序列 PASS (默认权重)
pytest tests/fused/test_role_weights.py -q                # 三组权重 PASS
```
Expected: skeleton 测试 PASS；fused 单序列 PASS；role_weights 3 passed。

- [ ] **Step 5: 提交**

```bash
git add src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py tests/fused/test_fused_fa_path.py tests/fused/test_role_weights.py
git commit -m "feat(fused): role 偏好改权重驱动（保留 fall-through，(4,1,1) 等价旧 bidx%6）"
```

---

## Task 3: bench 可复用 + CLI 旋钮

**Files:**
- Modify: `benchmarks/bench_fused_fa_oproj_ar.py`

**Interfaces:**
- Produces: `bench_one(seqlens, H_local, hidden, w_fa, w_oproj, w_ar, sg, ws, rank, dev, iters, warmup) -> dict`（返回 `{"fused_ms": float, "base_ms": float, "ratio": float}`）。
- Consumes: Task 1 的 `choose_launch_config`（仅 `--auto` 路径）。

- [ ] **Step 1: 抽出 `bench_one`**

把 `main()` 中"构造缓冲 → compile → 计时 fused 与基线 → 算 ratio"的逻辑搬进模块级函数 `bench_one(...)`，其中：
- `_, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, sg)` 用入参 `sg`；
- `FusedFaOprojAr(...)` 传 `super_group_n_tiles=sg, w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar`；
- 返回 `dict(fused_ms=..., base_ms=..., ratio=base_ms/fused_ms)`。

- [ ] **Step 2: CLI 加旋钮**

`main()` 的 argparse 增加：

```python
    ap.add_argument("--w_fa", type=int, default=4)
    ap.add_argument("--w_oproj", type=int, default=1)
    ap.add_argument("--w_ar", type=int, default=1)
    ap.add_argument("--sg", type=int, default=4)
    ap.add_argument("--auto", action="store_true",
                    help="用 choose_launch_config 自动选 (w_fa,w_oproj,w_ar,sg)")
```

`main()` 决定配置：

```python
    meta = build_row_desc(seqlens)
    if args.auto:
        from mega_attention.metadata.launch_heuristic import choose_launch_config
        cfg = choose_launch_config(meta, hidden, tp_size=ws)
        w_fa, w_oproj, w_ar, sg = cfg.w_fa, cfg.w_oproj, cfg.w_ar, cfg.sg
    else:
        w_fa, w_oproj, w_ar, sg = args.w_fa, args.w_oproj, args.w_ar, args.sg
    res = bench_one(seqlens, H_local, hidden, w_fa, w_oproj, w_ar, sg,
                    ws, rank, dev, args.iters, args.warmup)
    if rank == 0:
        print(f"[fused] w=({w_fa},{w_oproj},{w_ar}) sg={sg} "
              f"fused={res['fused_ms']:.3f}ms base={res['base_ms']:.3f}ms "
              f"ratio={res['ratio']:.2f}x", flush=True)
```

- [ ] **Step 3: 冒烟验证（默认权重等价旧路径）**

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py \
  --seqlens 2048,2048 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py \
  --seqlens 2048,2048 --hidden 2048 --h_local 8 --auto --iters 30 --warmup 10
```
Expected: 两条都打印 ratio；默认那条 ratio 与历史 ~1.2× 一致（未回归）。

- [ ] **Step 4: 提交**

```bash
git add benchmarks/bench_fused_fa_oproj_ar.py
git commit -m "bench(fused): 抽出 bench_one + CLI --w_fa/--w_oproj/--w_ar/--sg/--auto"
```

---

## Task 4: 8×H200 sweep 标定

**Files:**
- Create: `benchmarks/sweep_launch_config.py`

**Interfaces:**
- Consumes: Task 3 的 `bench_one`。
- Produces: stdout 一张 markdown 表 + `benchmarks/sweep_results.md`（每 shape 各配置的 ratio 与最优配置）。

- [ ] **Step 1: 写 sweep 脚本**

```python
# benchmarks/sweep_launch_config.py
"""8×H200 sweep: 12 shape × (配比, sg) 网格, 找每组最优 -> 标定粗桶表."""
import os

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from mega_attention.metadata.row_desc import build_row_desc
from mega_attention.metadata.launch_heuristic import estimate_work_ratio
from benchmarks.bench_fused_fa_oproj_ar import bench_one

SHAPES = [
    ("2048,2048", 8, 2048), ("1024,1024,1024,1024", 16, 2048),
    ("4096,4096", 8, 4096), ("8192", 8, 2048),
    ("2048,2048,2048,2048,2048,2048,2048,2048", 8, 2048),
    ("8192,8192", 8, 2048), ("8192,8192", 8, 4096), ("8192,8192", 16, 7168),
    ("16384", 8, 2048), ("16384", 16, 4096),
    ("16384,16384", 8, 2048), ("32768", 8, 2048),
]
GRID = [  # (w_fa, w_oproj, w_ar, sg); w_ar 在 tp>1 生效
    (4, 1, 1, 2), (4, 1, 1, 4), (4, 1, 1, 8),
    (2, 1, 1, 4), (8, 1, 1, 4), (8, 1, 1, 8), (5, 1, 1, 8),
]


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    enable_symm_mem_for_group(dist.group.WORLD.group_name)

    lines = ["| shape | r | best (w_fa,w_oproj,w_ar,sg) | best ratio |",
             "| --- | --- | --- | --- |"]
    for seqstr, h_local, hidden in SHAPES:
        seqlens = [int(x) for x in seqstr.split(",")]
        meta = build_row_desc(seqlens)
        r = estimate_work_ratio(meta, hidden)
        best = None
        for (wf, wo, wa, sg) in GRID:
            res = bench_one(seqlens, h_local, hidden, wf, wo, wa, sg,
                            ws, rank, dev, iters=30, warmup=10)
            if rank == 0:
                print(f"  {seqstr} h{h_local} hid{hidden} r={r:.2f} "
                      f"w=({wf},{wo},{wa}) sg={sg} ratio={res['ratio']:.3f}", flush=True)
            if best is None or res["ratio"] > best[1]:
                best = ((wf, wo, wa, sg), res["ratio"])
        if rank == 0:
            lines.append(f"| {seqstr} h{h_local} hid{hidden} | {r:.2f} | "
                         f"{best[0]} | {best[1]:.3f}x |")
    if rank == 0:
        out = "\n".join(lines)
        print(out, flush=True)
        with open("benchmarks/sweep_results.md", "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 sweep**

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
torchrun --nproc_per_node=8 benchmarks/sweep_launch_config.py 2>&1 | tee benchmarks/sweep_log.txt
```
Expected: 生成 `benchmarks/sweep_results.md`，12 行各带 r、最优配置、最优 ratio。

> **若 sweep 显示某 shape 的所有配置 ratio 几乎不变**（旋钮无收益）：说明瓶颈是 mode 切换抖动而非配比，停下来在 `sweep_results.md` 记录该现象，并向用户报告——是否转机制 2（硬保留 FA CTA）。不要擅自扩范围。

- [ ] **Step 3: 提交结果**

```bash
git add benchmarks/sweep_launch_config.py benchmarks/sweep_results.md
git commit -m "bench(fused): 8×H200 launch-config sweep 脚本 + 标定结果"
```

---

## Task 5: 填标定表（2 桶）+ 12 组回归

**已据 Task 4 sweep + 用户确认定案：2 桶数据驱动，tp=1 不作目标（该算子为 TP>1 设计）。**
两个桶的稳健最优 sg 都是 4（sg2/sg8 的零星胜出非桶内稳健解），故 sg 恒为 4。

标定表（来自 `benchmarks/sweep_results.md` 桶内平均 ratio 最优）：

```text
tp > 1 (主目标):
    r < 2.0   -> (w_fa, w_oproj, w_ar, sg) = (2, 1, 1, 4)
    r >= 2.0  ->                            (8, 1, 1, 4)
tp == 1 (非目标, 保持函数可用; w_ar=0 镜像):
    r < 2.0   -> (2, 1, 0, 4)
    r >= 2.0  -> (8, 1, 0, 4)
```

依据（桶内平均 ratio，tp=8）：r<2 桶 (2,1,1,4)=1.148 优于 (4,1,1,4)=1.128；
r≥2 桶 (8,1,1,4)=1.21(中)/1.04(高) 优于 (4,1,1,4)=1.19/—。

**Files:**
- Modify: `src/mega_attention/metadata/launch_heuristic.py`（改为单切点 `_R_LO=2.0`，两桶表）
- Modify: `tests/metadata/test_launch_heuristic.py`（按标定值收紧断言）
- Modify: `README.md`（更新 benchmark 表，加 auto 列）

**Interfaces:** 无新接口；只更新常量与文档。

- [ ] **Step 1: 改 `choose_launch_config` 为 2 桶标定表**

把 Task 1 的 `_R_HI` 删除、`_R_LO = 2.0`，并把两个 `if/elif/else` 三桶替换为单切点两桶（用上面的标定值）：

```python
_R_LO = 2.0   # 单切点: r<2 平衡/O_proj 偏; r>=2 FA 偏 (Task4 8×H200 标定)


def choose_launch_config(meta, hidden: int, tp_size: int,
                         N_TILE: int = 128, num_sms: int = 132) -> LaunchConfig:
    """按 r 2 桶查表返回 (w_fa,w_oproj,w_ar,sg). tp==1 时 w_ar=0 (非目标场景)."""
    r = estimate_work_ratio(meta, hidden, N_TILE)
    num_out = cdiv(hidden, N_TILE)
    wa = 1 if tp_size > 1 else 0
    if r < _R_LO:
        wf, wo, sg = 2, 1, 4
    else:
        wf, wo, sg = 8, 1, 4
    sg = max(1, min(sg, num_out))
    return LaunchConfig(w_fa=wf, w_oproj=wo, w_ar=wa, sg=sg)
```

- [ ] **Step 2: 收紧单测**

在 `tests/metadata/test_launch_heuristic.py` 把 `test_choose_fa_heavy_biases_fa_and_coarsens_sg` 之外，新增针对标定桶的显式断言（替换/补充）：

```python
def test_choose_calibrated_buckets_tp8():
    # r<2 桶 (平衡): 4096,4096 hid4096 -> (2,1,1,4)
    bal = choose_launch_config(build_row_desc([4096, 4096]), hidden=4096, tp_size=8)
    assert (bal.w_fa, bal.w_oproj, bal.w_ar, bal.sg) == (2, 1, 1, 4)
    # r>=2 桶 (FA 偏): 16384 hid2048 (r≈8) -> (8,1,1,4)
    fa = choose_launch_config(build_row_desc([16384]), hidden=2048, tp_size=8)
    assert (fa.w_fa, fa.w_oproj, fa.w_ar, fa.sg) == (8, 1, 1, 4)


def test_choose_tp1_mirrors_with_war_zero():
    cfg = choose_launch_config(build_row_desc([16384]), hidden=2048, tp_size=1)
    assert (cfg.w_fa, cfg.w_oproj, cfg.w_ar, cfg.sg) == (8, 1, 0, 4)
```

然后运行：

Run: `pytest tests/metadata/test_launch_heuristic.py -q`
Expected: PASS（原有 + 2 个新测试全过；`test_choose_fa_heavy_biases_fa_and_coarsens_sg` 仍成立：8/1≥2/1 且 sg 4≥4）

- [ ] **Step 3: 12 组 --auto 回归**

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 2048,2048 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 1024,1024,1024,1024 --hidden 2048 --h_local 16 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 4096,4096 --hidden 4096 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 8192 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 2048,2048,2048,2048,2048,2048,2048,2048 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 8192,8192 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 8192,8192 --hidden 4096 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 8192,8192 --hidden 7168 --h_local 16 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 16384 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 16384 --hidden 4096 --h_local 16 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 16384,16384 --hidden 2048 --h_local 8 --iters 30 --warmup 10
torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --auto --seqlens 32768 --hidden 2048 --h_local 8 --iters 30 --warmup 10
```
Expected（对照 `sweep_results.md` 对应桶配置列）：A 类提升（如 4096,4096→~1.40×、8192,8192 hid2048→~1.26×）；B 类长单序列不回退（32768→~0.92× 同旧）。把 12 个 `[fused] ... ratio=...` 收集成表。

- [ ] **Step 4: 更新 README 并提交**

在 [README.md](README.md) benchmark 表加一列 "auto ratio"（`--auto` 实测），并在表下加一句说明启发式为 2 桶（r<2→(2,1,1) / r≥2→(8,1,1)，sg=4），以及 DeepSeek/32K 两类 launch 启发式不改善的说明。

```bash
git add src/mega_attention/metadata/launch_heuristic.py tests/metadata/test_launch_heuristic.py README.md
git commit -m "feat(metadata): 2 桶标定表（r<2→(2,1,1) / r≥2→(8,1,1), sg=4）+ 12组 --auto 回归"
```

---

## Self-Review

**Spec coverage:**
- §3.1 `r` 估算 → Task 1。 §3.2 role 软配比（kernel）→ Task 2。 §3.3 sg 变体 → Task 3（CLI/bench_one 传 sg）+ Task 1（choose 选 sg）。 §3.4 粗桶查表 → Task 1（结构）+ Task 5（标定值）。 §3.5 模块接口 → Task 1/2/3。 §4 验证标定 → Task 4/5。 §5 不做项 → Global Constraints。全部覆盖。
- 风险出口（旋钮无收益→机制 2）→ Task 4 Step 2 显式记录并停下确认。

**Placeholder scan:** 标定数值是设计上的"待 sweep 填入"，Task 1 给了可运行初始值、Task 5 用真实数据覆盖，非占位失败。其余步骤均有完整代码/命令。

**Type consistency:** `choose_launch_config(meta, hidden, tp_size, ...)`、`LaunchConfig(w_fa,w_oproj,w_ar,sg)`、`bench_one(...)->dict(fused_ms,base_ms,ratio)`、`schedule_pick(..., role, ...)`、`FusedFaOprojAr(..., w_fa,w_oproj,w_ar)`、`run_case(..., w_fa,w_oproj,w_ar)` 在各 Task 间一致。
