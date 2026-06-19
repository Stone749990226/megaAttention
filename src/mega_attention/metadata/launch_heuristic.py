"""Host-side launch heuristic for the fused FA+O_proj+NVLS AR kernel.

设计依据: docs/design/launch_heuristic_role_sg_plan_zh.md (A 类).
r = FA_macs / OPROJ_macs 作分桶特征 (H_local 与 128^2*D 两边约掉):
    FA_macs    = 2 * Σ_t (m_block[t] + 1)        # ×2 = QK + PV
    OPROJ_macs = num_row_tiles * num_out_n_tiles
2 桶查表; 表值经 8×H200 sweep（Task 4）标定。单切点 _R_LO=2.0。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .row_desc import cdiv

# 单切点: r<2 平衡/O_proj 偏; r>=2 FA 偏 (Task4 8×H200 标定)
_R_LO = 2.0


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
