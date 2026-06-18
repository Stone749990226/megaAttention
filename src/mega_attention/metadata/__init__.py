"""varlen row tile 调度所需的 host 侧元数据工具。"""

from .row_desc import (
    AR_M_TILE,
    FA_M_TILE,
    OPROJ_M_TILE,
    ROW_M_TILE,
    RowDescMeta,
    build_row_desc,
    cdiv,
    csym_numel,
    cu_seqlens_from_seqlens,
    decode_oproj_slot,
    oproj_task_counts,
    oscratch_numel,
)

__all__ = [
    "AR_M_TILE",
    "FA_M_TILE",
    "OPROJ_M_TILE",
    "ROW_M_TILE",
    "RowDescMeta",
    "build_row_desc",
    "cdiv",
    "csym_numel",
    "cu_seqlens_from_seqlens",
    "decode_oproj_slot",
    "oproj_task_counts",
    "oscratch_numel",
]
