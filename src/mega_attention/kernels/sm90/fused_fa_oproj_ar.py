#!/usr/bin/env python3
"""
Fused FA + O_proj + NVLS AllReduce persistent kernel (Hopper SM90, CuTe DSL).

PHASE 1 -- scheduler skeleton with STUB payloads (single rank, no real math).

This file grows across the plan's phases. Right now it implements only the
multi-mode persistent *scheduler* and the cross-mode handoff protocol, with
trivial known-answer stub "compute" so the plumbing can be validated without any
floating-point math (see test_scheduler_skeleton.py). Real FA / O_proj GEMM /
NVLS reduce drop into the do_fa / do_oproj / do_ar stubs in later phases.

Protocol implemented (设计文稿.md):
  * persistent grid of `num_ctas` CTAs; device-wide grid_sync at init + exit.
  * FA tasks claimed from a global atomic counter -> (row_tile, head).
  * head_ready_count[row_tile] release/acquire; the last head of a row publishes
    that row's O_proj tasks into an ordered ready queue (reserve/publish_tail).
  * O_proj tasks popped via CAS on consume_head.
  * each O_proj task pushes a per-(row_tile,super_group) AR readiness (single
    rank => last-arriver is immediate), claimed once by an AR owner task.
  * cta_id % 6 static preference table; all CTAs fall through to other sources.
  * termination when fa_done / oproj_done / ar_done all hit their targets.

STUB semantics (axis A, known-answer):
  do_fa    : writes head_marker[row_tile,head] = row_tile*H_local+head+1 (nonzero)
  do_oproj : reads ALL H_local markers of its row, asserts each present+correct
             (any miss => order_err++), writes a partial checksum.
  do_ar    : terminal-protected once-only completion.
"""
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from quack import layout_utils as qlu

# Phase-5 (5b-1b): real FA payload reuses the validated dynamic-varlen FA pieces.
from mega_attention.kernels.sm90.fa_varlen import softmax_block_dyn, cdiv, LOG2E

# ---- mode codes (broadcast through smem) ----
MODE_IDLE = 0
MODE_FA = 1
MODE_OPROJ = 2
MODE_AR = 3
MODE_DONE = 4

# ---- ctrl[] scalar slots (all Uint32) ----
C_FA_COUNTER = 0
C_FA_DONE = 1
C_OP_RESERVE = 2
C_OP_PUBLISH = 3
C_OP_CONSUME = 4
C_OP_DONE = 5
C_AR_DONE = 6
C_AR_CURSOR = 7
C_ORDER_ERR = 8
C_GS_INIT = 9
C_GS_EXIT = 10
NUM_CTRL = 11

FINISH_TAG = 0x80000000


# ============================================================ grid_sync ====
@cute.jit
def gsync(ctrl: cute.Tensor, idx: cutlass.Constexpr,
          num_ctas: cutlass.Constexpr, bidx, tidx):
    """Mega-MoE device-wide barrier on ctrl[idx] (Uint32, zeroed by host)."""
    cute.arch.sync_threads()
    if tidx == 0:
        delta = cutlass.Uint32(1)
        if bidx == 0:
            delta = cutlass.Uint32(FINISH_TAG - (num_ctas - 1))
        old = cute.arch.atomic_add(ctrl.iterator + idx, delta,
                                   sem="release", scope="gpu")
        spinning = True
        while spinning:
            cur = cute.arch.atomic_add(ctrl.iterator + idx, cutlass.Uint32(0),
                                       sem="acquire", scope="gpu")
            if ((cur ^ old) & cutlass.Uint32(FINISH_TAG)) != 0:
                spinning = False
    cute.arch.sync_threads()


# =============================================== work-source claim helpers ==
@cute.jit
def try_fa(ctrl: cute.Tensor, num_fa: cutlass.Constexpr):
    """Bounded atomic claim of the next FA task. Returns (found, fa_task_id)."""
    found = cutlass.Int32(0)
    arg = cutlass.Int32(-1)
    cnt = cute.arch.atomic_add(ctrl.iterator + C_FA_COUNTER, cutlass.Uint32(0),
                               sem="relaxed", scope="gpu")
    if cnt < num_fa:                       # pre-check avoids unbounded over-claim
        tid = cute.arch.atomic_add(ctrl.iterator + C_FA_COUNTER, cutlass.Uint32(1),
                                   sem="relaxed", scope="gpu")
        if tid < num_fa:
            found = cutlass.Int32(1)
            arg = tid.to(cutlass.Int32)
    return found, arg


@cute.jit
def try_pop_oproj(ctrl: cute.Tensor, oproj_queue: cute.Tensor):
    """CAS pop from the O_proj ready queue. Returns (found, slot_id)."""
    found = cutlass.Int32(0)
    arg = cutlass.Int32(-1)
    head = cute.arch.atomic_add(ctrl.iterator + C_OP_CONSUME, cutlass.Uint32(0),
                                sem="relaxed", scope="gpu")
    tail = cute.arch.atomic_add(ctrl.iterator + C_OP_PUBLISH, cutlass.Uint32(0),
                                sem="acquire", scope="gpu")
    if head < tail:
        old = cute.arch.atomic_cas(ctrl.iterator + C_OP_CONSUME,
                                   cmp=head, val=head + cutlass.Uint32(1),
                                   sem="relaxed", scope="gpu")
        if old == head:
            slot = oproj_queue[head]       # published => entry visible
            found = cutlass.Int32(1)
            arg = slot.to(cutlass.Int32)
    return found, arg


@cute.jit
def try_claim_ar(ctrl: cute.Tensor, ar_probe: cute.Tensor,
                 total_oproj: cutlass.Constexpr):
    """Scan (from a cursor hint) for a ready AR tile and CAS-claim it (1->2).

    Returns (found, ar_slot_id). Single-owner: only the CAS winner proceeds.
    O(total) worst case -- fine for the skeleton's modest task space.
    """
    found = cutlass.Int32(0)
    arg = cutlass.Int32(-1)
    start = cute.arch.atomic_add(ctrl.iterator + C_AR_CURSOR, cutlass.Uint32(0),
                                 sem="relaxed", scope="gpu")
    i = cutlass.Uint32(0)
    n = cutlass.Uint32(total_oproj)
    while (i < n) and (found == 0):
        idx = (start + i) % n
        p = cute.arch.atomic_add(ar_probe.iterator + idx, cutlass.Uint32(0),
                                 sem="acquire", scope="gpu")
        if p == 1:
            old = cute.arch.atomic_cas(ar_probe.iterator + idx,
                                       cmp=cutlass.Uint32(1), val=cutlass.Uint32(2),
                                       sem="acquire", scope="gpu")
            if old == 1:
                found = cutlass.Int32(1)
                arg = idx.to(cutlass.Int32)
        i = i + cutlass.Uint32(1)
    if found != 0:
        # advance cursor hint past the claimed slot (relaxed; best-effort)
        cute.arch.atomic_exch(ctrl.iterator + C_AR_CURSOR,
                              arg.to(cutlass.Uint32) + cutlass.Uint32(1),
                              sem="relaxed", scope="gpu")
    return found, arg


# ============================================== producer: publish O_proj ===
@cute.jit
def publish_oproj(ctrl: cute.Tensor, oproj_queue: cute.Tensor,
                  row_tile: cutlass.Int32, num_super_groups: cutlass.Constexpr):
    """Reserve / write / ordered-publish this row's num_super_groups O_proj tasks."""
    n = cutlass.Uint32(num_super_groups)
    start = cute.arch.atomic_add(ctrl.iterator + C_OP_RESERVE, n,
                                 sem="relaxed", scope="gpu")
    base = row_tile.to(cutlass.Uint32) * n
    i = cutlass.Uint32(0)
    while i < n:
        oproj_queue[start + i] = base + i          # slot_id = row*nsg + i
        i = i + cutlass.Uint32(1)
    cute.arch.fence_acq_rel_gpu()                   # entries before publish
    # ordered publish: wait until publish_tail == our reservation start
    spinning = True
    while spinning:
        pt = cute.arch.atomic_add(ctrl.iterator + C_OP_PUBLISH, cutlass.Uint32(0),
                                  sem="acquire", scope="gpu")
        if pt == start:
            spinning = False
    cute.arch.atomic_add(ctrl.iterator + C_OP_PUBLISH, n,
                         sem="release", scope="gpu")


# =================================================== stub payloads (axis A) =
@cute.jit
def do_fa(ctrl: cute.Tensor, head_marker: cute.Tensor, fa_exec: cute.Tensor,
          head_ready: cute.Tensor, oproj_queue: cute.Tensor,
          fa_task_id: cutlass.Int32,
          H_local: cutlass.Constexpr, num_super_groups: cutlass.Constexpr):
    """STUB FA: write a known marker, bump head_ready, last head publishes O_proj."""
    row_tile = fa_task_id // H_local
    head = fa_task_id % H_local
    marker = row_tile * cutlass.Int32(H_local) + head + cutlass.Int32(1)
    head_marker[fa_task_id] = marker.to(cutlass.Uint32)     # "O_scratch store"
    cute.arch.atomic_add(fa_exec.iterator + fa_task_id.to(cutlass.Uint32),
                         cutlass.Uint32(1), sem="relaxed", scope="gpu")
    cute.arch.fence_acq_rel_gpu()                            # release marker
    old = cute.arch.atomic_add(head_ready.iterator + row_tile.to(cutlass.Uint32),
                               cutlass.Uint32(1), sem="acq_rel", scope="gpu")
    if (old + cutlass.Uint32(1)) == cutlass.Uint32(H_local):
        publish_oproj(ctrl, oproj_queue, row_tile, num_super_groups)
    cute.arch.atomic_add(ctrl.iterator + C_FA_DONE, cutlass.Uint32(1),
                         sem="release", scope="gpu")


@cute.jit
def do_oproj(ctrl: cute.Tensor, head_marker: cute.Tensor, oproj_exec: cute.Tensor,
             ready_count_owner: cute.Tensor, ar_probe: cute.Tensor,
             partial_check: cute.Tensor, slot_id: cutlass.Int32,
             H_local: cutlass.Constexpr, num_super_groups: cutlass.Constexpr,
             tp_size: cutlass.Constexpr):
    """STUB O_proj: verify all heads of the row are present, push AR readiness."""
    row_tile = slot_id // num_super_groups
    nsg = slot_id % num_super_groups
    # ordering check: every head marker of this row must be present + correct
    h = cutlass.Int32(0)
    bad = cutlass.Uint32(0)
    while h < cutlass.Int32(H_local):
        idx = (row_tile * cutlass.Int32(H_local) + h).to(cutlass.Uint32)
        mk = head_marker[idx]
        expected = (row_tile * cutlass.Int32(H_local) + h + cutlass.Int32(1)).to(cutlass.Uint32)
        if mk != expected:
            bad = bad + cutlass.Uint32(1)
        h = h + cutlass.Int32(1)
    if bad != 0:
        cute.arch.atomic_add(ctrl.iterator + C_ORDER_ERR, bad,
                             sem="relaxed", scope="gpu")
    cute.arch.atomic_add(oproj_exec.iterator + slot_id.to(cutlass.Uint32),
                         cutlass.Uint32(1), sem="relaxed", scope="gpu")
    # stub "partial" checksum
    partial_check[slot_id.to(cutlass.Uint32)] = (
        row_tile * cutlass.Int32(1000) + nsg + cutlass.Int32(1)).to(cutlass.Uint32)
    # push-to-owner ready_count (single rank: tp_size==1 => immediate last-arriver)
    cute.arch.fence_acq_rel_gpu()                           # (sys-scope in Phase 4)
    old = cute.arch.atomic_add(
        ready_count_owner.iterator + slot_id.to(cutlass.Uint32),
        cutlass.Uint32(1), sem="acq_rel", scope="gpu")
    if (old + cutlass.Uint32(1)) == cutlass.Uint32(tp_size):
        cute.arch.atomic_exch(ar_probe.iterator + slot_id.to(cutlass.Uint32),
                              cutlass.Uint32(1), sem="release", scope="gpu")
    cute.arch.atomic_add(ctrl.iterator + C_OP_DONE, cutlass.Uint32(1),
                         sem="release", scope="gpu")


@cute.jit
def do_ar(ctrl: cute.Tensor, ar_done_flag: cute.Tensor, ar_exec: cute.Tensor,
          ar_slot_id: cutlass.Int32):
    """STUB AR owner: terminal-protected once-only completion (identity reduce)."""
    cute.arch.atomic_add(ar_exec.iterator + ar_slot_id.to(cutlass.Uint32),
                         cutlass.Uint32(1), sem="relaxed", scope="gpu")
    old_done = cute.arch.atomic_cas(ar_done_flag.iterator + ar_slot_id.to(cutlass.Uint32),
                                    cmp=cutlass.Uint32(0), val=cutlass.Uint32(1),
                                    sem="acq_rel", scope="gpu")
    if old_done == 0:
        # Phase 4: multimem.ld_reduce + multimem.st here. Skeleton: identity.
        cute.arch.atomic_add(ctrl.iterator + C_AR_DONE, cutlass.Uint32(1),
                             sem="release", scope="gpu")


# ===================================================== schedule (leader) ====
@cute.jit
def schedule_pick(ctrl, oproj_queue, ar_probe,
                  cls, num_fa: cutlass.Constexpr, total_oproj: cutlass.Constexpr):
    """Leader-only: return (mode, arg). DONE iff all three targets met."""
    mode = cutlass.Int32(MODE_IDLE)
    arg = cutlass.Int32(-1)

    fa_d = cute.arch.atomic_add(ctrl.iterator + C_FA_DONE, cutlass.Uint32(0),
                                sem="acquire", scope="gpu")
    op_d = cute.arch.atomic_add(ctrl.iterator + C_OP_DONE, cutlass.Uint32(0),
                                sem="acquire", scope="gpu")
    ar_d = cute.arch.atomic_add(ctrl.iterator + C_AR_DONE, cutlass.Uint32(0),
                                sem="acquire", scope="gpu")
    all_done = (fa_d >= num_fa) and (op_d >= total_oproj) and (ar_d >= total_oproj)

    if all_done:
        mode = cutlass.Int32(MODE_DONE)
    else:
        # preference order s0,s1,s2 by cls = cta_id % 6 (codes 1=FA,2=OPROJ,3=AR)
        s0 = cutlass.Int32(MODE_FA); s1 = cutlass.Int32(MODE_OPROJ); s2 = cutlass.Int32(MODE_AR)
        if cls == 4:
            s0 = cutlass.Int32(MODE_OPROJ); s1 = cutlass.Int32(MODE_AR); s2 = cutlass.Int32(MODE_FA)
        if cls == 5:
            s0 = cutlass.Int32(MODE_AR); s1 = cutlass.Int32(MODE_FA); s2 = cutlass.Int32(MODE_OPROJ)

        found = cutlass.Int32(0)
        # try s0, then s1, then s2
        ci = cutlass.Int32(0)
        while ci < cutlass.Int32(3):
            src = s0
            if ci == 1:
                src = s1
            if ci == 2:
                src = s2
            if found == 0:
                f = cutlass.Int32(0); a = cutlass.Int32(-1)
                if src == MODE_FA:
                    f, a = try_fa(ctrl, num_fa)
                if src == MODE_OPROJ:
                    f, a = try_pop_oproj(ctrl, oproj_queue)
                if src == MODE_AR:
                    f, a = try_claim_ar(ctrl, ar_probe, total_oproj)
                if f != 0:
                    found = cutlass.Int32(1)
                    mode = src
                    arg = a
            ci = ci + cutlass.Int32(1)
    return mode, arg


# ============================================================= driver/kernel =
class FusedFaOprojArSkeleton:
    """Phase 1 scheduler skeleton. threads_per_cta is 1 warp-group (128) for now;
    Phase 2 widens to 3 warp groups (384) and fills do_fa with real FA."""

    def __init__(self, num_fa, num_row_tiles, H_local, num_super_groups,
                 total_oproj, num_ctas, tp_size=1, threads_per_cta=128):
        self.num_fa = num_fa
        self.num_row_tiles = num_row_tiles
        self.H_local = H_local
        self.num_super_groups = num_super_groups
        self.total_oproj = total_oproj
        self.num_ctas = num_ctas
        self.tp_size = tp_size
        self.threads_per_cta = threads_per_cta

    @cute.jit
    def __call__(self, ctrl, head_ready, oproj_queue, ready_count_owner,
                 ar_probe, ar_done_flag, head_marker, fa_exec, oproj_exec,
                 ar_exec, partial_check, stream: cuda.CUstream):
        self.kernel(ctrl, head_ready, oproj_queue, ready_count_owner, ar_probe,
                    ar_done_flag, head_marker, fa_exec, oproj_exec, ar_exec,
                    partial_check).launch(
            grid=[self.num_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ctrl: cute.Tensor, head_ready: cute.Tensor,
               oproj_queue: cute.Tensor, ready_count_owner: cute.Tensor,
               ar_probe: cute.Tensor, ar_done_flag: cute.Tensor,
               head_marker: cute.Tensor, fa_exec: cute.Tensor,
               oproj_exec: cute.Tensor, ar_exec: cute.Tensor,
               partial_check: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cls = bidx % 6

        num_fa = cutlass.const_expr(self.num_fa)
        total_oproj = cutlass.const_expr(self.total_oproj)
        H_local = cutlass.const_expr(self.H_local)
        num_super_groups = cutlass.const_expr(self.num_super_groups)
        tp_size = cutlass.const_expr(self.tp_size)
        num_ctas = cutlass.const_expr(self.num_ctas)

        # smem broadcast of (mode, arg) from leader to the whole CTA
        smem = cutlass.utils.SmemAllocator()
        sma = smem.allocate_tensor(cutlass.Int32, cute.make_layout(2),
                                   byte_alignment=4)

        gsync(ctrl, C_GS_INIT, num_ctas, bidx, tidx)

        looping = True
        while looping:
            if tidx == 0:
                mode, arg = schedule_pick(ctrl, oproj_queue, ar_probe, cls,
                                          num_fa, total_oproj)
                sma[0] = mode
                sma[1] = arg
            cute.arch.sync_threads()
            mode = sma[0]
            arg = sma[1]

            if mode == MODE_DONE:
                looping = False
            else:
                if tidx == 0:
                    if mode == MODE_FA:
                        do_fa(ctrl, head_marker, fa_exec, head_ready, oproj_queue,
                              arg, H_local, num_super_groups)
                    if mode == MODE_OPROJ:
                        do_oproj(ctrl, head_marker, oproj_exec, ready_count_owner,
                                 ar_probe, partial_check, arg, H_local,
                                 num_super_groups, tp_size)
                    if mode == MODE_AR:
                        do_ar(ctrl, ar_done_flag, ar_exec, arg)
                cute.arch.sync_threads()      # mode-switch teardown barrier

        gsync(ctrl, C_GS_EXIT, num_ctas, bidx, tidx)


# ============================================ 5b-1b: real FA-path fused kernel =
@cute.jit
def do_oproj_stub_v2(ctrl: cute.Tensor, head_ready: cute.Tensor,
                     oproj_exec: cute.Tensor, ready_count_owner: cute.Tensor,
                     ar_probe: cute.Tensor, partial_check: cute.Tensor,
                     slot_id: cutlass.Int32, H_local: cutlass.Constexpr,
                     num_super_groups: cutlass.Constexpr, tp_size: cutlass.Constexpr):
    """5b-1b O_proj STUB: order-check via head_ready (real FA writes O_scratch, no
    markers). When this runs, head_ready[row_tile] must equal H_local (publish only
    happens at H_local) — validates the FA->O_proj happens-before with real FA."""
    row_tile = slot_id // num_super_groups
    nsg = slot_id % num_super_groups
    hr = cute.arch.atomic_add(head_ready.iterator + row_tile.to(cutlass.Uint32),
                              cutlass.Uint32(0), sem="acquire", scope="gpu")
    if hr != cutlass.Uint32(H_local):
        cute.arch.atomic_add(ctrl.iterator + C_ORDER_ERR, cutlass.Uint32(1),
                             sem="relaxed", scope="gpu")
    cute.arch.atomic_add(oproj_exec.iterator + slot_id.to(cutlass.Uint32),
                         cutlass.Uint32(1), sem="relaxed", scope="gpu")
    partial_check[slot_id.to(cutlass.Uint32)] = (
        row_tile * cutlass.Int32(1000) + nsg + cutlass.Int32(1)).to(cutlass.Uint32)
    cute.arch.fence_acq_rel_gpu()
    old = cute.arch.atomic_add(ready_count_owner.iterator + slot_id.to(cutlass.Uint32),
                               cutlass.Uint32(1), sem="acq_rel", scope="gpu")
    if (old + cutlass.Uint32(1)) == cutlass.Uint32(tp_size):
        cute.arch.atomic_exch(ar_probe.iterator + slot_id.to(cutlass.Uint32),
                              cutlass.Uint32(1), sem="release", scope="gpu")
    cute.arch.atomic_add(ctrl.iterator + C_OP_DONE, cutlass.Uint32(1),
                         sem="release", scope="gpu")


class FusedFaOprojAr:
    """5b-1b: persistent fused kernel with REAL FA (writes O_scratch_local), O_proj
    and AR still STUBs (single rank). FA payload = the validated FaWsAttnPacked logic
    inlined into the dispatch loop with LONG-LIVED pipeline states (consumer split
    fa_k_cons/fa_v_cons), per 设计稿 "Runtime task descriptor" + "long-lived pipeline
    state". Leader claims+broadcasts (mode,arg); FA runs on all 3 WGs; O_proj/AR
    leader-only stubs. M=N=D=128, kv_stages=2, causal varlen prompt prefill.
    """

    def __init__(self, num_fa, num_row_tiles, H_local, D, num_super_groups,
                 total_oproj, num_ctas, tp_size=1, kv_stages=2,
                 softmax_scale=None, acc_dtype=cutlass.Float32):
        self.num_fa = num_fa
        self.num_row_tiles = num_row_tiles
        self.H_local = H_local
        self.M = 128
        self.N = 128
        self.D = D
        self.num_super_groups = num_super_groups
        self.total_oproj = total_oproj
        self.num_ctas = num_ctas
        self.tp_size = tp_size
        self.kv_stages = kv_stages
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        self.num_dma_threads = 128
        self.mma_atom_layout_mnk = (2, 1, 1)
        self.num_mma_threads = 256
        self.num_mma_warps = self.num_mma_threads // 32
        self.threads = self.num_dma_threads + self.num_mma_threads   # 384
        self.align = 1024

    def _smem(self, dtype, rows, cols, stages):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, stages), order=(0, 1, 2))

    @cute.jit
    def __call__(self, ctrl, head_ready, oproj_queue, ready_count_owner, ar_probe,
                 ar_done_flag, fa_exec, oproj_exec, ar_exec, partial_check,
                 mQ, mK, mV, mOscr, mCuQ, mCuK, mFaB, mFaMb, stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem(dt, self.M, self.D, 1)
        sK_l = self._smem(dt, self.N, self.D, self.kv_stages)
        sV_l = self._smem(dt, self.N, self.D, self.kv_stages)

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        # packed-varlen K/V: view head LAST [tot,D,H]; atom box covers (token,D);
        # partition the returned tma_tensor (basis strides). See fa_varlen_sm90.
        mK_v = qlu.select(mK, [0, 2, 1])
        mV_v = qlu.select(mV, [0, 2, 1])
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK_v, cute.select(sK_l, mode=[0, 1]), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV_v, cute.select(sV_l, mode=[0, 1]), (self.N, self.D), num_multicast=1)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk, tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk, tiler_mn=(64, self.D),
            a_source=warpgroup.OperandSource.RMEM)

        @cute.struct
        class Smem:
            bc: cute.struct.MemRange[cutlass.Int32, 4]
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_v: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_l)], self.align]
            sK: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sK_l)], self.align]
            sV: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sV_l)], self.align]

        self.kernel(ctrl, head_ready, oproj_queue, ready_count_owner, ar_probe,
                    ar_done_flag, fa_exec, oproj_exec, ar_exec, partial_check,
                    tma_k, tK, tma_v, tV, mOscr, mQ, mCuQ, mCuK, mFaB, mFaMb,
                    mma_qk, mma_pv, sQ_l, sK_l, sV_l, Smem).launch(
            grid=[self.num_ctas, 1, 1], block=[self.threads, 1, 1],
            cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ctrl: cute.Tensor, head_ready: cute.Tensor,
               oproj_queue: cute.Tensor, ready_count_owner: cute.Tensor,
               ar_probe: cute.Tensor, ar_done_flag: cute.Tensor, fa_exec: cute.Tensor,
               oproj_exec: cute.Tensor, ar_exec: cute.Tensor, partial_check: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor,
               mOscr: cute.Tensor, mQ: cute.Tensor, mCuQ: cute.Tensor, mCuK: cute.Tensor,
               mFaB: cute.Tensor, mFaMb: cute.Tensor, mma_qk: cute.TiledMma,
               mma_pv: cute.TiledMma, sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        cls = bidx % 6

        num_fa = cutlass.const_expr(self.num_fa)
        total_oproj = cutlass.const_expr(self.total_oproj)
        H_local = cutlass.const_expr(self.H_local)
        num_super_groups = cutlass.const_expr(self.num_super_groups)
        tp_size = cutlass.const_expr(self.tp_size)
        num_ctas = cutlass.const_expr(self.num_ctas)
        slog2 = cutlass.const_expr(self.scale_log2)
        Dc = cutlass.const_expr(self.D)
        nthr = cutlass.const_expr(self.threads)
        MD = cutlass.const_expr(self.M * self.D)

        al = cutlass.utils.SmemAllocator()
        st = al.allocate(Smem)
        sma_ptr = st.bc.data_ptr()
        sma = cute.make_tensor(sma_ptr, cute.make_layout(4))
        sQ = st.sQ.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
        sK = st.sK.get_tensor(sK_l.outer, swizzle=sK_l.inner)
        sV = st.sV.get_tensor(sV_l.outer, swizzle=sV_l.inner)
        tx_k = cute.size_in_bytes(self.dt, cute.slice_(sK_l, (None, None, 0)))
        tx_v = cute.size_in_bytes(self.dt, cute.slice_(sV_l, (None, None, 0)))

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v)

        # pipelines + LONG-LIVED states (created ONCE, threaded across dispatch loop)
        pl_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_k.data_ptr(), num_stages=self.kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=tx_k)
        pl_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_v.data_ptr(), num_stages=self.kv_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=tx_v)
        fa_k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
        fa_v_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
        fa_k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
        fa_v_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)

        # consumer fragments / accumulators (persist; refilled per task)
        lane = tidx - self.num_dma_threads
        thr_qk = mma_qk.get_slice(lane)
        thr_pv = mma_pv.get_slice(lane)
        tCrQ = mma_qk.make_fragment_A(thr_qk.partition_A(sQ))
        tCrK = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
        sVt = qlu.transpose_view(sV)
        tCrV = mma_pv.make_fragment_B(thr_pv.partition_B(sVt))
        idS = cute.make_identity_tensor((self.M, self.N))
        idO = cute.make_identity_tensor((self.M, self.D))
        acc_S = cute.make_rmem_tensor(thr_qk.partition_C(idS).shape[:3], self.acc_dtype)
        acc_O = cute.make_rmem_tensor(thr_pv.partition_C(idO).shape[:3], self.acc_dtype)
        coord_mn = qlu.reshape_acc_to_mn(thr_qk.partition_C(idS))
        acc_mn = qlu.reshape_acc_to_mn(acc_S)
        acc_O_mn = qlu.reshape_acc_to_mn(acc_O)
        nrows = cutlass.const_expr(cute.size(acc_mn, mode=[0]))
        nkb_qk = cutlass.const_expr(cute.size(tCrQ, mode=[2]))
        row_max = cute.make_rmem_tensor(nrows, self.acc_dtype)
        row_sum = cute.make_rmem_tensor(nrows, self.acc_dtype)
        row_scale = cute.make_rmem_tensor(nrows, self.acc_dtype)
        tOrP_v = qlu.reshape_acc_to_frgA(acc_S)
        tOrP = cute.make_rmem_tensor_like(tOrP_v, self.dt)
        nkb_pv = cutlass.const_expr(cute.size(tOrP, mode=[2]))

        gsync(ctrl, C_GS_INIT, num_ctas, bidx, tidx)

        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
        else:
            cute.arch.setmaxregister_increase(232)

        looping = True
        while looping:
            if tidx == 0:
                mode, arg = schedule_pick(ctrl, oproj_queue, ar_probe, cls, num_fa, total_oproj)
                sma[0] = mode
                sma[1] = arg
            cute.arch.sync_threads()
            mode = sma[0]
            arg = sma[1]

            if mode == MODE_DONE:
                looping = False
            else:
                if mode == MODE_FA:
                    # ---- decode runtime FA descriptor (every thread) ----
                    ft = arg // cutlass.Int32(H_local)
                    head = arg % cutlass.Int32(H_local)
                    b = mFaB[ft]
                    mb = mFaMb[ft]
                    q_start = mCuQ[b]
                    k_start = mCuK[b]
                    q_len = mCuQ[b + cutlass.Int32(1)] - q_start
                    k_len = q_len
                    q_tile_pk = q_start + mb * cutlass.Int32(128)
                    mask_q_off = mb * cutlass.Int32(128)
                    nblk = mb + cutlass.Int32(1)

                    # cooperative packed Q load (zero-fill invalid rows), then publish
                    zero = self.dt(0.0)
                    for i in cutlass.range_constexpr(cdiv(MD, nthr)):
                        idx = tidx + i * nthr
                        if idx < MD:
                            row = idx // Dc
                            col = idx % Dc
                            if (mask_q_off + row) < q_len:
                                sQ[row, col, 0] = mQ[q_tile_pk + row, head, col]
                            else:
                                sQ[row, col, 0] = zero
                    cute.arch.sync_threads()

                    mK_cur = cute.domain_offset((k_start, None, None), mK)[None, None, head]
                    mV_cur = cute.domain_offset((k_start, None, None), mV)[None, None, head]
                    gK = cute.local_tile(mK_cur, (self.N, self.D), (None, 0))
                    gV = cute.local_tile(mV_cur, (self.N, self.D), (None, 0))
                    tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                        tma_k, 0, cute.make_layout(1),
                        cute.group_modes(sK, 0, cute.rank(sK) - 1),
                        cute.group_modes(gK, 0, cute.rank(gK) - 1))
                    tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                        tma_v, 0, cute.make_layout(1),
                        cute.group_modes(sV, 0, cute.rank(sV) - 1),
                        cute.group_modes(gV, 0, cute.rank(gV) - 1))

                    if wg_idx == 0:
                        if warp_idx == 0:
                            for j in cutlass.range(nblk, unroll=1):
                                pl_k.producer_acquire(fa_k_prod)
                                cute.copy(tma_k, tKgK[(None, j)], tKsK[(None, fa_k_prod.index)],
                                          tma_bar_ptr=pl_k.producer_get_barrier(fa_k_prod))
                                pl_k.producer_commit(fa_k_prod)
                                fa_k_prod.advance()
                                pl_v.producer_acquire(fa_v_prod)
                                cute.copy(tma_v, tVgV[(None, j)], tVsV[(None, fa_v_prod.index)],
                                          tma_bar_ptr=pl_v.producer_get_barrier(fa_v_prod))
                                pl_v.producer_commit(fa_v_prod)
                                fa_v_prod.advance()

                    if wg_idx >= 1:
                        acc_O.fill(0.0)
                        # Step A: block 0
                        pl_k.consumer_wait(fa_k_cons)
                        acc_S.fill(0.0)
                        mma_qk.set(warpgroup.Field.ACCUMULATE, True)
                        cute.nvgpu.warpgroup.fence()
                        for kb in cutlass.range_constexpr(nkb_qk):
                            cute.gemm(mma_qk, acc_S, tCrQ[(None, None, kb, 0)],
                                      tCrK[(None, None, kb, fa_k_cons.index)], acc_S)
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(0)
                        pl_k.consumer_release(fa_k_cons)
                        fa_k_cons.advance()
                        softmax_block_dyn(acc_mn, row_max, row_sum, row_scale, nrows, slog2,
                                          True, coord_mn, cutlass.Int32(0), mask_q_off, k_len)
                        tOrP.store(tOrP_v.load().to(self.dt))

                        # Middle: QK(j) overlaps PV(j-1)
                        for j in cutlass.range(1, nblk, unroll=1):
                            pl_k.consumer_wait(fa_k_cons)
                            acc_S.fill(0.0)
                            mma_qk.set(warpgroup.Field.ACCUMULATE, True)
                            cute.nvgpu.warpgroup.fence()
                            for kb in cutlass.range_constexpr(nkb_qk):
                                cute.gemm(mma_qk, acc_S, tCrQ[(None, None, kb, 0)],
                                          tCrK[(None, None, kb, fa_k_cons.index)], acc_S)
                            cute.nvgpu.warpgroup.commit_group()
                            pl_v.consumer_wait(fa_v_cons)
                            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
                            for kb in cutlass.range_constexpr(nkb_pv):
                                cute.gemm(mma_pv, acc_O, tOrP[(None, None, kb)],
                                          tCrV[(None, None, kb, fa_v_cons.index)], acc_O)
                            cute.nvgpu.warpgroup.commit_group()
                            cute.nvgpu.warpgroup.wait_group(1)
                            pl_k.consumer_release(fa_k_cons)
                            fa_k_cons.advance()
                            softmax_block_dyn(acc_mn, row_max, row_sum, row_scale, nrows, slog2,
                                              False, coord_mn, j, mask_q_off, k_len)
                            cute.nvgpu.warpgroup.wait_group(0)
                            pl_v.consumer_release(fa_v_cons)
                            fa_v_cons.advance()
                            for r in cutlass.range_constexpr(nrows):
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                            tOrP.store(qlu.reshape_acc_to_frgA(acc_S).load().to(self.dt))

                        # Step E: final PV
                        pl_v.consumer_wait(fa_v_cons)
                        mma_pv.set(warpgroup.Field.ACCUMULATE, True)
                        cute.nvgpu.warpgroup.fence()
                        for kb in cutlass.range_constexpr(nkb_pv):
                            cute.gemm(mma_pv, acc_O, tOrP[(None, None, kb)],
                                      tCrV[(None, None, kb, fa_v_cons.index)], acc_O)
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(0)
                        pl_v.consumer_release(fa_v_cons)
                        fa_v_cons.advance()

                        # finalize + store O_scratch[ft, :, head, :]
                        gOscr = mOscr[ft, None, head, None]
                        gO = cute.local_tile(gOscr, (self.M, self.D), (None, None))
                        tCgO = thr_pv.partition_C(gO[(None, None, 0, 0)])
                        for r in cutlass.range_constexpr(nrows):
                            if (mask_q_off + coord_mn[r, 0][0]) < q_len:
                                s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                                inv = cutlass.Float32(1.0) / s
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
                            else:
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * cutlass.Float32(0.0))
                        tCgO.store(acc_O.load().to(mOscr.element_type))

                    # ---- control tail: elected lane after all WGs done ----
                    cute.arch.sync_threads()
                    if tidx == 0:
                        cute.arch.atomic_add(fa_exec.iterator + arg.to(cutlass.Uint32),
                                             cutlass.Uint32(1), sem="relaxed", scope="gpu")
                        cute.arch.fence_acq_rel_gpu()
                        old = cute.arch.atomic_add(head_ready.iterator + ft.to(cutlass.Uint32),
                                                   cutlass.Uint32(1), sem="acq_rel", scope="gpu")
                        if (old + cutlass.Uint32(1)) == cutlass.Uint32(H_local):
                            publish_oproj(ctrl, oproj_queue, ft, num_super_groups)
                        cute.arch.atomic_add(ctrl.iterator + C_FA_DONE, cutlass.Uint32(1),
                                             sem="release", scope="gpu")

                if mode == MODE_OPROJ:
                    if tidx == 0:
                        do_oproj_stub_v2(ctrl, head_ready, oproj_exec, ready_count_owner,
                                         ar_probe, partial_check, arg, H_local,
                                         num_super_groups, tp_size)
                if mode == MODE_AR:
                    if tidx == 0:
                        do_ar(ctrl, ar_done_flag, ar_exec, arg)
                cute.arch.sync_threads()

        gsync(ctrl, C_GS_EXIT, num_ctas, bidx, tidx)
