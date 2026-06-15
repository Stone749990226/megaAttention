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
