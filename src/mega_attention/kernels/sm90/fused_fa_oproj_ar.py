#!/usr/bin/env python3
"""
Hopper SM90 persistent fused kernel for causal varlen full-prompt prefill.

Scope invariant:
  * One persistent grid runs FA -> O_proj -> tensor-parallel NVLS AllReduce.
  * The first-version semantic contract is causal varlen prefill with q_len == k_len.
  * FA/O_proj/AR share a 128-row task identity; no decode, append prefill, SplitKV,
    paged KV, non-causal, or non-SM90 path belongs here.

Scheduler protocol:
  * A CTA leader claims one work item and broadcasts (mode, arg) through shared memory.
  * FA work comes from a global task counter and maps to (fa_row_tile, head).
  * The last local head for a row tile publishes that row's O_proj super-group tasks
    through the ordered ready queue.
  * O_proj completion pushes readiness to the deterministic AR owner for the same
    (fa_row_tile, n_super_group) identity.
  * Termination is legal only after FA, O_proj, and this rank's AR-owner work all
    reached their done counts.

Agent note: keep comments anchored to stable protocol names and invariants, not to
design-document line numbers or temporary experiment phases.
"""
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.distributed as cda
from cutlass.cute.nvgpu import warpgroup
from quack import layout_utils as qlu

# FA payload helper shared with the standalone dynamic-varlen FA path.
from mega_attention.kernels.sm90.fa_varlen import softmax_block_dyn, LOG2E

# Mode codes broadcast through shared memory by the CTA leader.
MODE_IDLE = 0
MODE_FA = 1
MODE_OPROJ = 2
MODE_AR = 3
MODE_DONE = 4

# ctrl[] scalar slots. All entries are Uint32 and zero-initialized by the host.
C_FA_COUNTER = 0
C_FA_DONE = 1
C_OP_RESERVE = 2
C_OP_PUBLISH = 3
C_OP_CONSUME = 4
C_OP_DONE = 5
C_AR_DONE = 6
C_AR_CURSOR = 7
C_GS_INIT = 8
C_GS_EXIT = 9
NUM_CTRL = 10

FINISH_TAG = 0x80000000


# ============================================================================
# Device-wide synchronization helpers
@cute.jit
def gsync(ctrl: cute.Tensor, idx: cutlass.Constexpr,
          num_ctas: cutlass.Constexpr, bidx, tidx):
    """Single-kernel grid barrier using a host-zeroed ctrl[idx] word."""
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


@cute.jit
def nvl_barrier(sig_local: cute.Tensor, sig_mc: cute.Tensor, slot: cutlass.Constexpr,
                tp_size: cutlass.Constexpr, bidx, tidx):
    """Cross-rank NVLink barrier used around the persistent kernel body.

    Protocol invariant: only CTA0 participates in the cross-rank signal, while local
    grid_sync before/after this helper keeps the rest of the rank quiesced. The
    monotonic add-one signal needs a distinct slot for each barrier within a launch.
    """
    cute.arch.sync_threads()
    if bidx == 0:
        if tidx == 0:
            cda.multimem_red_add1(sig_mc.iterator + slot, order="release", scope="sys")
            cda.spin_lock_ld_lt_relaxed_wait(sig_local.iterator + slot,
                                             expected_val=cutlass.Int32(tp_size), scope="sys")
    cute.arch.sync_threads()


# ============================================================================
# Work-source claim helpers
@cute.jit
def try_fa(ctrl: cute.Tensor, num_fa: cutlass.Constexpr):
    """Bounded atomic claim of the next FA task. Returns (found, fa_task_id)."""
    found = cutlass.Int32(0)
    arg = cutlass.Int32(-1)
    cnt = cute.arch.load(ctrl.iterator + C_FA_COUNTER, cutlass.Uint32,
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
    """CAS pop from the published O_proj ready queue.

    Happens-before: seeing C_OP_PUBLISH with acquire makes the queue entry visible.
    CAS on C_OP_CONSUME gives single-consumer ownership of that slot.
    """
    found = cutlass.Int32(0)
    arg = cutlass.Int32(-1)
    head = cute.arch.load(ctrl.iterator + C_OP_CONSUME, cutlass.Uint32,
                          sem="relaxed", scope="gpu")
    tail = cute.arch.load(ctrl.iterator + C_OP_PUBLISH, cutlass.Uint32,
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
def try_claim_ar(ctrl: cute.Tensor, ar_ready_bits: cute.Tensor,
                 owner_words_alloc: cutlass.Constexpr, tp_size: cutlass.Constexpr,
                 rank: cutlass.Constexpr, local_owned_ar: cutlass.Constexpr):
    """Claim one AR owner task from this rank's owner-local ready bitset.

    The scan starts from a shared word cursor, finds the lowest set bit in a word,
    and clears it with atomicAnd. Clearing a set bit is the ownership claim.

    Identity invariant: owner_idx = word * 64 + bit_index, and
    ar_slot_id = owner_idx * tp_size + rank for this owner rank.

    Tail invariant: owner_slots_alloc may be rounded up. Bits whose owner_idx is
    outside local_owned_ar are invalid tail slots; clear-and-skip them so a stray bit
    cannot wedge the scanner or over-count C_AR_DONE.
    """
    found = cutlass.Int32(0)
    arg = cutlass.Int32(-1)
    found_w = cutlass.Uint32(0)
    start = cute.arch.load(ctrl.iterator + C_AR_CURSOR, cutlass.Uint32,
                           sem="relaxed", scope="gpu")
    i = cutlass.Uint32(0)
    n = cutlass.Uint32(owner_words_alloc)
    while (i < n) and (found == 0):
        w = (start + i) % n
        word = cute.arch.load(ar_ready_bits.iterator + w, cutlass.Int64,
                              sem="acquire", scope="gpu")
        if word != cutlass.Int64(0):
            lowest = word & (cutlass.Int64(0) - word)         # isolate lowest set bit
            bit_index = cute.arch.popc(lowest - cutlass.Int64(1))   # trailing-zero count
            owner_idx = w.to(cutlass.Int32) * cutlass.Int32(64) + bit_index.to(cutlass.Int32)
            old = cute.arch.atomic_and(ar_ready_bits.iterator + w, ~lowest,
                                       sem="acq_rel", scope="gpu")
            if (old & lowest) != cutlass.Int64(0):            # we cleared it -> won
                if owner_idx < cutlass.Int32(local_owned_ar):  # valid (non-tail) slot
                    found = cutlass.Int32(1)
                    found_w = w
                    arg = owner_idx * cutlass.Int32(tp_size) + cutlass.Int32(rank)
                # else: tail-invalid -> cleared above, skip (no claim, no count)
        i = i + cutlass.Uint32(1)
    if found != 0:
        cute.arch.atomic_exch(ctrl.iterator + C_AR_CURSOR, found_w,
                              sem="relaxed", scope="gpu")
    return found, arg


# ============================================================================
# Producers: FA -> O_proj ready queue
@cute.jit
def publish_oproj(ctrl: cute.Tensor, oproj_queue: cute.Tensor,
                  row_tile: cutlass.Int32, num_super_groups: cutlass.Constexpr):
    """Reserve, write, then ordered-publish this row tile's O_proj tasks.

    Queue invariant: consumers may read only [consume_head, publish_tail). Reservation
    can run ahead, but publish_tail advances in reservation order so consumers never
    observe holes or uninitialized queue entries.
    """
    n = cutlass.Uint32(num_super_groups)
    start = cute.arch.atomic_add(ctrl.iterator + C_OP_RESERVE, n,
                                 sem="relaxed", scope="gpu")
    base = row_tile.to(cutlass.Uint32) * n
    i = cutlass.Uint32(0)
    while i < n:
        oproj_queue[start + i] = base + i          # slot_id = row*nsg + i
        i = i + cutlass.Uint32(1)
    cute.arch.fence_acq_rel_gpu()                   # entries before publish
    # Ordered publish: wait until all earlier reservations are visible.
    spinning = True
    while spinning:
        pt = cute.arch.load(ctrl.iterator + C_OP_PUBLISH, cutlass.Uint32,
                            sem="acquire", scope="gpu")
        if pt == start:
            spinning = False
    cute.arch.atomic_add(ctrl.iterator + C_OP_PUBLISH, n,
                         sem="release", scope="gpu")


# ============================================================================
# Producers: O_proj -> AR owner readiness
@cute.jit
def publish_ar_ready(ready_count_owner: cute.Tensor, ar_ready_bits: cute.Tensor,
                     ar_slot_id: cutlass.Int32, tp_size: cutlass.Constexpr):
    """Single-rank push-to-owner readiness for one AR slot.

    The owner is deterministic: owner_rank = ar_slot_id % tp_size and
    owner_idx = ar_slot_id // tp_size. For tp_size == 1, the current rank is always
    the last arriver and sets the ready bit immediately.
    """
    owner_idx = ar_slot_id // cutlass.Int32(tp_size)
    old = cute.arch.atomic_add(ready_count_owner.iterator + owner_idx.to(cutlass.Uint32),
                               cutlass.Uint32(1), sem="acq_rel", scope="gpu")
    if (old + cutlass.Uint32(1)) == cutlass.Uint32(tp_size):
        word = (owner_idx // cutlass.Int32(64)).to(cutlass.Uint32)
        bit = cutlass.Int64(1) << (owner_idx % cutlass.Int32(64)).to(cutlass.Int64)
        cute.arch.atomic_or(ar_ready_bits.iterator + word, bit,
                            sem="release", scope="gpu")


@cute.jit
def publish_ar_ready_xrank(rc_local: cute.Tensor, rb_local: cute.Tensor,
                           ar_slot_id: cutlass.Int32, tp_size: cutlass.Constexpr,
                           rc_ptrs: cutlass.Constexpr, rb_ptrs: cutlass.Constexpr):
    """Cross-rank push-to-owner readiness for one AR slot.

    This uses sys-scope peer atomics to increment owner_rank's ready_count[owner_idx].
    The rank that observes tp_size arrivals publishes the owner-local ready bit.
    rc_ptrs/rb_ptrs are per-rank symmetric virtual addresses baked into the kernel.
    """
    owner_rank = ar_slot_id % cutlass.Int32(tp_size)
    owner_idx = ar_slot_id // cutlass.Int32(tp_size)
    word = (owner_idx // cutlass.Int32(64)).to(cutlass.Uint32)
    bit = cutlass.Int64(1) << (owner_idx % cutlass.Int32(64)).to(cutlass.Int64)
    for r in cutlass.range_constexpr(tp_size):
        if owner_rank == cutlass.Int32(r):
            rc_peer = cute.make_ptr(rc_local.element_type, rc_ptrs[r],
                                    cute.AddressSpace.gmem, assumed_align=4)
            old = cute.arch.atomic_add(rc_peer + owner_idx.to(cutlass.Uint32),
                                       cutlass.Uint32(1), sem="acq_rel", scope="sys")
            if (old + cutlass.Uint32(1)) == cutlass.Uint32(tp_size):
                rb_peer = cute.make_ptr(rb_local.element_type, rb_ptrs[r],
                                        cute.AddressSpace.gmem, assumed_align=8)
                cute.arch.atomic_or(rb_peer + word, bit, sem="release", scope="sys")


@cute.jit
def do_ar(ctrl: cute.Tensor, ar_done_bits: cute.Tensor,
         ar_slot_id: cutlass.Int32, tp_size: cutlass.Constexpr):
    """Complete an AR owner task for the single-rank identity case.

    For tp_size == 1, C_sym partial is already the final value. The done bit still
    enforces single completion and drives the common scheduler termination protocol.
    """
    owner_idx = ar_slot_id // cutlass.Int32(tp_size)
    word = (owner_idx // cutlass.Int32(64)).to(cutlass.Uint32)
    bit = cutlass.Int64(1) << (owner_idx % cutlass.Int32(64)).to(cutlass.Int64)
    old_done = cute.arch.atomic_or(ar_done_bits.iterator + word, bit,
                                   sem="acq_rel", scope="gpu")
    if (old_done & bit) == cutlass.Int64(0):
        # Single-rank identity path: no data movement, only terminal accounting.
        cute.arch.atomic_add(ctrl.iterator + C_AR_DONE, cutlass.Uint32(1),
                             sem="release", scope="gpu")


# ============================================================================
# CTA leader scheduler
@cute.jit
def schedule_pick(ctrl, oproj_queue, ar_ready_bits,
                  role, num_fa: cutlass.Constexpr, total_oproj: cutlass.Constexpr,
                  owner_words_alloc: cutlass.Constexpr, tp_size: cutlass.Constexpr,
                  rank: cutlass.Constexpr, local_owned_ar: cutlass.Constexpr):
    """Pick one task for the CTA leader and return (mode, arg).

    Role controls only the preferred probe order. Every CTA falls through to the other
    work sources, so role assignment is a bias, not a static partition.
    """
    mode = cutlass.Int32(MODE_IDLE)
    arg = cutlass.Int32(-1)

    fa_d = cute.arch.load(ctrl.iterator + C_FA_DONE, cutlass.Uint32,
                          sem="acquire", scope="gpu")
    op_d = cute.arch.load(ctrl.iterator + C_OP_DONE, cutlass.Uint32,
                          sem="acquire", scope="gpu")
    ar_d = cute.arch.load(ctrl.iterator + C_AR_DONE, cutlass.Uint32,
                          sem="acquire", scope="gpu")
    all_done = (fa_d >= num_fa) and (op_d >= total_oproj) and (ar_d >= local_owned_ar)

    if all_done:
        mode = cutlass.Int32(MODE_DONE)
    else:
        # Preference order by role; all roles fall through to all work sources.
        s0 = cutlass.Int32(MODE_FA); s1 = cutlass.Int32(MODE_OPROJ); s2 = cutlass.Int32(MODE_AR)
        if role == 1:
            s0 = cutlass.Int32(MODE_OPROJ); s1 = cutlass.Int32(MODE_AR); s2 = cutlass.Int32(MODE_FA)
        if role == 2:
            s0 = cutlass.Int32(MODE_AR); s1 = cutlass.Int32(MODE_FA); s2 = cutlass.Int32(MODE_OPROJ)

        found = cutlass.Int32(0)
        # Try s0, then s1, then s2.
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
                    f, a = try_claim_ar(ctrl, ar_ready_bits, owner_words_alloc, tp_size,
                                        rank, local_owned_ar)
                if f != 0:
                    found = cutlass.Int32(1)
                    mode = src
                    arg = a
            ci = ci + cutlass.Int32(1)
    return mode, arg


# ============================================================================
# Driver and kernel


class FusedFaOprojAr:
    """CuTe DSL wrapper for the persistent fused kernel.

    Runtime task descriptors are decoded inside the dispatch loop from (mode, arg).
    Pipeline objects and PipelineState cursors are long-lived across tasks; payload
    fragments and accumulators are mode-local. WG0 is the TMA producer, while WG1/WG2
    consume the lower and upper 64 rows of the 128-row tile.
    """

    def __init__(self, num_fa, num_row_tiles, H_local, D, num_super_groups,
                 total_oproj, num_ctas, hidden, tp_size=1, rank=0, kv_stages=2,
                 q_per_kv=1,
                 N_TILE=128, super_group_n_tiles=4, K_CHUNK=64, oproj_stages=4,
                 csym_mc_ptr=0, nvl_mc_ptr=0, nvl_local_ptr=0, rc_ptrs=(), rb_ptrs=(),
                 softmax_scale=None, acc_dtype=cutlass.Float32,
                 w_fa=4, w_oproj=1, w_ar=1):
        # Multi-rank NVLS pointers are baked as closure constants. Do not move them to
        # cute.compile args: 64-bit virtual addresses would be truncated.
        self.csym_mc_ptr = csym_mc_ptr          # C_sym multicast VA (multimem reduce)
        self.nvl_mc_ptr = nvl_mc_ptr            # nvl_barrier signal multicast VA
        self.nvl_local_ptr = nvl_local_ptr      # this rank's nvl signal VA (spin read)
        self.rc_ptrs = tuple(rc_ptrs)           # per-rank ready_count_owner peer VAs
        self.rb_ptrs = tuple(rb_ptrs)           # per-rank ar_ready_bits peer VAs
        self.num_fa = num_fa
        self.num_row_tiles = num_row_tiles
        self.H_local = H_local
        # 标准 GQA：H_local 是 Q head 数，K/V 复用 H_kv_local 个 head。
        assert H_local % q_per_kv == 0, (H_local, q_per_kv)
        self.q_per_kv = q_per_kv                 # 连续分组比；==1 即 MHA
        self.H_kv_local = H_local // q_per_kv
        self.M = 128
        self.N = 128
        self.D = D
        self.num_super_groups = num_super_groups
        self.total_oproj = total_oproj
        self.num_ctas = num_ctas
        self.tp_size = tp_size
        self.rank = rank
        # CTA role preference weights. These are scheduler biases, not fixed roles.
        self.w_fa = w_fa
        self.w_oproj = w_oproj
        self.w_ar = w_ar if tp_size > 1 else 0
        # Deterministic AR owner mapping:
        # owner_rank = ar_slot_id % tp_size; owner_idx = ar_slot_id // tp_size.
        self.owner_slots_alloc = (total_oproj + tp_size - 1) // tp_size
        self.owner_words_alloc = (self.owner_slots_alloc + 63) // 64
        self.local_owned_ar_tasks = max(total_oproj - rank, 0)
        self.local_owned_ar_tasks = (self.local_owned_ar_tasks + tp_size - 1) // tp_size
        self.kv_stages = kv_stages
        # O_proj/AR tile parameters for the fixed first-version kernel variant.
        self.hidden = hidden
        self.N_TILE = N_TILE
        self.super_group_n_tiles = super_group_n_tiles
        self.K_local = H_local * D
        self.K_CHUNK = K_CHUNK
        self.n_kchunks = self.K_local // K_CHUNK
        self.oproj_stages = oproj_stages
        self.num_out_n_tiles = (hidden + N_TILE - 1) // N_TILE
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
    def __call__(self, ctrl, head_ready, oproj_queue, ready_count_owner, ar_ready_bits,
                 ar_done_bits,
                 mQ, mK, mV, mOscr, mWo, mCsym, mCuQ, mCuK, mFaB, mFaMb,
                 stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem(dt, self.M, self.D, 1)
        sK_l = self._smem(dt, self.N, self.D, self.kv_stages)
        sV_l = self._smem(dt, self.N, self.D, self.kv_stages)
        # O_proj SMEM operands: A[M,K_CHUNK] and Wo[K_CHUNK,N_TILE], both staged.
        sA_l = self._smem(dt, self.M, self.K_CHUNK, self.oproj_stages)
        sWo_l = self._smem(dt, self.K_CHUNK, self.N_TILE, self.oproj_stages)

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        # Packed-varlen K/V are viewed as head-last [tot, D, H] so the TMA atom
        # copies a contiguous (token, D) tile for the selected head.
        mK_v = qlu.select(mK, [0, 2, 1])
        mV_v = qlu.select(mV, [0, 2, 1])
        # Q uses the same head-last packed-varlen view and a one-stage TMA pipeline.
        mQ_v = qlu.select(mQ, [0, 2, 1])
        tma_q, tQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mQ_v, cute.select(sQ_l, mode=[0, 1]), (self.M, self.D), num_multicast=1)
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK_v, cute.select(sK_l, mode=[0, 1]), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV_v, cute.select(sV_l, mode=[0, 1]), (self.N, self.D), num_multicast=1)

        # O_proj A view invariant:
        # O_scratch[R,128,H,D] is contiguous in (H,D), so flattening to
        # [R*128, K_local] preserves K = h*D + d. The O_proj M tile index is ft.
        R = cutlass.const_expr(self.num_row_tiles)
        K_local = cutlass.const_expr(self.K_local)
        mOscr2d = cute.make_tensor(
            mOscr.iterator,
            cute.make_layout((R * self.M, K_local, 1), stride=(K_local, 1, 1)))
        tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mOscr2d, cute.slice_(sA_l, (None, None, 0)), (self.M, self.K_CHUNK), num_multicast=1)
        tma_wo, tWo = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mWo, cute.slice_(sWo_l, (None, None, 0)), (self.K_CHUNK, self.N_TILE), num_multicast=1)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk, tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk, tiler_mn=(64, self.D),
            a_source=warpgroup.OperandSource.RMEM)
        # O_proj GEMM operand contract: A is K-major in SMEM; transpose_view(sWo)
        # presents W_o as the MN-major B operand expected by the MMA atom.
        mma_op = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk,
            tiler_mn=(64, self.N_TILE))

        # SMEM overlay invariant:
        # FA tensors (sQ/sK/sV) and O_proj tensors (sA/sWo) share one byte range. This
        # is a union, not a sum. A CTA is in exactly one payload mode at a time, and
        # each mode drains its TMA/WGMMA work before the next mode may overwrite the
        # overlay. Pipeline mbarriers stay separate from the tensor overlay.
        eltb = cutlass.const_expr(dt.width // 8)
        ae = cutlass.const_expr(self.align // eltb)                 # align in elements

        def _au(x):                                                 # align-up (elems)
            return ((x + ae - 1) // ae) * ae
        cq, ck, cv = (cutlass.const_expr(cute.cosize(sQ_l)),
                      cutlass.const_expr(cute.cosize(sK_l)),
                      cutlass.const_expr(cute.cosize(sV_l)))
        ca, cwo = (cutlass.const_expr(cute.cosize(sA_l)),
                   cutlass.const_expr(cute.cosize(sWo_l)))
        off_sK = cutlass.const_expr(_au(cq))
        off_sV = cutlass.const_expr(_au(off_sK + ck))
        fa_total = cutlass.const_expr(_au(off_sV + cv))
        off_sWo = cutlass.const_expr(_au(ca))
        op_total = cutlass.const_expr(_au(off_sWo + cwo))
        overlay_n = cutlass.const_expr(max(fa_total, op_total))
        self._off_sK = off_sK
        self._off_sV = off_sV
        self._off_sWo = off_sWo

        @cute.struct
        class Smem:
            bc: cute.struct.MemRange[cutlass.Int32, 4]
            mbar_q: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_v: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_ab: cute.struct.MemRange[cutlass.Int64, self.oproj_stages * 2]
            overlay: cute.struct.Align[cute.struct.MemRange[dt, overlay_n], self.align]

        # Multi-rank NVLS views from baked constant VAs. For tp_size == 1 they are
        # null/unused but still constructed to keep the compiled call signature fixed.
        csym_mc = cute.make_tensor(
            cute.make_ptr(mCsym.element_type, self.csym_mc_ptr, cute.AddressSpace.gmem,
                          assumed_align=16), mCsym.layout)
        nvl_mc = cute.make_tensor(
            cute.make_ptr(ctrl.element_type, self.nvl_mc_ptr, cute.AddressSpace.gmem,
                          assumed_align=4), cute.make_layout(4))
        nvl_local = cute.make_tensor(
            cute.make_ptr(ctrl.element_type, self.nvl_local_ptr, cute.AddressSpace.gmem,
                          assumed_align=4), cute.make_layout(4))

        self.kernel(ctrl, head_ready, oproj_queue, ready_count_owner, ar_ready_bits,
                    ar_done_bits,
                    tma_k, tK, tma_v, tV, mOscr, mOscr2d, mWo, mCsym, csym_mc,
                    nvl_mc, nvl_local, tma_q, tQ, mCuQ, mCuK,
                    mFaB, mFaMb, tma_a, tA, tma_wo, tWo,
                    mma_qk, mma_pv, mma_op, sQ_l, sK_l, sV_l, sA_l, sWo_l, Smem).launch(
            grid=[self.num_ctas, 1, 1], block=[self.threads, 1, 1],
            cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, ctrl: cute.Tensor, head_ready: cute.Tensor,
               oproj_queue: cute.Tensor, ready_count_owner: cute.Tensor,
               ar_ready_bits: cute.Tensor, ar_done_bits: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor,
               mOscr: cute.Tensor, mOscr2d: cute.Tensor, mWo: cute.Tensor, mCsym: cute.Tensor,
               csym_mc: cute.Tensor, nvl_mc: cute.Tensor, nvl_local: cute.Tensor,
               tma_q: cute.CopyAtom, tQ: cute.Tensor, mCuQ: cute.Tensor, mCuK: cute.Tensor,
               mFaB: cute.Tensor, mFaMb: cute.Tensor,
               tma_a: cute.CopyAtom, tA: cute.Tensor, tma_wo: cute.CopyAtom, tWo: cute.Tensor,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma, mma_op: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, sA_l: cute.ComposedLayout,
               sWo_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        w_fa = cutlass.const_expr(self.w_fa)
        w_fo = cutlass.const_expr(self.w_fa + self.w_oproj)
        w_m = cutlass.const_expr(self.w_fa + self.w_oproj + self.w_ar)
        k = bidx % cutlass.Int32(w_m)
        role = cutlass.Int32(0)
        if k >= cutlass.Int32(w_fa):
            role = cutlass.Int32(1)
        if k >= cutlass.Int32(w_fo):
            role = cutlass.Int32(2)

        num_fa = cutlass.const_expr(self.num_fa)
        total_oproj = cutlass.const_expr(self.total_oproj)
        H_local = cutlass.const_expr(self.H_local)
        num_super_groups = cutlass.const_expr(self.num_super_groups)
        tp_size = cutlass.const_expr(self.tp_size)
        num_ctas = cutlass.const_expr(self.num_ctas)
        slog2 = cutlass.const_expr(self.scale_log2)
        nthr = cutlass.const_expr(self.threads)
        # O_proj compile-time parameters for this launch variant.
        n_kchunks = cutlass.const_expr(self.n_kchunks)
        sgnt = cutlass.const_expr(self.super_group_n_tiles)
        N_TILE = cutlass.const_expr(self.N_TILE)
        hidden = cutlass.const_expr(self.hidden)
        num_out_n_tiles = cutlass.const_expr(self.num_out_n_tiles)
        # AR owner-local compile-time parameters for this rank.
        rank = cutlass.const_expr(self.rank)
        owner_words_alloc = cutlass.const_expr(self.owner_words_alloc)
        local_owned_ar = cutlass.const_expr(self.local_owned_ar_tasks)

        al = cutlass.utils.SmemAllocator()
        st = al.allocate(Smem)
        sma_ptr = st.bc.data_ptr()
        sma = cute.make_tensor(sma_ptr, cute.make_layout(4))
        # SMEM overlay materialization:
        # FA and O_proj tensors carve the same backing range. The swizzle must live on
        # the recast pointer, not just on the layout, for WGMMA to interpret SMEM
        # addresses correctly.
        ov = st.overlay.data_ptr()
        sQ = cute.make_tensor(cute.recast_ptr(ov, sQ_l.inner, self.dt), sQ_l.outer)
        sK = cute.make_tensor(cute.recast_ptr(ov + cutlass.const_expr(self._off_sK), sK_l.inner, self.dt), sK_l.outer)
        sV = cute.make_tensor(cute.recast_ptr(ov + cutlass.const_expr(self._off_sV), sV_l.inner, self.dt), sV_l.outer)
        sA = cute.make_tensor(cute.recast_ptr(ov, sA_l.inner, self.dt), sA_l.outer)
        sWo = cute.make_tensor(cute.recast_ptr(ov + cutlass.const_expr(self._off_sWo), sWo_l.inner, self.dt), sWo_l.outer)
        tx_q = cute.size_in_bytes(self.dt, cute.slice_(sQ_l, (None, None, 0)))
        tx_k = cute.size_in_bytes(self.dt, cute.slice_(sK_l, (None, None, 0)))
        tx_v = cute.size_in_bytes(self.dt, cute.slice_(sV_l, (None, None, 0)))
        tx_ab = (cute.size_in_bytes(self.dt, cute.slice_(sA_l, (None, None, 0)))
                 + cute.size_in_bytes(self.dt, cute.slice_(sWo_l, (None, None, 0))))

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_wo)

        # Long-lived pipeline invariant:
        # Pipeline objects and PipelineState cursors are created once and threaded
        # across all tasks. Do not recreate PipelineState inside a mode branch unless
        # the matching SMEM mbarrier is also reinitialized.
        pl_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_q.data_ptr(), num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=tx_q)
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
        pl_ab = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_ab.data_ptr(), num_stages=self.oproj_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=tx_ab)
        fa_k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
        fa_v_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
        fa_k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
        fa_v_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
        fa_q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        fa_q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        op_ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.oproj_stages)
        op_ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.oproj_stages)

        # Register lifetime invariant:
        # Large consumer fragments/accumulators are created inside their payload mode
        # and only for consumer warp groups. Only protocol state lives across tasks.
        # This keeps the FA/O_proj payload lifetimes explicit for future agent edits.
        lane = tidx - self.num_dma_threads

        gsync(ctrl, C_GS_INIT, num_ctas, bidx, tidx)
        if cutlass.const_expr(tp_size > 1):
            nvl_barrier(nvl_local, nvl_mc, 0, tp_size, bidx, tidx)   # init: all ranks ready

        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
        else:
            cute.arch.setmaxregister_increase(232)

        looping = True
        while looping:
            if tidx == 0:
                mode, arg = schedule_pick(ctrl, oproj_queue, ar_ready_bits, role, num_fa,
                                          total_oproj, owner_words_alloc, tp_size, rank,
                                          local_owned_ar)
                sma[0] = mode
                sma[1] = arg
            cute.arch.sync_threads()
            mode = sma[0]
            arg = sma[1]

            if mode == MODE_DONE:
                looping = False
            else:
                if mode == MODE_FA:
                    # Decode runtime FA descriptor. Every thread derives the same
                    # descriptor from arg; no CTA-local descriptor protocol is used.
                    ft = arg // cutlass.Int32(H_local)
                    head = arg % cutlass.Int32(H_local)          # Q head
                    # 标准 GQA：连续 q_per_kv 个 Q head 共享一个 K/V head。
                    q_per_kv = cutlass.const_expr(self.q_per_kv)
                    kv_head = head // cutlass.Int32(q_per_kv)
                    b = mFaB[ft]
                    mb = mFaMb[ft]
                    q_start = mCuQ[b]
                    k_start = mCuK[b]
                    q_len = mCuQ[b + cutlass.Int32(1)] - q_start
                    k_len = mCuK[b + cutlass.Int32(1)] - k_start
                    offset = k_len - q_len               # bottom-right aligned causal
                    mask_q_off = mb * cutlass.Int32(128)  # = m_idx_min (seq-local q tile)
                    # FA4 BlockInfo.get_n_block_min_max (纯 causal, 非 local/split):
                    m_idx_max = (mb + cutlass.Int32(1)) * cutlass.Int32(128)
                    nblk = min(cute.ceil_div(k_len, 128),
                               cute.ceil_div(m_idx_max + offset, 128))
                    # 首个需 causal mask 的块 (get_n_block_min_causal_local_mask)。
                    n_blk_causal = cutlass.max(cutlass.Int32(0),
                                               (mask_q_off + offset) // cutlass.Int32(128))
                    # peeled 首块 (n=nblk-1) 之外, 还需 causal mask 的中间块数。offset==0
                    # 时为 0 -> 退化为完整 prompt prefill 的"仅对角块 mask"路径。
                    causal_cnt = cutlass.max(cutlass.Int32(0),
                                             nblk - cutlass.Int32(1) - n_blk_causal)

                    mQ_cur = cute.domain_offset((q_start, None, None), tQ)[None, None, head]
                    gQ = cute.local_tile(mQ_cur, (self.M, self.D), (None, 0))
                    mK_cur = cute.domain_offset((k_start, None, None), mK)[None, None, kv_head]
                    mV_cur = cute.domain_offset((k_start, None, None), mV)[None, None, kv_head]
                    gK = cute.local_tile(mK_cur, (self.N, self.D), (None, 0))
                    gV = cute.local_tile(mV_cur, (self.N, self.D), (None, 0))
                    tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
                        tma_q, 0, cute.make_layout(1),
                        cute.group_modes(sQ, 0, cute.rank(sQ) - 1),
                        cute.group_modes(gQ, 0, cute.rank(gQ) - 1))
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
                            # Q TMA invariant:
                            # Issue Q once before the K/V stream so it is available
                            # for the first QK block. Padding rows in the final row tile
                            # are not zero-filled; they are masked before O_scratch is
                            # made visible to O_proj.
                            pl_q.producer_acquire(fa_q_prod)
                            cute.copy(tma_q, tQgQ[(None, mb)], tQsQ[(None, fa_q_prod.index)],
                                      tma_bar_ptr=pl_q.producer_get_barrier(fa_q_prod))
                            pl_q.producer_commit(fa_q_prod)
                            fa_q_prod.advance()
                            for i in cutlass.range(nblk, unroll=1):
                                j = nblk - cutlass.Int32(1) - i      # 右->左：对角块先 load
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
                        # FA payload fragments are consumer-WG local and die at the end
                        # of this mode branch.
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

                        acc_O.fill(0.0)
                        # Q must be fully loaded before any QK MMA reads sQ.
                        pl_q.consumer_wait(fa_q_cons)
                        # 右->左：第一个处理的是对角块 (n = nblk-1)，唯一需要 causal+k_len mask。
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
                                          True, coord_mn, nblk - cutlass.Int32(1),
                                          mask_q_off, k_len, offset=offset)
                        tOrP.store(tOrP_v.load().to(self.dt))

                        # 右->左中间块分两段 (n = nblk-1-i, 递减)。前 causal_cnt 个块跨过
                        # causal 边界, 需逐元素 mask；其余块全在边界左侧, 走 no-mask 快路径。
                        # 两段共享同一 fa_k_cons/fa_v_cons/tOrP/acc_O pipeline 计数器顺序推进。
                        for i in cutlass.range(1, cutlass.Int32(1) + causal_cnt, unroll=1):
                            n = nblk - cutlass.Int32(1) - i
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
                                              False, coord_mn, n, mask_q_off, k_len,
                                              need_mask=True, offset=offset)
                            cute.nvgpu.warpgroup.wait_group(0)
                            pl_v.consumer_release(fa_v_cons)
                            fa_v_cons.advance()
                            for r in cutlass.range_constexpr(nrows):
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                            tOrP.store(qlu.reshape_acc_to_frgA(acc_S).load().to(self.dt))

                        # no-mask 区间: n 全在 causal 边界左侧、完全可见, 跳过逐元素比较。
                        for i in cutlass.range(cutlass.Int32(1) + causal_cnt, nblk, unroll=1):
                            n = nblk - cutlass.Int32(1) - i
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
                                              False, coord_mn, n, mask_q_off, k_len,
                                              need_mask=False)
                            cute.nvgpu.warpgroup.wait_group(0)
                            pl_v.consumer_release(fa_v_cons)
                            fa_v_cons.advance()
                            for r in cutlass.range_constexpr(nrows):
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                            tOrP.store(qlu.reshape_acc_to_frgA(acc_S).load().to(self.dt))

                        # Mode-local lifetime: after all QK work has waited, sQ is dead
                        # and the Q pipeline stage can be released for the next FA task.
                        pl_q.consumer_release(fa_q_cons)
                        fa_q_cons.advance()
                        # Final PV consumes the last softmax block.
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

                        # Store this head's O_scratch tile. The later head_ready release
                        # is the visibility handoff to O_proj.
                        gOscr = mOscr[ft, None, head, None]
                        gO = cute.local_tile(gOscr, (self.M, self.D), (None, None))
                        tCgO = thr_pv.partition_C(gO[(None, None, 0, 0)])
                        for r in cutlass.range_constexpr(nrows):
                            # Hazard: warp_reduction_sum is a warp-collective shuffle.
                            # Call it unconditionally for every row. Guarding the call
                            # behind valid_m can diverge a warp at a partial row tile and
                            # hang the CTA.
                            s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                            if (mask_q_off + coord_mn[r, 0][0]) < q_len:
                                inv = cutlass.Float32(1.0) / s
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
                            else:
                                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * cutlass.Float32(0.0))
                        tCgO.store(acc_O.load().to(mOscr.element_type))

                    # FA control tail. After the CTA sync, all O_scratch stores for this
                    # FA task are complete from the CTA's point of view.
                    cute.arch.sync_threads()
                    if tidx == 0:
                        cute.arch.fence_acq_rel_gpu()            # O_scratch store before ready
                        old = cute.arch.atomic_add(head_ready.iterator + ft.to(cutlass.Uint32),
                                                   cutlass.Uint32(1), sem="acq_rel", scope="gpu")
                        if (old + cutlass.Uint32(1)) == cutlass.Uint32(H_local):
                            publish_oproj(ctrl, oproj_queue, ft, num_super_groups)
                        cute.arch.atomic_add(ctrl.iterator + C_FA_DONE, cutlass.Uint32(1),
                                             sem="release", scope="gpu")

                if mode == MODE_OPROJ:
                    # Decode O_proj slot. arg is the stable slot_id:
                    # fa_row_tile = slot_id / num_super_groups,
                    # n_super_group = slot_id % num_super_groups.
                    ft = arg // cutlass.Int32(num_super_groups)
                    nsg = arg % cutlass.Int32(num_super_groups)
                    base_out = nsg * cutlass.Int32(sgnt)
                    rem_sg = cutlass.Int32(num_out_n_tiles) - base_out
                    valid_n_tiles = rem_sg
                    if rem_sg > cutlass.Int32(sgnt):
                        valid_n_tiles = cutlass.Int32(sgnt)
                    b = mFaB[ft]
                    mb = mFaMb[ft]
                    q_start = mCuQ[b]
                    q_len = mCuQ[b + cutlass.Int32(1)] - q_start
                    valid_m = q_len - mb * cutlass.Int32(128)
                    if valid_m > cutlass.Int32(128):
                        valid_m = cutlass.Int32(128)

                    # TMA tensor contract: tA/tWo carry basis strides matching the atom's
                    # gmem basis; concrete ft, k_chunk, and out_n are selected at copy time.
                    gA = cute.local_tile(tA, (self.M, self.K_CHUNK), (None, None, None))
                    gWo = cute.local_tile(tWo, (self.K_CHUNK, N_TILE), (None, None))
                    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                        tma_a, 0, cute.make_layout(1),
                        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
                    tWosWo, tWogWo = cute.nvgpu.cpasync.tma_partition(
                        tma_wo, 0, cute.make_layout(1),
                        cute.group_modes(sWo, 0, 2), cute.group_modes(gWo, 0, 2))

                    if wg_idx == 0:
                        if warp_idx == 0:
                            for sg in cutlass.range(valid_n_tiles, unroll=1):
                                out_n = base_out + sg
                                for kc in cutlass.range_constexpr(n_kchunks):
                                    pl_ab.producer_acquire(op_ab_prod)
                                    bar = pl_ab.producer_get_barrier(op_ab_prod)
                                    cute.copy(tma_a, tAgA[(None, ft, kc, 0)],
                                              tAsA[(None, op_ab_prod.index)], tma_bar_ptr=bar)
                                    cute.copy(tma_wo, tWogWo[(None, kc, out_n)],
                                              tWosWo[(None, op_ab_prod.index)], tma_bar_ptr=bar)
                                    pl_ab.producer_commit(op_ab_prod)
                                    op_ab_prod.advance()

                    if wg_idx >= 1:
                        # O_proj payload fragments are consumer-WG local and die at the
                        # end of this mode branch.
                        thr_op = mma_op.get_slice(lane)
                        tCrA = mma_op.make_fragment_A(thr_op.partition_A(sA))
                        sWot = qlu.transpose_view(sWo)
                        tCrWo = mma_op.make_fragment_B(thr_op.partition_B(sWot))
                        idC = cute.make_identity_tensor((self.M, N_TILE))
                        acc_C = cute.make_rmem_tensor(thr_op.partition_C(idC).shape[:3], self.acc_dtype)
                        coord_C = qlu.reshape_acc_to_mn(thr_op.partition_C(idC))
                        acc_C_mn = qlu.reshape_acc_to_mn(acc_C)
                        nrows = cutlass.const_expr(cute.size(acc_C_mn, mode=[0]))
                        ncols_C = cutlass.const_expr(cute.size(acc_C_mn, mode=[1]))
                        nkb_op = cutlass.const_expr(cute.size(tCrA, mode=[2]))

                        for sg in cutlass.range(valid_n_tiles, unroll=1):
                            out_n = base_out + sg
                            remn = cutlass.Int32(hidden) - out_n * cutlass.Int32(N_TILE)
                            valid_n = remn
                            if remn > cutlass.Int32(N_TILE):
                                valid_n = cutlass.Int32(N_TILE)
                            acc_C.fill(0.0)
                            mma_op.set(warpgroup.Field.ACCUMULATE, True)
                            cute.nvgpu.warpgroup.fence()
                            for kc in cutlass.range_constexpr(n_kchunks):
                                pl_ab.consumer_wait(op_ab_cons)
                                for kb in cutlass.range_constexpr(nkb_op):
                                    cute.gemm(mma_op, acc_C,
                                              tCrA[(None, None, kb, op_ab_cons.index)],
                                              tCrWo[(None, None, kb, op_ab_cons.index)], acc_C)
                                cute.nvgpu.warpgroup.commit_group()
                                cute.nvgpu.warpgroup.wait_group(0)
                                pl_ab.consumer_release(op_ab_cons)
                                op_ab_cons.advance()
                            # Tail invariant: only valid token rows and valid hidden
                            # columns are written to C_sym. Invalid tail elements are not
                            # required to be cleared.
                            gC = mCsym[ft, None, out_n, None]
                            tCgC = thr_op.partition_C(gC)
                            gC_mn = qlu.reshape_acc_to_mn(tCgC)
                            for r in cutlass.range_constexpr(nrows):
                                crow = coord_C[r, None]
                                for c in cutlass.range_constexpr(ncols_C):
                                    m = crow[c][0]
                                    n = crow[c][1]
                                    if (m < valid_m) and (n < valid_n):
                                        gC_mn[r, c] = acc_C_mn[r, c].to(self.dt)

                    cute.arch.sync_threads()
                    if tidx == 0:
                        cute.arch.fence_acq_rel_gpu()            # C_sym partial before ready
                        # arg is also ar_slot_id. The last rank to publish readiness sets
                        # the owner-local ready bit.
                        if cutlass.const_expr(tp_size == 1):
                            publish_ar_ready(ready_count_owner, ar_ready_bits, arg, tp_size)
                        else:
                            publish_ar_ready_xrank(ready_count_owner, ar_ready_bits, arg,
                                                   tp_size, self.rc_ptrs, self.rb_ptrs)
                        cute.arch.atomic_add(ctrl.iterator + C_OP_DONE, cutlass.Uint32(1),
                                             sem="release", scope="gpu")
                if mode == MODE_AR:
                    if cutlass.const_expr(tp_size == 1):
                        if tidx == 0:
                            do_ar(ctrl, ar_done_bits, arg, tp_size)
                    else:
                        # AR owner reduces the whole C_sym tile through the multicast
                        # view, parallelized over contiguous 8-bf16 chunks in N.
                        ft_ar = arg // cutlass.Int32(num_super_groups)
                        nsg_ar = arg % cutlass.Int32(num_super_groups)
                        base_ar = nsg_ar * cutlass.Int32(sgnt)
                        rem_ar = cutlass.Int32(num_out_n_tiles) - base_ar
                        vnt_ar = rem_ar
                        if rem_ar > cutlass.Int32(sgnt):
                            vnt_ar = cutlass.Int32(sgnt)
                        npr = cutlass.const_expr(N_TILE // 8)        # 8-bf16 chunks per row
                        total = vnt_ar * cutlass.Int32(128) * cutlass.Int32(npr)
                        c = cutlass.Int32(tidx)
                        while c < total:
                            nc = c % cutlass.Int32(npr)
                            rem = c // cutlass.Int32(npr)
                            m = rem % cutlass.Int32(128)
                            sg = rem // cutlass.Int32(128)
                            out_n = base_ar + sg
                            off = (((ft_ar * cutlass.Int32(128) + m) * cutlass.Int32(num_out_n_tiles)
                                    + out_n) * cutlass.Int32(N_TILE) + nc * cutlass.Int32(8))
                            x, y, z, w = cda.multimem_ld_reduce_8xbf16(csym_mc.iterator + off)
                            cda.multimem_st_4xb32(csym_mc.iterator + off, x, y, z, w)
                            c = c + cutlass.Int32(nthr)
                        cute.arch.sync_threads()
                        if tidx == 0:
                            owner_idx = arg // cutlass.Int32(tp_size)
                            word = (owner_idx // cutlass.Int32(64)).to(cutlass.Uint32)
                            bit = cutlass.Int64(1) << (owner_idx % cutlass.Int32(64)).to(cutlass.Int64)
                            old_done = cute.arch.atomic_or(ar_done_bits.iterator + word, bit,
                                                           sem="acq_rel", scope="gpu")
                            if (old_done & bit) == cutlass.Int64(0):
                                cute.arch.atomic_add(ctrl.iterator + C_AR_DONE, cutlass.Uint32(1),
                                                     sem="release", scope="gpu")
                cute.arch.sync_threads()

        gsync(ctrl, C_GS_EXIT, num_ctas, bidx, tidx)
        if cutlass.const_expr(tp_size > 1):
            nvl_barrier(nvl_local, nvl_mc, 1, tp_size, bidx, tidx)   # all ranks exited
