#!/usr/bin/env python3
"""
Design-structured FA tile microkernel (Hopper SM90, CuTe DSL), warp-specialized
per causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md (FA Mode section).

Single class: FaWsAttnKV — one 128-row FA tile = softmax(Q K^T * scale) V over
Lk keys, structured for the final fused FA+O_proj+AR kernel:

  * 3 warp groups: WG0 = TMA producer (single DMA warp loads K/V), WG1/WG2 =
    WGMMA consumers. atom_layout_mnk=(2,1,1), tiler_mn=(64,N): WG1 owns rows
    0..63, WG2 owns rows 64..127. Online-softmax reductions stay within each
    WG's own 64 rows (warp-quad reduce) — no cross-WG row_max/row_sum/acc_O
    merge (设计稿: "softmax 规约只在各自 owned 的 64 行内完成").
  * Q cooperative-loaded once; K and V on SEPARATE 2-stage TMA pipelines
    (different lifetimes: K freed after QK, V after PV).
  * FA4-style intra-wg overlap: QK(current block) overlaps PV(previous block)
    via two committed WGMMA groups + wait_group(1)/wait_group(0).
  * causal prompt prefill + varlen tail predication via q_start / valid_m / k_len.

Uses cutlass.cute / hopper_helpers / cutlass.pipeline primitives (adapts
oproj_ar_sm90.py's proven pipeline structure; no flash_attn.cute import).
"""
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from quack import layout_utils as qlu

LOG2E = 1.4426950408889634


def cdiv(a, b):
    return (a + b - 1) // b


@cute.jit
def softmax_block(acc_mn: cute.Tensor, row_max: cute.Tensor, row_sum: cute.Tensor,
                  row_scale: cute.Tensor, nrows: cutlass.Constexpr,
                  slog2: cutlass.Constexpr, is_first: cutlass.Constexpr,
                  coord_mn: cute.Tensor, n_block: cutlass.Constexpr,
                  causal: cutlass.Constexpr, q_start: cutlass.Constexpr,
                  k_len: cutlass.Constexpr):
    """One online-softmax step on the QK C-fragment (per-thread mn view).

    Updates row_max/row_sum in place; writes P (=exp) back into acc_mn; fills
    row_scale[r] = correction factor for acc_O (1.0 on the first block). row_sum
    accumulates LOCAL (per-thread) partial sums; the quad reduction is deferred
    to finalize (matches flash_fwd online_softmax).
    """
    for r in cutlass.range_constexpr(nrows):
        if cutlass.const_expr(causal or n_block * 128 + 127 >= k_len):
            coord_row = coord_mn[r, None]
            for c in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                q_pos = q_start + coord_row[c][0]
                kv_pos = n_block * 128 + coord_row[c][1]
                valid = True
                if cutlass.const_expr(causal):
                    valid = valid and (kv_pos <= q_pos)
                if cutlass.const_expr(n_block * 128 + 127 >= k_len):
                    valid = valid and (kv_pos < k_len)
                if not valid:
                    acc_mn[r, c] = -cutlass.Float32.inf
        row = acc_mn[r, None].load()
        m_old = row_max[r]
        if cutlass.const_expr(is_first):
            m = row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
        else:
            m = row.reduce(cute.ReductionOp.MAX, m_old, 0)
        m = cute.arch.warp_reduction_max(m, threads_in_group=4)
        pexp = cute.math.exp2(row * slog2 - m * slog2, fastmath=True)
        ls = pexp.reduce(cute.ReductionOp.ADD, 0.0, 0)
        if cutlass.const_expr(is_first):
            row_scale[r] = cutlass.Float32(1.0)
            row_sum[r] = ls
        else:
            rs = cute.math.exp2((m_old - m) * slog2, fastmath=True)
            row_scale[r] = rs
            row_sum[r] = row_sum[r] * rs + ls
        row_max[r] = m
        acc_mn[r, None].store(pexp)


class FaWsAttnKV:
    """2-WG kv-block attention: O[M,D] = softmax(Q K^T * scale) V over Lk keys.

    step 2b (multi-block, tile_m=64/128). Q loaded once via its own 1-stage
    pipeline; K/V per block via a 2-stage pipeline. Register-resident online
    softmax with cross-block correction (softmax_block). Supports causal prompt
    prefill and varlen tail predication through q_start / valid_m / k_len.
    """

    def __init__(self, M, N, D, Lk, softmax_scale=None, acc_dtype=cutlass.Float32,
                 kv_stages=2, causal=False, q_start=0, valid_m=None, k_len=None):
        self.M = M
        self.N = N          # kv block size
        self.D = D
        self.Lk = Lk
        self.nblk = Lk // N
        self.causal = causal
        self.q_start = q_start
        self.valid_m = valid_m if valid_m is not None else M
        self.k_len = k_len if k_len is not None else Lk
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        self.kv_stages = kv_stages
        self.num_dma_threads = 128
        self.mma_atom_layout_mnk = (2, 1, 1) if M > 64 else (1, 1, 1)
        self.num_mma_threads = 128 * self.mma_atom_layout_mnk[0]
        # Consumer CooperativeGroup size = number of consumer WARPS, not threads.
        # PipelineTmaAsync's empty (consumer->producer) barrier is arrived once per
        # warp (is_signalling_thread = lane 0 of each warp), so its expected arrival
        # count must equal the warp count. Passing num_mma_threads here makes the
        # empty barrier require 8x too many arrivals -> producer_acquire hangs the
        # first time a stage is REUSED (nblk > kv_stages). Matches FA4
        # flash_fwd_sm90.py (mma_warps = num_mma_threads // WARP_SIZE).
        self.num_mma_warps = self.num_mma_threads // 32
        self.threads = self.num_dma_threads + self.num_mma_threads
        self.align = 1024

    def _smem(self, dtype, rows, cols, stages):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, stages), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem(dt, self.M, self.D, 1)
        sK_l = self._smem(dt, self.N, self.D, self.kv_stages)
        sV_l = self._smem(dt, self.N, self.D, self.kv_stages)

        # FA4 Hopper: Q cooperative-loaded once; K and V via SEPARATE TMA
        # pipelines (K released right after QK, V after PV — different lifetimes,
        # must not share one pipeline). 设计文稿 "FA K/V pipeline 与 intra-wg overlap".
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK, cute.slice_(sK_l, (None, None, 0)), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV, cute.slice_(sV_l, (None, None, 0)), (self.N, self.D), num_multicast=1)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk,
            tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk,
            tiler_mn=(64, self.D),
            a_source=warpgroup.OperandSource.RMEM)

        @cute.struct
        class Smem:
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_v: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_l)], self.align]
            sK: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sK_l)], self.align]
            sV: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sV_l)], self.align]

        self.kernel(mQ, tma_k, tK, tma_v, tV, mO, mma_qk, mma_pv,
                    sQ_l, sK_l, sV_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, mO: cute.Tensor,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        slog2 = cutlass.const_expr(self.scale_log2)
        nblk = cutlass.const_expr(self.nblk)
        nthr = cutlass.const_expr(self.threads)
        MD = cutlass.const_expr(self.M * self.D)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v)

        tx_k = cute.size_in_bytes(self.dt, cute.slice_(sK_l, (None, None, 0)))
        tx_v = cute.size_in_bytes(self.dt, cute.slice_(sV_l, (None, None, 0)))

        al = cutlass.utils.SmemAllocator()
        st = al.allocate(Smem)
        sQ = st.sQ.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
        sK = st.sK.get_tensor(sK_l.outer, swizzle=sK_l.inner)
        sV = st.sV.get_tensor(sV_l.outer, swizzle=sV_l.inner)

        # cooperative synchronous Q load (all threads); pipeline create()'s
        # CTA agent_sync below publishes it before any consumer reads sQ.
        for i in cutlass.range_constexpr(cdiv(MD, nthr)):
            idx = tidx + i * nthr
            if idx < MD:
                sQ[idx // self.D, idx % self.D, 0] = mQ[idx // self.D, idx % self.D, 0]

        prodk = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        prodv = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consk = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps)
        consv = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps)
        pl_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_k.data_ptr(), num_stages=self.kv_stages,
            producer_group=prodk, consumer_group=consk, tx_count=tx_k)
        pl_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_v.data_ptr(), num_stages=self.kv_stages,
            producer_group=prodv, consumer_group=consv, tx_count=tx_v)

        gK = cute.local_tile(mK, (self.N, self.D), (None, None, None))
        gV = cute.local_tile(mV, (self.N, self.D), (None, None, None))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        # ---- WG0: producer (separate K and V pipelines, lockstep states) ----
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                kp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
                vp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
                for j in cutlass.range_constexpr(nblk):
                    pl_k.producer_acquire(kp)
                    cute.copy(tma_k, tKgK[(None, j, 0, 0)], tKsK[(None, kp.index)],
                              tma_bar_ptr=pl_k.producer_get_barrier(kp))
                    pl_k.producer_commit(kp)
                    pl_v.producer_acquire(vp)
                    cute.copy(tma_v, tVgV[(None, j, 0, 0)], tVsV[(None, vp.index)],
                              tma_bar_ptr=pl_v.producer_get_barrier(vp))
                    pl_v.producer_commit(vp)
                    kp.advance()
                    vp.advance()

        # ---- WG1/WG2: consumers (both warpgroups run this block; the (2,1,1)
        #      tiled MMA splits M so WG1 owns rows 0..63, WG2 owns rows 64..127,
        #      each with its own softmax/acc_O — no cross-WG merge). Structure:
        #      Step A = QK-only prologue (block 0); middle loop = FA4 intra-wg
        #      overlap of QK(current) with PV(previous); Step E = PV-only epilogue.
        if wg_idx >= 1:
            cute.arch.setmaxregister_increase(232)
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
            gO = cute.local_tile(mO, (self.M, self.D), (None, None, None))
            tCgO = thr_pv.partition_C(gO[(None, None, 0, 0, 0)])
            nkb_qk = cutlass.const_expr(cute.size(tCrQ, mode=[2]))

            acc_mn = qlu.reshape_acc_to_mn(acc_S)
            nrows = cutlass.const_expr(cute.size(acc_mn, mode=[0]))
            row_max = cute.make_rmem_tensor(nrows, self.acc_dtype)
            row_sum = cute.make_rmem_tensor(nrows, self.acc_dtype)
            row_scale = cute.make_rmem_tensor(nrows, self.acc_dtype)
            acc_O.fill(0.0)
            acc_O_mn = qlu.reshape_acc_to_mn(acc_O)

            sr = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
            tOrP_v = qlu.reshape_acc_to_frgA(acc_S)
            tOrP = cute.make_rmem_tensor_like(tOrP_v, self.dt)
            nkb_pv = cutlass.const_expr(cute.size(tOrP, mode=[2]))

            # ---- Step A: first block — QK + softmax only; tOrP=P(0), acc_O=0 ----
            pl_k.consumer_wait(sr)
            acc_S.fill(0.0)
            mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(nkb_qk):
                cute.gemm(mma_qk, acc_S, tCrQ[(None, None, kb, 0)],
                          tCrK[(None, None, kb, sr.index)], acc_S)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            pl_k.consumer_release(sr)
            softmax_block(acc_mn, row_max, row_sum, row_scale, nrows, slog2, True,
                          coord_mn, 0, self.causal, self.q_start, self.k_len)
            tOrP.store(tOrP_v.load().to(self.dt))

            # ---- Middle: current=j overlaps PV(previous=j-1) ----
            for j in cutlass.range_constexpr(1, nblk):
                srv = sr.clone()        # V uses previous block's stage
                sr.advance()            # K advances to current block
                pl_k.consumer_wait(sr)
                acc_S.fill(0.0)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(nkb_qk):
                    cute.gemm(mma_qk, acc_S, tCrQ[(None, None, kb, 0)],
                              tCrK[(None, None, kb, sr.index)], acc_S)
                cute.nvgpu.warpgroup.commit_group()
                pl_v.consumer_wait(srv)
                mma_pv.set(warpgroup.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkb_pv):
                    cute.gemm(mma_pv, acc_O, tOrP[(None, None, kb)],
                              tCrV[(None, None, kb, srv.index)], acc_O)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(1)   # QK(current) done; PV(prev) flying
                pl_k.consumer_release(sr)
                softmax_block(acc_mn, row_max, row_sum, row_scale, nrows, slog2, False,
                              coord_mn, j, self.causal, self.q_start, self.k_len)
                cute.nvgpu.warpgroup.wait_group(0)   # PV(prev) done
                pl_v.consumer_release(srv)
                for r in cutlass.range_constexpr(nrows):
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                tOrP.store(qlu.reshape_acc_to_frgA(acc_S).load().to(self.dt))  # P(current)

            # ---- Step E: final PV (previous = last block) ----
            srv = sr.clone()
            pl_v.consumer_wait(srv)
            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(nkb_pv):
                cute.gemm(mma_pv, acc_O, tOrP[(None, None, kb)],
                          tCrV[(None, None, kb, srv.index)], acc_O)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            pl_v.consumer_release(srv)

            # finalize: quad-reduce row_sum, divide
            for r in cutlass.range_constexpr(nrows):
                if coord_mn[r, 0][0] < self.valid_m:
                    s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                    inv = cutlass.Float32(1.0) / s
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
                else:
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * cutlass.Float32(0.0))
            tCgO.store(acc_O.load().to(mO.element_type))
