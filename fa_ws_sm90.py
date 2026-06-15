#!/usr/bin/env python3
"""
Design-structured FA (Hopper SM90, CuTe DSL) — warp-specialized, built per the
revised 设计文稿.md (3-WG: WG0 TMA producer + WG1/WG2 WGMMA consumers, TMA
2-stage pipeline, register-resident online softmax). Built in increments:

  step 1 (this commit): 2-WG QK tile via PipelineTmaAsync.
      WG0 = DMA producer (TMA load Q,K), WG1 = MMA consumer (QK WGMMA) -> S.
      De-risks the proven TMA-pipeline path (raw mbarrier TMA hung; see
      [cute-dsl-scheduler-gotchas]) + warp specialization + WGMMA.
  step 2: + register online softmax + P@V + kv-loop (1 consumer WG, tile_m=64).
  step 3: tile_m=128 with WG1/WG2 each 64 rows + causal + varlen.

Clean rewrite using cutlass.cute / hopper_helpers / cutlass.pipeline primitives
(adapts oproj_ar_sm90.py's proven pipeline structure; no flash_attn.cute import).
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


class FaWsQK:
    """2-WG single-tile S = Q[M,D] @ K[N,D]^T via TMA pipeline + WGMMA."""

    def __init__(self, M, N, D, stages=2, acc_dtype=cutlass.Float32):
        self.M = M
        self.N = N
        self.D = D
        self.stages = stages
        self.acc_dtype = acc_dtype
        self.tile_shape_mnk = (M, N, D)
        self.num_dma_threads = 128
        self.num_mma_threads = 128
        self.threads = self.num_dma_threads + self.num_mma_threads
        self.align = 1024

    def _smem_staged(self, dtype, rows, cols, stages):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, stages), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mS: cute.Tensor,
                 stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem_staged(dt, self.M, self.D, self.stages)   # A
        sK_l = self._smem_staged(dt, self.N, self.D, self.stages)   # B

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_q, tQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mQ, cute.slice_(sQ_l, (None, None, 0)), (self.M, self.D), num_multicast=1)
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK, cute.slice_(sK_l, (None, None, 0)), (self.N, self.D), num_multicast=1)

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.N))

        @cute.struct
        class Smem:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_l)], self.align]
            sK: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sK_l)], self.align]

        self.kernel(tma_q, tQ, tma_k, tK, mS, tiled_mma, sQ_l, sK_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, tma_q: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, mS: cute.Tensor,
               tiled_mma: cute.TiledMma, sQ_l: cute.ComposedLayout,
               sK_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)

        sQ_smem = cute.slice_(sQ_l, (None, None, 0))
        sK_smem = cute.slice_(sK_l, (None, None, 0))
        tx_bytes = (cute.size_in_bytes(self.dt, sQ_smem)
                    + cute.size_in_bytes(self.dt, sK_smem))

        al = cutlass.utils.SmemAllocator()
        storage = al.allocate(Smem)
        mbar_ptr = storage.mbar.data_ptr()
        sQ = storage.sQ.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
        sK = storage.sK.get_tensor(sK_l.outer, swizzle=sK_l.inner)

        prod_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_grp = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
        # Single non-cluster CTA: defer_sync=False lets create() do the
        # mbarrier_init_fence + CTA-wide agent_sync init internally.
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr, num_stages=self.stages,
            producer_group=prod_grp, consumer_group=cons_grp,
            tx_count=tx_bytes)

        # gmem tiles (single tile, L=1) + TMA partition
        gQ = cute.local_tile(mQ, (self.M, self.D), (None, None, None))
        gK = cute.local_tile(mK, (self.N, self.D), (None, None, None))
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2))

        k_tile_cnt = 1

        # ---- WG0: DMA producer (warp 0 issues TMA) ----
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,
                                                    self.stages)
                for _ in range(k_tile_cnt):
                    mainloop.producer_acquire(prod)
                    bar = mainloop.producer_get_barrier(prod)
                    cute.copy(tma_q, tQgQ[(None, 0, 0, 0)], tQsQ[(None, prod.index)],
                              tma_bar_ptr=bar)
                    cute.copy(tma_k, tKgK[(None, 0, 0, 0)], tKsK[(None, prod.index)],
                              tma_bar_ptr=bar)
                    mainloop.producer_commit(prod)
                    prod.advance()
                # producer_tail omitted for single-tile step-1 (k_tile_cnt < stages);
                # used properly in step-2 kv-loop.

        # ---- WG1: MMA consumer ----
        if wg_idx >= 1:
            cute.arch.setmaxregister_increase(232)
            thr_mma = tiled_mma.get_slice(tidx - self.num_dma_threads)
            tCrQ = tiled_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tCrK = tiled_mma.make_fragment_B(thr_mma.partition_B(sK))
            gS = cute.local_tile(mS, (self.M, self.N), (None, None, None))
            tCgS = thr_mma.partition_C(gS[(None, None, 0, 0, 0)])
            acc = cute.make_rmem_tensor(tCgS.shape[:3], self.acc_dtype)
            num_kb = cute.size(tCrQ, mode=[2])

            read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer,
                                                self.stages)
            rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer,
                                               self.stages)
            acc.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.fence()
            for _ in range(k_tile_cnt):
                mainloop.consumer_wait(read)
                for kb in cutlass.range_constexpr(num_kb):
                    cute.gemm(tiled_mma, acc, tCrQ[(None, None, kb, read.index)],
                              tCrK[(None, None, kb, read.index)], acc)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                mainloop.consumer_release(rel)
                read.advance()
                rel.advance()
            tCgS.store(acc.load())


class FaWsAttn:
    """2-WG single-kv-block attention: O[M,D] = softmax(Q K^T * scale) V.

    step 2 (non-causal, single block Lk=N, tile_m=64). Design-structured:
    WG0 TMA-loads Q,K,V; WG1 consumer does QK WGMMA -> register online softmax
    (reshape_acc_to_mn + warp_reduction over the quad) -> P kept in registers as
    the PV A-operand (reshape_acc_to_frgA, a_source=RMEM) -> PV WGMMA (V as [N,D]
    MN-major, no transpose) -> rescale by 1/row_sum -> store O.
    """

    def __init__(self, M, N, D, softmax_scale=None, acc_dtype=cutlass.Float32):
        self.M = M
        self.N = N      # kv block size == Lk for step 2
        self.D = D
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        self.stages = 1
        self.num_dma_threads = 128
        self.num_mma_threads = 128
        self.threads = 256
        self.align = 1024

    def _smem(self, dtype, rows, cols):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, 1), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem(dt, self.M, self.D)
        sK_l = self._smem(dt, self.N, self.D)
        sV_l = self._smem(dt, self.N, self.D)

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_q, tQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mQ, cute.slice_(sQ_l, (None, None, 0)), (self.M, self.D), num_multicast=1)
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK, cute.slice_(sK_l, (None, None, 0)), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV, cute.slice_(sV_l, (None, None, 0)), (self.N, self.D), num_multicast=1)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.D),
            a_source=warpgroup.OperandSource.RMEM)

        @cute.struct
        class Smem:
            mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_l)], self.align]
            sK: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sK_l)], self.align]
            sV: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sV_l)], self.align]

        self.kernel(tma_q, tQ, tma_k, tK, tma_v, tV, mO, mma_qk, mma_pv,
                    sQ_l, sK_l, sV_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, tma_q: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, mO: cute.Tensor,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        slog2 = cutlass.const_expr(self.scale_log2)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v)

        tx = (cute.size_in_bytes(self.dt, cute.slice_(sQ_l, (None, None, 0)))
              + cute.size_in_bytes(self.dt, cute.slice_(sK_l, (None, None, 0)))
              + cute.size_in_bytes(self.dt, cute.slice_(sV_l, (None, None, 0))))

        al = cutlass.utils.SmemAllocator()
        st = al.allocate(Smem)
        sQ = st.sQ.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
        sK = st.sK.get_tensor(sK_l.outer, swizzle=sK_l.inner)
        sV = st.sV.get_tensor(sV_l.outer, swizzle=sV_l.inner)

        prod = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
        ml = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar.data_ptr(), num_stages=self.stages,
            producer_group=prod, consumer_group=cons, tx_count=tx)

        gQ = cute.local_tile(mQ, (self.M, self.D), (None, None, None))
        gK = cute.local_tile(mK, (self.N, self.D), (None, None, None))
        gV = cute.local_tile(mV, (self.N, self.D), (None, None, None))
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        # ---- WG0: producer ----
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                p = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
                ml.producer_acquire(p)
                bar = ml.producer_get_barrier(p)
                cute.copy(tma_q, tQgQ[(None, 0, 0, 0)], tQsQ[(None, p.index)], tma_bar_ptr=bar)
                cute.copy(tma_k, tKgK[(None, 0, 0, 0)], tKsK[(None, p.index)], tma_bar_ptr=bar)
                cute.copy(tma_v, tVgV[(None, 0, 0, 0)], tVsV[(None, p.index)], tma_bar_ptr=bar)
                ml.producer_commit(p)

        # ---- WG1: consumer ----
        if wg_idx >= 1:
            cute.arch.setmaxregister_increase(232)
            lane = tidx - self.num_dma_threads
            thr_qk = mma_qk.get_slice(lane)
            thr_pv = mma_pv.get_slice(lane)
            tCrQ = mma_qk.make_fragment_A(thr_qk.partition_A(sQ))
            tCrK = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
            # PV B operand: V stays [N,D] in smem; transpose_view gives the [D,N]
            # MN-major B view (no data movement).
            sVt = qlu.transpose_view(sV)
            tCrV = mma_pv.make_fragment_B(thr_pv.partition_B(sVt))
            idS = cute.make_identity_tensor((self.M, self.N))
            idO = cute.make_identity_tensor((self.M, self.D))
            acc_S = cute.make_rmem_tensor(thr_qk.partition_C(idS).shape[:3], self.acc_dtype)
            acc_O = cute.make_rmem_tensor(thr_pv.partition_C(idO).shape[:3], self.acc_dtype)
            gO = cute.local_tile(mO, (self.M, self.D), (None, None, None))
            tCgO = thr_pv.partition_C(gO[(None, None, 0, 0, 0)])

            read = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            rel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
            ml.consumer_wait(read)

            # QK WGMMA
            acc_S.fill(0.0)
            mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
                cute.gemm(mma_qk, acc_S, tCrQ[(None, None, kb, read.index)],
                          tCrK[(None, None, kb, read.index)], acc_S)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            # register online softmax (single block -> is_first, no correction)
            acc_mn = qlu.reshape_acc_to_mn(acc_S)
            nrows = cutlass.const_expr(cute.size(acc_mn, mode=[0]))
            row_sum = cute.make_rmem_tensor(nrows, self.acc_dtype)
            for r in cutlass.range_constexpr(nrows):
                row = acc_mn[r, None].load()
                m = row.reduce(cute.ReductionOp.MAX, -self.acc_dtype.inf, 0)
                m = cute.arch.warp_reduction_max(m, threads_in_group=4)
                pexp = cute.math.exp2(row * slog2 - m * slog2, fastmath=True)
                s = pexp.reduce(cute.ReductionOp.ADD, 0.0, 0)
                s = cute.arch.warp_reduction_sum(s, threads_in_group=4)
                row_sum[r] = s
                acc_mn[r, None].store(pexp)

            # P (registers) -> bf16 A-operand fragment
            tOrP_v = qlu.reshape_acc_to_frgA(acc_S)
            tOrP = cute.make_rmem_tensor_like(tOrP_v, self.dt)
            tOrP.store(tOrP_v.load().to(self.dt))

            # PV WGMMA (A from RMEM, V as [N,D] MN-major)
            acc_O.fill(0.0)
            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
                cute.gemm(mma_pv, acc_O, tOrP[(None, None, kb)],
                          tCrV[(None, None, kb, read.index)], acc_O)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            ml.consumer_release(rel)

            # rescale by 1/row_sum and store
            acc_O_mn = qlu.reshape_acc_to_mn(acc_O)
            for r in cutlass.range_constexpr(nrows):
                inv = cutlass.Float32(1.0) / row_sum[r]
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
            tCgO.store(acc_O.load().to(mO.element_type))


@cute.jit
def softmax_block(acc_mn: cute.Tensor, row_max: cute.Tensor, row_sum: cute.Tensor,
                  row_scale: cute.Tensor, nrows: cutlass.Constexpr,
                  slog2: cutlass.Constexpr, is_first: cutlass.Constexpr):
    """One online-softmax step on the QK C-fragment (per-thread mn view).

    Updates row_max/row_sum in place; writes P (=exp) back into acc_mn; fills
    row_scale[r] = correction factor for acc_O (1.0 on the first block). row_sum
    accumulates LOCAL (per-thread) partial sums; the quad reduction is deferred
    to finalize (matches flash_fwd online_softmax).
    """
    for r in cutlass.range_constexpr(nrows):
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

    step 2b (non-causal, multi-block, tile_m=64). Q loaded once via its own
    1-stage pipeline; K/V per block via a 2-stage pipeline. Register-resident
    online softmax with cross-block correction (softmax_block). Validated vs torch
    / the FaTileSm90 oracle.
    """

    def __init__(self, M, N, D, Lk, softmax_scale=None, acc_dtype=cutlass.Float32,
                 kv_stages=2):
        self.M = M
        self.N = N          # kv block size
        self.D = D
        self.Lk = Lk
        self.nblk = Lk // N
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        self.kv_stages = kv_stages
        self.num_dma_threads = 128
        self.num_mma_threads = 128
        self.threads = 256
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

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_q, tQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mQ, cute.slice_(sQ_l, (None, None, 0)), (self.M, self.D), num_multicast=1)
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK, cute.slice_(sK_l, (None, None, 0)), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV, cute.slice_(sV_l, (None, None, 0)), (self.N, self.D), num_multicast=1)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.D),
            a_source=warpgroup.OperandSource.RMEM)

        @cute.struct
        class Smem:
            mbar_q: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_kv: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_l)], self.align]
            sK: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sK_l)], self.align]
            sV: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sV_l)], self.align]

        self.kernel(tma_q, tQ, tma_k, tK, tma_v, tV, mO, mma_qk, mma_pv,
                    sQ_l, sK_l, sV_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, tma_q: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, mO: cute.Tensor,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        slog2 = cutlass.const_expr(self.scale_log2)
        nblk = cutlass.const_expr(self.nblk)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v)

        tx_q = cute.size_in_bytes(self.dt, cute.slice_(sQ_l, (None, None, 0)))
        tx_kv = (cute.size_in_bytes(self.dt, cute.slice_(sK_l, (None, None, 0)))
                 + cute.size_in_bytes(self.dt, cute.slice_(sV_l, (None, None, 0))))

        al = cutlass.utils.SmemAllocator()
        st = al.allocate(Smem)
        sQ = st.sQ.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
        sK = st.sK.get_tensor(sK_l.outer, swizzle=sK_l.inner)
        sV = st.sV.get_tensor(sV_l.outer, swizzle=sV_l.inner)

        prod = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        prod2 = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons_q = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
        cons_kv = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
        pl_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_q.data_ptr(), num_stages=1,
            producer_group=prod, consumer_group=cons_q, tx_count=tx_q)
        pl_kv = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar_kv.data_ptr(), num_stages=self.kv_stages,
            producer_group=prod2, consumer_group=cons_kv, tx_count=tx_kv)

        gQ = cute.local_tile(mQ, (self.M, self.D), (None, None, None))
        gK = cute.local_tile(mK, (self.N, self.D), (None, None, None))
        gV = cute.local_tile(mV, (self.N, self.D), (None, None, None))
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        # ---- WG0: producer ----
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                qp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                pl_q.producer_acquire(qp)
                cute.copy(tma_q, tQgQ[(None, 0, 0, 0)], tQsQ[(None, 0)],
                          tma_bar_ptr=pl_q.producer_get_barrier(qp))
                pl_q.producer_commit(qp)
                kp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,
                                                  self.kv_stages)
                for j in cutlass.range_constexpr(nblk):
                    pl_kv.producer_acquire(kp)
                    bar = pl_kv.producer_get_barrier(kp)
                    cute.copy(tma_k, tKgK[(None, j, 0, 0)], tKsK[(None, kp.index)], tma_bar_ptr=bar)
                    cute.copy(tma_v, tVgV[(None, j, 0, 0)], tVsV[(None, kp.index)], tma_bar_ptr=bar)
                    pl_kv.producer_commit(kp)
                    kp.advance()
                # NOTE: multi-stage reuse (nblk > kv_stages) + producer_tail drain
                # still WIP — needs oproj-style delayed-release (software-pipelined)
                # consumer. Online-correction math validated at nblk <= kv_stages.

        # ---- WG1: consumer ----
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

            qc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            pl_q.consumer_wait(qc)
            kr = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)
            krel = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stages)

            for j in cutlass.range_constexpr(nblk):
                is_first = cutlass.const_expr(j == 0)
                pl_kv.consumer_wait(kr)
                # QK
                acc_S.fill(0.0)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(nkb_qk):
                    cute.gemm(mma_qk, acc_S, tCrQ[(None, None, kb, 0)],
                              tCrK[(None, None, kb, kr.index)], acc_S)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                # online softmax (correction)
                softmax_block(acc_mn, row_max, row_sum, row_scale, nrows, slog2, is_first)
                # rescale acc_O by correction (no-op on first block, row_scale=1)
                if cutlass.const_expr(not is_first):
                    for r in cutlass.range_constexpr(nrows):
                        acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                # P -> bf16 RMEM A-operand
                tOrP_v = qlu.reshape_acc_to_frgA(acc_S)
                tOrP = cute.make_rmem_tensor_like(tOrP_v, self.dt)
                tOrP.store(tOrP_v.load().to(self.dt))
                # PV accumulate
                mma_pv.set(warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
                    cute.gemm(mma_pv, acc_O, tOrP[(None, None, kb)],
                              tCrV[(None, None, kb, kr.index)], acc_O)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                pl_kv.consumer_release(krel)
                kr.advance()
                krel.advance()

            # finalize: quad-reduce row_sum, divide
            for r in cutlass.range_constexpr(nrows):
                s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                inv = cutlass.Float32(1.0) / s
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
            tCgO.store(acc_O.load().to(mO.element_type))
