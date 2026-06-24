#!/usr/bin/env python3
"""
Intermediate dynamic-varlen FA tile kernels for Hopper SM90 CuTe DSL.

This file is not the core fused kernel. It de-risks the FA payload shape used by
fused_fa_oproj_ar.py: one compiled kernel must handle row tiles with runtime
q_start, valid_m, k_len, and nblk. The KV block loop therefore uses cutlass.range
instead of range_constexpr, and causal/tail predicates are runtime values.

Compile-time (kernel variant): M=128, N=D=128, kv_stages=2, causal prompt prefill.
Runtime (per task, from `params` int32 tensor): [q_start, valid_m, k_len, nblk].

FaWsAttnDyn uses a compact single-head addressing model, launching one FA task per
kernel for local validation of the payload mechanics used by the fused kernel.

Agent note: keep persistent scheduling, O_scratch -> O_proj readiness, and AR
owner protocol comments in fused_fa_oproj_ar.py. Comments here should describe
only the standalone FA payload mechanics.
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
def softmax_block_dyn(acc_mn: cute.Tensor, row_max: cute.Tensor, row_sum: cute.Tensor,
                      row_scale: cute.Tensor, nrows: cutlass.Constexpr,
                      slog2: cutlass.Constexpr, is_first: cutlass.Constexpr,
                      coord_mn: cute.Tensor, n_block, q_start, k_len,
                      need_mask: cutlass.Constexpr = True, offset=0):
    """Online-softmax step with runtime causal and k_len masking.

    is_first remains compile-time for the prologue/middle split. n_block, q_start,
    k_len, and offset are runtime. bottom-right aligned causal (offset = k_len -
    q_len): a score is valid only when kv_pos <= q_pos + offset and kv_pos < k_len.
    offset == 0 退化为完整 prompt prefill 的标准下三角。row_sum stores local
    per-thread partial sums; the warp-quad reduction is deferred to finalization.

    need_mask is compile-time. 右->左遍历分三段：peeled 首块与 causal 中间区间传
    need_mask=True；causal 边界左侧全可见块传 need_mask=False 跳过逐元素比较。
    """
    for r in cutlass.range_constexpr(nrows):
        coord_row = coord_mn[r, None]
        if cutlass.const_expr(need_mask):
            for c in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
                q_pos = q_start + coord_row[c][0]
                kv_pos = n_block * 128 + coord_row[c][1]
                if (kv_pos > q_pos + offset) or (kv_pos >= k_len):
                    acc_mn[r, c] = -cutlass.Float32.inf
        row = acc_mn[r, None].load()
        m_old = row_max[r]
        if cutlass.const_expr(is_first):
            m = row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
        else:
            m = row.reduce(cute.ReductionOp.MAX, m_old, 0)
        m = cute.arch.warp_reduction_max(m, threads_in_group=4)
        # bottom-right causal (offset>0) 下, peeled 首块或某些 causal 中间块可能对低 q_pos
        # 行整行落在 causal reach 之外 -> 整行 -inf, m=-inf。此时 exp2(-inf-(-inf))=NaN。
        # q==k 时对角元素恒可见, m 不会为 -inf, 本分支不触发, 故对回归路径零影响。
        m_safe = m
        if m == -cutlass.Float32.inf:
            m_safe = cutlass.Float32(0.0)
        pexp = cute.math.exp2(row * slog2 - m_safe * slog2, fastmath=True)
        ls = pexp.reduce(cute.ReductionOp.ADD, 0.0, 0)
        if cutlass.const_expr(is_first):
            row_scale[r] = cutlass.Float32(1.0)
            row_sum[r] = ls
        else:
            rs = cutlass.Float32(1.0)            # 默认: m==-inf, acc_O 仍 0, 无需 rescale
            if m != -cutlass.Float32.inf:
                rs = cute.math.exp2((m_old - m) * slog2, fastmath=True)
            row_scale[r] = rs
            row_sum[r] = row_sum[r] * rs + ls
        row_max[r] = m
        acc_mn[r, None].store(pexp)


class FaWsAttnDyn:
    """Runtime-varlen FA payload prototype with compact single-head addressing.

    q_start, valid_m, k_len, and nblk are read from `params` at runtime so one
    compiled kernel can serve multiple tile shapes. Lk_max is only the allocation
    envelope for this standalone validation path.
    """

    def __init__(self, M, N, D, Lk_max, softmax_scale=None, acc_dtype=cutlass.Float32,
                 kv_stages=2):
        self.M = M
        self.N = N
        self.D = D
        self.Lk_max = Lk_max
        self.nblk_max = Lk_max // N
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        self.kv_stages = kv_stages
        self.num_dma_threads = 128
        self.mma_atom_layout_mnk = (2, 1, 1) if M > 64 else (1, 1, 1)
        self.num_mma_threads = 128 * self.mma_atom_layout_mnk[0]
        self.num_mma_warps = self.num_mma_threads // 32   # consumer group = #warps
        self.threads = self.num_dma_threads + self.num_mma_threads
        self.align = 1024

    def _smem(self, dtype, rows, cols, stages):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, stages), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, mParams: cute.Tensor, stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem(dt, self.M, self.D, 1)
        sK_l = self._smem(dt, self.N, self.D, self.kv_stages)
        sV_l = self._smem(dt, self.N, self.D, self.kv_stages)

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK, cute.slice_(sK_l, (None, None, 0)), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV, cute.slice_(sV_l, (None, None, 0)), (self.N, self.D), num_multicast=1)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk, tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk, tiler_mn=(64, self.D),
            a_source=warpgroup.OperandSource.RMEM)

        @cute.struct
        class Smem:
            mbar_k: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            mbar_v: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            sQ: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sQ_l)], self.align]
            sK: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sK_l)], self.align]
            sV: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sV_l)], self.align]

        self.kernel(mQ, tma_k, tK, tma_v, tV, mO, mParams, mma_qk, mma_pv,
                    sQ_l, sK_l, sV_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, mO: cute.Tensor,
               mParams: cute.Tensor, mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        slog2 = cutlass.const_expr(self.scale_log2)
        nthr = cutlass.const_expr(self.threads)
        MD = cutlass.const_expr(self.M * self.D)

        # Runtime task descriptor for this standalone FA task.
        q_start = mParams[0]
        valid_m = mParams[1]
        k_len = mParams[2]
        nblk = mParams[3]

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

        # WG0 producer: dynamic nblk K/V stream.
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                kp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
                vp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
                for j in cutlass.range(nblk, unroll=1):
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

        # WG1/WG2 consumers: dynamic nblk with QK(current)/PV(previous) overlap.
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

            # First block initializes the online-softmax state.
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
            softmax_block_dyn(acc_mn, row_max, row_sum, row_scale, nrows, slog2,
                              True, coord_mn, cutlass.Int32(0), q_start, k_len)
            tOrP.store(tOrP_v.load().to(self.dt))

            # Middle blocks overlap QK(current) with PV(previous).
            for j in cutlass.range(1, nblk, unroll=1):
                srv = sr.clone()
                sr.advance()
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
                cute.nvgpu.warpgroup.wait_group(1)
                pl_k.consumer_release(sr)
                softmax_block_dyn(acc_mn, row_max, row_sum, row_scale, nrows, slog2,
                                  False, coord_mn, j, q_start, k_len)
                cute.nvgpu.warpgroup.wait_group(0)
                pl_v.consumer_release(srv)
                for r in cutlass.range_constexpr(nrows):
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                tOrP.store(qlu.reshape_acc_to_frgA(acc_S).load().to(self.dt))

            # Final PV consumes the last softmax block.
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

            # Finalize: warp-quad reduce row_sum, divide, and mask invalid rows.
            for r in cutlass.range_constexpr(nrows):
                # Hazard: warp_reduction_sum is a warp-collective shuffle. Call it
                # unconditionally; guarding it with valid_m can diverge a partial row
                # tile and hang the CTA.
                s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                if coord_mn[r, 0][0] < valid_m:
                    inv = cutlass.Float32(1.0) / s
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
                else:
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * cutlass.Float32(0.0))
            tCgO.store(acc_O.load().to(mO.element_type))
