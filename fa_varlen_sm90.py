#!/usr/bin/env python3
"""
Dynamic (runtime-varlen) FA tile kernel (Hopper SM90, CuTe DSL).

This is the FA payload for the FUSED persistent kernel: ONE compiled kernel must
handle tiles of DIFFERENT runtime shape (causal_varlen_prefill..._plan_zh.md
"Runtime task descriptor 与动态 varlen payload"). vs the compile-time FaWsAttnKV
microkernel (Step 3), here q_start / valid_m / k_len / nblk are RUNTIME values
read in-kernel from a params tensor; the KV-block loop uses cutlass.range (NOT
range_constexpr); the causal + k_len masks are fully runtime.

Compile-time (kernel variant): M=128, N=D=128, kv_stages=2, causal prompt prefill.
Runtime (per task, from `params` int32 tensor): [q_start, valid_m, k_len, nblk].

Addressing for THIS standalone de-risk step matches FaWsAttnKV's validated test:
contiguous per-(tile,head) Q[M,D] / K,V[Lk,D] for one head; q_start is the tile's
global Q-row offset (for causal q_pos), k positions are 0..k_len-1. The packed
multi-sequence domain_offset addressing is added when fusing into the dispatch.

Structure mirrors FaWsAttnKV exactly (3WG, (2,1,1) MMA, separate K/V 2-stage
PipelineTmaAsync with consumer group=#warps, intra-wg QK(cur)/PV(prev) overlap),
only the loop bounds + mask predicates become runtime.
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
                      coord_mn: cute.Tensor, n_block, q_start, k_len):
    """Online-softmax step with RUNTIME causal + k_len masking (q_len==k_len prompt).

    is_first is COMPILE-TIME (Step A vs middle); n_block / q_start / k_len are
    RUNTIME. Every (row, col): valid iff kv_pos <= q_pos AND kv_pos < k_len.
    row_sum keeps LOCAL per-thread partial sums; quad reduction deferred to finalize.
    """
    for r in cutlass.range_constexpr(nrows):
        coord_row = coord_mn[r, None]
        for c in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
            q_pos = q_start + coord_row[c][0]
            kv_pos = n_block * 128 + coord_row[c][1]
            if (kv_pos > q_pos) or (kv_pos >= k_len):
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


class FaWsAttnDyn:
    """Runtime-varlen FA tile. Same warp-spec/pipeline as FaWsAttnKV; q_start /
    valid_m / k_len / nblk are read from `params` at runtime so ONE compiled
    kernel serves all shapes. Lk is the max allocated KV length (compile-time)."""

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

        # ---- runtime task descriptor ----
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

        # ---- WG0: producer (dynamic nblk) ----
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

        # ---- WG1/WG2: consumers (dynamic nblk; intra-wg QK(cur)/PV(prev) overlap) ----
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

            # Step A: block 0 — QK + softmax only
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

            # Middle: current=j overlaps PV(previous=j-1), j = 1..nblk-1 (runtime)
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

            # Step E: final PV (previous = last processed block)
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

            # finalize: quad-reduce row_sum, divide; mask invalid rows
            for r in cutlass.range_constexpr(nrows):
                if coord_mn[r, 0][0] < valid_m:
                    s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                    inv = cutlass.Float32(1.0) / s
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
                else:
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * cutlass.Float32(0.0))
            tCgO.store(acc_O.load().to(mO.element_type))


class FaWsAttnPacked:
    """Fused-kernel FA payload, standalone-testable. Reads packed varlen Q/K/V
    [tot, H, D] at RUNTIME (fa_row_tile, head) decoded from cu_seqlens + fa_row_desc,
    runs dynamic causal FA, writes O_scratch[fa_row_tile, :, head, :].

    Same 3WG / (2,1,1) / separate-K/V-pipeline / intra-wg-overlap structure as
    FaWsAttnDyn; adds FA4-style packed-varlen TMA addressing (atom over the full
    packed tensor; in-kernel cute.domain_offset by cu_seqlens[batch] + head slice).
    Grid [1,1,1], one task per launch; ONE compiled kernel serves all tasks
    (task id read from mTask[0] at runtime).
    """

    def __init__(self, M, N, D, H_local, softmax_scale=None,
                 acc_dtype=cutlass.Float32, kv_stages=2):
        self.M = M
        self.N = N
        self.D = D
        self.H_local = H_local
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        self.kv_stages = kv_stages
        self.num_dma_threads = 128
        self.mma_atom_layout_mnk = (2, 1, 1) if M > 64 else (1, 1, 1)
        self.num_mma_threads = 128 * self.mma_atom_layout_mnk[0]
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
                 mOscr: cute.Tensor, mCuQ: cute.Tensor, mCuK: cute.Tensor,
                 mFaB: cute.Tensor, mFaMb: cute.Tensor, mTask: cute.Tensor,
                 stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._smem(dt, self.M, self.D, 1)
        sK_l = self._smem(dt, self.N, self.D, self.kv_stages)
        sV_l = self._smem(dt, self.N, self.D, self.kv_stages)

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        # Permute the packed K/V gmem view [tot, H, D] -> [tot, D, H] so the rank-2
        # TMA tiler (N, D) maps positionally onto (token, D) instead of (token, head)
        # — cta_v_map = composition(identity(gmem.shape), tiler) tiles by mode order,
        # and with head at mode 1 the box's D-extent would collapse to H. Head becomes
        # mode 2, selected at runtime in-kernel.
        mK_t = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[0, 2, 1]))
        mV_t = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[0, 2, 1]))
        tma_k, tK = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mK_t, cute.slice_(sK_l, (None, None, 0)), (self.N, self.D), num_multicast=1)
        tma_v, tV = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mV_t, cute.slice_(sV_l, (None, None, 0)), (self.N, self.D), num_multicast=1)

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

        self.kernel(mQ, tma_k, mK_t, tma_v, mV_t, mOscr, mCuQ, mCuK, mFaB, mFaMb, mTask,
                    mma_qk, mma_pv, sQ_l, sK_l, sV_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, tma_k: cute.CopyAtom, mK: cute.Tensor,
               tma_v: cute.CopyAtom, mV: cute.Tensor, mOscr: cute.Tensor,
               mCuQ: cute.Tensor, mCuK: cute.Tensor, mFaB: cute.Tensor,
               mFaMb: cute.Tensor, mTask: cute.Tensor,
               mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sV_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        slog2 = cutlass.const_expr(self.scale_log2)
        H_local = cutlass.const_expr(self.H_local)
        Dc = cutlass.const_expr(self.D)
        nthr = cutlass.const_expr(self.threads)
        MD = cutlass.const_expr(self.M * self.D)

        # ---- runtime task descriptor (each WG decodes from mTask) ----
        tid = mTask[0]
        ft = tid // cutlass.Int32(H_local)
        head = tid % cutlass.Int32(H_local)
        b = mFaB[ft]
        mb = mFaMb[ft]
        q_start = mCuQ[b]
        k_start = mCuK[b]
        q_len = mCuQ[b + cutlass.Int32(1)] - q_start
        k_len = q_len                                   # complete prompt prefill
        q_tile_pk = q_start + mb * cutlass.Int32(128)   # packed Q-row offset
        mask_q_off = mb * cutlass.Int32(128)            # seq-local q tile offset
        nblk = mb + cutlass.Int32(1)                    # causal: tile mb sees blocks 0..mb

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

        # cooperative packed Q load (zero-fill invalid rows to avoid OOB / garbage)
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

        # FA4 packed-varlen addressing: offset the packed tensor by the batch's
        # token start, slice the (runtime) head, then tile by (N, D).
        # mK/mV are the permuted [tot, D, H] views. Offset token (mode0); KEEP the head
        # mode (the atom's gbasis references mode 2) — tile (tot->N blocks, D fixed),
        # head stays as a full mode and is selected at the copy index below.
        mK_off = cute.domain_offset((k_start, 0, 0), mK)     # [tot, D, H]
        mV_off = cute.domain_offset((k_start, 0, 0), mV)
        gK = cute.local_tile(mK_off, (self.N, self.D), (None, None, None))  # [N,D,n_tblk,1,H]
        gV = cute.local_tile(mV_off, (self.N, self.D), (None, None, None))
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        # ---- WG0: producer (dynamic nblk) ----
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                kp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
                vp = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stages)
                for j in cutlass.range(nblk, unroll=1):
                    pl_k.producer_acquire(kp)
                    cute.copy(tma_k, tKgK[(None, j, 0, head)], tKsK[(None, kp.index)],
                              tma_bar_ptr=pl_k.producer_get_barrier(kp))
                    pl_k.producer_commit(kp)
                    pl_v.producer_acquire(vp)
                    cute.copy(tma_v, tVgV[(None, j, 0, head)], tVsV[(None, vp.index)],
                              tma_bar_ptr=pl_v.producer_get_barrier(vp))
                    pl_v.producer_commit(vp)
                    kp.advance()
                    vp.advance()

        # ---- WG1/WG2: consumers (dynamic nblk; intra-wg QK(cur)/PV(prev) overlap) ----
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
            gOscr = mOscr[ft, None, head, None]                      # [128, D]
            gO = cute.local_tile(gOscr, (self.M, self.D), (None, None))
            tCgO = thr_pv.partition_C(gO[(None, None, 0, 0)])
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

            # Step A: block 0
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
                              True, coord_mn, cutlass.Int32(0), mask_q_off, k_len)
            tOrP.store(tOrP_v.load().to(self.dt))

            # Middle: j = 1..nblk-1 (runtime), QK(j) overlaps PV(j-1)
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
                                  False, coord_mn, j, mask_q_off, k_len)
                cute.nvgpu.warpgroup.wait_group(0)
                pl_v.consumer_release(srv)
                for r in cutlass.range_constexpr(nrows):
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
                tOrP.store(qlu.reshape_acc_to_frgA(acc_S).load().to(self.dt))

            # Step E: final PV
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

            # finalize: quad-reduce row_sum, divide; mask rows past the sequence
            for r in cutlass.range_constexpr(nrows):
                if (mask_q_off + coord_mn[r, 0][0]) < q_len:
                    s = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
                    inv = cutlass.Float32(1.0) / s
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)
                else:
                    acc_O_mn[r, None].store(acc_O_mn[r, None].load() * cutlass.Float32(0.0))
            tCgO.store(acc_O.load().to(mOscr.element_type))
