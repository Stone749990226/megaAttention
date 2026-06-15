#!/usr/bin/env python3
"""
Standalone single-tile FlashAttention building blocks (Hopper SM90, CuTe DSL).

Phase 2 axis-B microkernel: develop + validate FA compute in isolation (vs torch)
BEFORE integrating into the persistent scheduler skeleton. Built in increments:

  increment 1 (this commit): single-tile S = Q @ K^T via WGMMA, fp32 out.
  later: + online softmax, + P@V, + kv-loop, + causal mask, + varlen.

Load mechanism note: for the standalone *correctness* microkernel we fill smem
with a simple cooperative synchronous copy (each thread writes logical (row,col)
into the swizzled smem tensor). TMA is intentionally deferred to Phase 2d, where
the persistent kernel uses the proven cutlass.pipeline.PipelineTmaAsync path
(raw-mbarrier TMA had a completion subtlety not worth fighting for correctness
validation). Clean rewrite: cutlass.cute / hopper_helpers primitives only.
"""
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup


def cdiv(a, b):
    return (a + b - 1) // b


@cute.jit
def coop_load(smem_t: cute.Tensor, gmem_t: cute.Tensor,
              rows: cutlass.Constexpr, cols: cutlass.Constexpr, tidx,
              num_threads: cutlass.Constexpr):
    """Cooperative synchronous gmem->smem copy of a [rows, cols] tile (L=1).

    Each thread writes a strided subset of logical (r,c) elements; indexing the
    (possibly swizzled) smem tensor by logical coords routes to the right slot.
    """
    n = rows * cols
    for i in cutlass.range_constexpr(cdiv(n, num_threads)):
        idx = tidx + i * num_threads
        if idx < n:
            r = idx // cols
            c = idx % cols
            smem_t[r, c, 0] = gmem_t[r, c, 0]


@cute.jit
def coop_load_block(smem_t: cute.Tensor, gmem_t: cute.Tensor, row_off,
                    rows: cutlass.Constexpr, cols: cutlass.Constexpr, tidx,
                    num_threads: cutlass.Constexpr):
    """smem_t[r,c] = gmem_t[row_off + r, c] for a [rows, cols] sub-block (L=1)."""
    n = rows * cols
    for i in cutlass.range_constexpr(cdiv(n, num_threads)):
        idx = tidx + i * num_threads
        if idx < n:
            r = idx // cols
            c = idx % cols
            smem_t[r, c, 0] = gmem_t[row_off + r, c, 0]


@cute.jit
def coop_load_block_T(smem_t: cute.Tensor, gmem_t: cute.Tensor, row_off,
                      Dd: cutlass.Constexpr, Nn: cutlass.Constexpr, tidx,
                      num_threads: cutlass.Constexpr):
    """Transposing load: smem_t[d,n] = gmem_t[row_off + n, d] (smem [Dd,Nn] <- gmem [Nn,Dd])."""
    n_elems = Dd * Nn
    for i in cutlass.range_constexpr(cdiv(n_elems, num_threads)):
        idx = tidx + i * num_threads
        if idx < n_elems:
            d = idx // Nn
            nn = idx % Nn
            smem_t[d, nn, 0] = gmem_t[row_off + nn, d, 0]


class QKTileSm90:
    """Single CTA, single warp group: S[M,N] = Q[M,D] @ K[N,D]^T (one tile)."""

    def __init__(self, M, N, D, acc_dtype=cutlass.Float32):
        self.M = M
        self.N = N
        self.D = D                       # contraction (head_dim)
        self.acc_dtype = acc_dtype
        self.num_threads = 128           # one warp group

    def _smem_layout(self, dtype, rows, cols):
        """K-major (cols contiguous) staged-by-1 swizzled smem layout."""
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype,
        )
        return cute.tile_to_shape(atom, (rows, cols, 1), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mS: cute.Tensor,
                 stream: cuda.CUstream):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        sQ_staged = self._smem_layout(self.q_dtype, self.M, self.D)
        sK_staged = self._smem_layout(self.k_dtype, self.N, self.D)

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype, self.k_dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.N),
        )

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_staged)], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_staged)], 1024]

        self.kernel(mQ, mK, mS, tiled_mma, sQ_staged, sK_staged, Smem).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mS: cute.Tensor,
               tiled_mma: cute.TiledMma, sQ_staged: cute.ComposedLayout,
               sK_staged: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Smem)
        sQ = storage.sQ.get_tensor(sQ_staged.outer, swizzle=sQ_staged.inner)
        sK = storage.sK.get_tensor(sK_staged.outer, swizzle=sK_staged.inner)

        # ---- cooperative synchronous load Q,K into smem ----
        coop_load(sQ, mQ, self.M, self.D, tidx, self.num_threads)
        coop_load(sK, mK, self.N, self.D, tidx, self.num_threads)
        cute.arch.sync_threads()

        # ---- WGMMA: S = Q @ K^T ----
        thr_mma = tiled_mma.get_slice(tidx)
        tCrQ = tiled_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tCrK = tiled_mma.make_fragment_B(thr_mma.partition_B(sK))

        gS = cute.local_tile(mS, (self.M, self.N), (None, None, None))
        tCgS = thr_mma.partition_C(gS[(None, None, 0, 0, 0)])
        acc = cute.make_rmem_tensor(tCgS.shape[:3], self.acc_dtype)

        num_k_blocks = cute.size(tCrQ, mode=[2])
        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        cute.nvgpu.warpgroup.fence()
        for kb in cutlass.range_constexpr(num_k_blocks):
            cute.gemm(tiled_mma, acc, tCrQ[(None, None, kb, 0)],
                      tCrK[(None, None, kb, 0)], acc)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        # ---- store fp32 acc -> gmem S ----
        tCgS.store(acc.load())


LOG2E = 1.4426950408889634


class FaTileSm90:
    """Single CTA: O[M,D] = softmax(Q[M,D] K[Lk,D]^T * scale) V[Lk,D] (one row tile).

    Increment 2 (non-causal): online softmax over Lk/N kv blocks. QK and PV both
    via WGMMA; the softmax + online correction run thread-per-row through smem
    (thread t owns row t) to avoid WGMMA C-fragment reductions in this first
    correct version. M must be <= num_threads. Lk % N == 0, D % 16 == 0.
    """

    def __init__(self, M, N, D, Lk, softmax_scale=None, acc_dtype=cutlass.Float32,
                 causal=False, q_start=0, valid_m=None, k_len=None):
        self.M = M
        self.N = N            # kv block size
        self.D = D            # head dim
        self.Lk = Lk          # total kv buffer length (loop bound)
        self.nblk = Lk // N
        # varlen predication: only valid_m query rows are written; only k_len
        # keys are valid (keys >= k_len in the last partial block are masked).
        self.valid_m = valid_m if valid_m is not None else M
        self.k_len = k_len if k_len is not None else Lk
        self.scale = softmax_scale if softmax_scale is not None else D ** -0.5
        self.scale_log2 = self.scale * LOG2E
        self.acc_dtype = acc_dtype
        # causal: query at absolute position (q_start + t) attends to key kv_pos
        # iff kv_pos <= q_start + t (prefill, seqlen_q == seqlen_k alignment).
        self.causal = causal
        self.q_start = q_start
        self.num_threads = 128

    def _swz(self, dtype, rows, cols):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, 1), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, stream: cuda.CUstream):
        dt = mQ.element_type
        self.dt = dt
        sQ_l = self._swz(dt, self.M, self.D)
        sK_l = self._swz(dt, self.N, self.D)
        sVt_l = self._swz(dt, self.D, self.N)     # V transposed [D, N]
        sP_l = self._swz(dt, self.M, self.N)

        mma_qk = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.N))
        mma_pv = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            self.acc_dtype, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.D))

        self.kernel(mQ, mK, mV, mO, mma_qk, mma_pv,
                    sQ_l, sK_l, sVt_l, sP_l).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mO: cute.Tensor, mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
               sQ_l: cute.ComposedLayout, sK_l: cute.ComposedLayout,
               sVt_l: cute.ComposedLayout, sP_l: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        M = cutlass.const_expr(self.M)
        N = cutlass.const_expr(self.N)
        D = cutlass.const_expr(self.D)
        nblk = cutlass.const_expr(self.nblk)
        nthr = cutlass.const_expr(self.num_threads)
        slog2 = cutlass.const_expr(self.scale_log2)
        causal = cutlass.const_expr(self.causal)
        q_start = cutlass.const_expr(self.q_start)
        vm = cutlass.const_expr(self.valid_m)
        klen = cutlass.const_expr(self.k_len)

        al = cutlass.utils.SmemAllocator()
        sQ = al.allocate_tensor(self.dt, sQ_l.outer, byte_alignment=1024, swizzle=sQ_l.inner)
        sK = al.allocate_tensor(self.dt, sK_l.outer, byte_alignment=1024, swizzle=sK_l.inner)
        sVt = al.allocate_tensor(self.dt, sVt_l.outer, byte_alignment=1024, swizzle=sVt_l.inner)
        sP = al.allocate_tensor(self.dt, sP_l.outer, byte_alignment=1024, swizzle=sP_l.inner)
        sS = al.allocate_tensor(self.acc_dtype, cute.make_layout((M, N), stride=(N, 1)),
                                byte_alignment=16)
        sOb = al.allocate_tensor(self.acc_dtype, cute.make_layout((M, D), stride=(D, 1)),
                                 byte_alignment=16)
        accO = al.allocate_tensor(self.acc_dtype, cute.make_layout((M, D), stride=(D, 1)),
                                  byte_alignment=16)

        thr_qk = mma_qk.get_slice(tidx)
        thr_pv = mma_pv.get_slice(tidx)

        # load Q once; zero the persistent O accumulator
        coop_load(sQ, mQ, M, D, tidx, nthr)
        for i in cutlass.range_constexpr(cdiv(M * D, nthr)):
            idx = tidx + i * nthr
            if idx < M * D:
                accO[idx // D, idx % D] = self.acc_dtype(0.0)
        cute.arch.sync_threads()

        row_max = cutlass.Float32(-1.0e30)
        row_sum = cutlass.Float32(0.0)

        for j in cutlass.range_constexpr(nblk):
            row_off = j * N
            coop_load_block(sK, mK, row_off, N, D, tidx, nthr)
            coop_load_block_T(sVt, mV, row_off, D, N, tidx, nthr)
            cute.arch.sync_threads()

            # ---- S = Q @ K_j^T (WGMMA) -> sS ----
            rQ = mma_qk.make_fragment_A(thr_qk.partition_A(sQ))
            rK = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
            cS = thr_qk.partition_C(sS)
            accS = cute.make_rmem_tensor(cS.shape[:3], self.acc_dtype)
            nk_qk = cute.size(rQ, mode=[2])
            mma_qk.set(warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(nk_qk):
                cute.gemm(mma_qk, accS, rQ[(None, None, kb, 0)], rK[(None, None, kb, 0)], accS)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cS.store(accS.load())
            cute.arch.sync_threads()

            # ---- thread-per-row online softmax (thread t owns row t) ----
            # Two mask sources: k-bound (kv_pos >= k_len) is fully constexpr here
            # (skipped at compile time); causal (kv_pos > q_pos) is dynamic in t.
            if tidx < M:
                q_pos = q_start + tidx           # absolute query position
                blk_max = cutlass.Float32(-1.0e30)
                for n in cutlass.range_constexpr(N):
                    if cutlass.const_expr((row_off + n) < klen):
                        cmask = False
                        if cutlass.const_expr(causal):
                            cmask = (row_off + n) > q_pos
                        if not cmask:
                            v = sS[tidx, n]
                            if v > blk_max:
                                blk_max = v
                new_max = row_max
                if blk_max > row_max:
                    new_max = blk_max
                corr = cute.math.exp2((row_max - new_max) * slog2, fastmath=True)
                psum = cutlass.Float32(0.0)
                for n in cutlass.range_constexpr(N):
                    if cutlass.const_expr((row_off + n) >= klen):
                        sP[tidx, n, 0] = self.dt(cutlass.Float32(0.0))
                    else:
                        cmask = False
                        if cutlass.const_expr(causal):
                            cmask = (row_off + n) > q_pos
                        if cmask:
                            sP[tidx, n, 0] = self.dt(cutlass.Float32(0.0))
                        else:
                            p = cute.math.exp2(sS[tidx, n] * slog2 - new_max * slog2, fastmath=True)
                            sP[tidx, n, 0] = self.dt(p)
                            psum = psum + p
                row_sum = row_sum * corr + psum
                row_max = new_max
                for d in cutlass.range_constexpr(D):
                    accO[tidx, d] = accO[tidx, d] * corr
            cute.arch.sync_threads()

            # ---- O_block = P @ V_j (WGMMA) -> sOb ; accO += O_block ----
            rP = mma_pv.make_fragment_A(thr_pv.partition_A(sP))
            rVt = mma_pv.make_fragment_B(thr_pv.partition_B(sVt))
            cO = thr_pv.partition_C(sOb)
            accOb = cute.make_rmem_tensor(cO.shape[:3], self.acc_dtype)
            nk_pv = cute.size(rP, mode=[2])
            mma_pv.set(warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range_constexpr(nk_pv):
                cute.gemm(mma_pv, accOb, rP[(None, None, kb, 0)], rVt[(None, None, kb, 0)], accOb)
                mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cO.store(accOb.load())
            cute.arch.sync_threads()

            if tidx < M:
                for d in cutlass.range_constexpr(D):
                    accO[tidx, d] = accO[tidx, d] + sOb[tidx, d]
            cute.arch.sync_threads()

        # ---- finalize: O = accO / row_sum (only valid_m rows are written) ----
        if tidx < vm:
            inv = cutlass.Float32(1.0) / row_sum
            for d in cutlass.range_constexpr(D):
                mO[tidx, d, 0] = mO.element_type(accO[tidx, d] * inv)
