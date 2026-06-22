#!/usr/bin/env python3
"""
Intermediate single-CTA O_proj microkernel for Hopper SM90 CuTe DSL.

This file is not the core fused kernel. It isolates the O_proj tile payload used
by fused_fa_oproj_ar.py: one CTA computes one O_proj super-group for one FA row
tile. There is no persistent scheduler and no NVLS AllReduce in this file.

Payload contract:

    A = O_scratch[fa_row_tile_id, :, :, :]  viewed as [128, K_local]   (K_local = H_local*D)
    for sg_tile in 0..valid_n_tiles-1:
        out_n_tile = base_out_n_tile + sg_tile
        C[128, N_TILE] = A @ W_o_local[:, out_n_tile*N_TILE : +N_TILE]
        store -> C_sym[fa_row_tile_id, m, out_n_tile, n]   (m<valid_m, n<valid_n only)

Structure:
  * 3 warp groups: WG0 = TMA producer (1 DMA warp loads A and W_o chunks), WG1/WG2
    = WGMMA consumers. atom_layout_mnk=(2,1,1), tiler_mn=(64,N_TILE): WG1 owns
    rows 0..63, WG2 rows 64..127; both run the full K loop, each keeps only its own
    64-row accumulator. There is no cross-WG accumulator merge.
  * GEMM operand layout: A is K-major in SMEM; transpose_view(sWo) presents W_o as
    the MN-major B operand expected by the MMA atom.
  * Shared A/B pipeline: A_chunk[128,K_CHUNK] + Wo_chunk[K_CHUNK,N_TILE] loaded into
    one stage per K-chunk. Conservative release uses wait_group(0) then
    consumer_release per chunk.
  * Predicated store: writes C_sym only where m<valid_m AND n<valid_n; invalid
    tail elements are not written.

Agent note: scheduling, O_proj -> AR owner readiness, and NVLS completion live in
fused_fa_oproj_ar.py. Keep this file focused on O_proj tile mechanics.
"""
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from quack import layout_utils as qlu


def cdiv(a, b):
    return (a + b - 1) // b


class OProjTile:
    """One O_proj super-group for one FA row tile.

    Tile identity parameters are compile-time in this validation microkernel. The
    fused kernel decodes the same identity from a runtime slot_id.
    """

    def __init__(self, M, N_TILE, K_local, hidden, num_out_n_tiles, num_row_tiles,
                 fa_row_tile_id, base_out_n_tile, valid_n_tiles, valid_m,
                 K_CHUNK=64, num_stages=4, acc_dtype=cutlass.Float32):
        self.M = M                          # OPROJ_M_TILE = 128
        self.N_TILE = N_TILE                # 128
        self.K_local = K_local              # H_local * D
        self.hidden = hidden
        self.num_out_n_tiles = num_out_n_tiles
        self.num_row_tiles = num_row_tiles
        self.fa_row_tile_id = fa_row_tile_id
        self.base_out_n_tile = base_out_n_tile
        self.valid_n_tiles = valid_n_tiles
        self.valid_m = valid_m
        self.K_CHUNK = K_CHUNK
        self.n_kchunks = K_local // K_CHUNK
        self.num_stages = num_stages
        self.acc_dtype = acc_dtype
        self.num_dma_threads = 128
        self.mma_atom_layout_mnk = (2, 1, 1) if M > 64 else (1, 1, 1)
        self.num_mma_threads = 128 * self.mma_atom_layout_mnk[0]
        # Barrier invariant: PipelineTmaAsync consumer group size is the number of
        # consumer warps, not threads. Empty-stage arrival happens once per warp;
        # passing a thread count can hang when stages are reused.
        self.num_mma_warps = self.num_mma_threads // 32
        self.threads = self.num_dma_threads + self.num_mma_threads
        self.align = 1024

    def _smem(self, dtype, rows, cols, stages):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, dtype, cols),
            dtype)
        return cute.tile_to_shape(atom, (rows, cols, stages), order=(0, 1, 2))

    @cute.jit
    def __call__(self, mA: cute.Tensor, mWo: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        # Input/output layout contract for this standalone O_proj payload:
        # mA : O_scratch tile  [M, K_local, 1]   (K-major: K_local contiguous)
        # mWo: W_o_local       [K_local, hidden, 1]
        # mC : C_sym 4D        [num_row_tiles, M, num_out_n_tiles, N_TILE]
        dt = mA.element_type
        self.dt = dt
        sA_l = self._smem(dt, self.M, self.K_CHUNK, self.num_stages)
        sWo_l = self._smem(dt, self.K_CHUNK, self.N_TILE, self.num_stages)

        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mA, cute.slice_(sA_l, (None, None, 0)), (self.M, self.K_CHUNK),
            num_multicast=1)
        tma_wo, tWo = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, mWo, cute.slice_(sWo_l, (None, None, 0)), (self.K_CHUNK, self.N_TILE),
            num_multicast=1)

        # Operand contract: A is K-major in SMEM; transpose_view(sWo) presents W_o
        # as the MN-major B operand. C = A @ W_o.
        mma = sm90_utils.make_trivial_tiled_mma(
            dt, dt, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            self.acc_dtype, atom_layout_mnk=self.mma_atom_layout_mnk,
            tiler_mn=(64, self.N_TILE))

        @cute.struct
        class Smem:
            mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sA_l)], self.align]
            sWo: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sWo_l)], self.align]

        self.kernel(tma_a, tA, tma_wo, tWo, mC, mma, sA_l, sWo_l, Smem).launch(
            grid=[1, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, tma_a: cute.CopyAtom, mA: cute.Tensor,
               tma_wo: cute.CopyAtom, mWo: cute.Tensor, mC: cute.Tensor,
               mma: cute.TiledMma, sA_l: cute.ComposedLayout,
               sWo_l: cute.ComposedLayout, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg_idx = cute.arch.make_warp_uniform(tidx // 128)

        n_kchunks = cutlass.const_expr(self.n_kchunks)
        valid_n_tiles = cutlass.const_expr(self.valid_n_tiles)
        base_out = cutlass.const_expr(self.base_out_n_tile)
        ft = cutlass.const_expr(self.fa_row_tile_id)
        N_TILE = cutlass.const_expr(self.N_TILE)
        hidden = cutlass.const_expr(self.hidden)
        valid_m = cutlass.const_expr(self.valid_m)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_wo)

        tx = (cute.size_in_bytes(self.dt, cute.slice_(sA_l, (None, None, 0)))
              + cute.size_in_bytes(self.dt, cute.slice_(sWo_l, (None, None, 0))))

        al = cutlass.utils.SmemAllocator()
        st = al.allocate(Smem)
        sA = st.sA.get_tensor(sA_l.outer, swizzle=sA_l.inner)
        sWo = st.sWo.get_tensor(sWo_l.outer, swizzle=sWo_l.inner)

        prod = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cons = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps)
        pl = pipeline.PipelineTmaAsync.create(
            barrier_storage=st.mbar.data_ptr(), num_stages=self.num_stages,
            producer_group=prod, consumer_group=cons, tx_count=tx)

        # Gmem tiling: mA tiles over K as (M, K_CHUNK); mWo tiles over (K, N) as
        # (K_CHUNK, N_TILE).
        gA = cute.local_tile(mA, (self.M, self.K_CHUNK), (None, None, None))
        gWo = cute.local_tile(mWo, (self.K_CHUNK, self.N_TILE), (None, None, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2))
        tWosWo, tWogWo = cute.nvgpu.cpasync.tma_partition(
            tma_wo, 0, cute.make_layout(1), cute.group_modes(sWo, 0, 2),
            cute.group_modes(gWo, 0, 2))

        # WG0 producer: flatten (sg_tile, k_chunk) into one staged stream of
        # A_chunk + Wo_chunk. A is re-streamed for each output N tile.
        if wg_idx == 0:
            cute.arch.setmaxregister_decrease(40)
            if warp_idx == 0:
                pp = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages)
                for sg in cutlass.range_constexpr(valid_n_tiles):
                    out_n_tile = base_out + sg
                    for kc in cutlass.range_constexpr(n_kchunks):
                        pl.producer_acquire(pp)
                        bar = pl.producer_get_barrier(pp)
                        cute.copy(tma_a, tAgA[(None, 0, kc, 0)],
                                  tAsA[(None, pp.index)], tma_bar_ptr=bar)
                        cute.copy(tma_wo, tWogWo[(None, kc, out_n_tile, 0)],
                                  tWosWo[(None, pp.index)], tma_bar_ptr=bar)
                        pl.producer_commit(pp)
                        pp.advance()

        # WG1/WG2 consumers: (2,1,1) MMA splits M into two 64-row slices. Each WG
        # runs the full K loop into its own accumulator and stores with tail predicates.
        if wg_idx >= 1:
            cute.arch.setmaxregister_increase(232)
            lane = tidx - self.num_dma_threads
            thr = mma.get_slice(lane)
            tCrA = mma.make_fragment_A(thr.partition_A(sA))
            sWot = qlu.transpose_view(sWo)
            tCrWo = mma.make_fragment_B(thr.partition_B(sWot))
            idC = cute.make_identity_tensor((self.M, self.N_TILE))
            acc_C = cute.make_rmem_tensor(thr.partition_C(idC).shape[:3], self.acc_dtype)
            coord_mn = qlu.reshape_acc_to_mn(thr.partition_C(idC))
            acc_C_mn = qlu.reshape_acc_to_mn(acc_C)
            nrows = cutlass.const_expr(cute.size(acc_C_mn, mode=[0]))
            ncols = cutlass.const_expr(cute.size(acc_C_mn, mode=[1]))
            nkb = cutlass.const_expr(cute.size(tCrA, mode=[2]))

            sc = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages)

            for sg in cutlass.range_constexpr(valid_n_tiles):
                out_n_tile = base_out + sg
                valid_n = cutlass.const_expr(min(N_TILE, hidden - out_n_tile * N_TILE))
                acc_C.fill(0.0)
                mma.set(warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for kc in cutlass.range_constexpr(n_kchunks):
                    pl.consumer_wait(sc)
                    for kb in cutlass.range_constexpr(nkb):
                        cute.gemm(mma, acc_C, tCrA[(None, None, kb, sc.index)],
                                  tCrWo[(None, None, kb, sc.index)], acc_C)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)     # conservative: no overlap
                    pl.consumer_release(sc)
                    sc.advance()

                # Tail invariant: only valid token rows and hidden columns are written.
                gC = mC[ft, None, out_n_tile, None]        # [M, N_TILE] view
                tCgC = thr.partition_C(gC)
                gC_mn = qlu.reshape_acc_to_mn(tCgC)
                for r in cutlass.range_constexpr(nrows):
                    crow = coord_mn[r, None]
                    for c in cutlass.range_constexpr(ncols):
                        m = crow[c][0]
                        n = crow[c][1]
                        if (m < valid_m) and (n < valid_n):
                            gC_mn[r, c] = acc_C_mn[r, c].to(self.dt)
