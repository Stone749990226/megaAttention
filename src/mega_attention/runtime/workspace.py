#!/usr/bin/env python3
"""Long-lived workspace for the fused FA + O_proj + NVLS AR kernel (设计稿: Workspace
生命周期与全局同步).

This is the sglang-facing runtime resource contract, NOT a generic attention API. It
allocates every kernel buffer once at a bucket capacity, rendezvous-es the symmetric
allocations, bakes the multicast / peer pointers into one compiled kernel, and then
launches any active batch whose `num_row_tiles <= max_num_row_tiles` WITHOUT host reset
and WITHOUT recompiling. The kernel-start directed cleaner zeros per-layer control state
and the phase/sign barriers reuse their slots, so cross-layer reuse needs no host zero_().

Lifecycle:

    ws = FusedFaOprojArWorkspace.create(group_name, max_num_row_tiles=..., hidden=...,
                                        H_local=..., D=..., tp_size=..., rank=...)
    ws.compile()                                  # one cute.compile at bucket capacity
    for layer in layers:
        ws.set_layer(meta, Q, K, V, W_o)          # fill active prefixes + actv
        ws.launch()                               # no host reset
        out = ws.csym                             # C_sym[valid] is this layer's output

`debug_zero_all()` is for fault localization only; the production / benchmark / test
default path must not call it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import (
    FusedFaOprojAr, NUM_CTRL, NUM_SYNC)
from mega_attention.metadata.row_desc import oproj_task_counts, active_counts


class FusedFaOprojArWorkspace:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self._compiled = None

    # ---------------------------------------------------------------- create
    @classmethod
    def create(cls, group_name, max_num_row_tiles, hidden, H_local, D,
               tp_size=1, rank=0, N_TILE=128, super_group_n_tiles=4, q_per_kv=1,
               max_num_batch=None, max_tot_k=None, dtype=torch.bfloat16, device=None):
        dev = torch.device(device if device is not None
                           else f"cuda:{torch.cuda.current_device()}")
        assert H_local % q_per_kv == 0
        H_kv = H_local // q_per_kv
        K_local = H_local * D
        if max_num_batch is None:
            max_num_batch = max_num_row_tiles          # worst case: one seq per tile
        num_out, num_super_groups, max_total_oproj = oproj_task_counts(
            max_num_row_tiles, hidden, N_TILE, super_group_n_tiles)
        hidden_pad = num_out * N_TILE
        tot_cap = max_num_row_tiles * 128              # Q-token capacity (O_scratch/C_sym)
        # K/V token capacity is independent of Q tiles: chunked prefill has k_len >> q_len
        # and T_k only sizes the FA K/V read, not O_scratch/C_sym (设计稿: 容量估算).
        if max_tot_k is None:
            max_tot_k = tot_cap
        max_owner_slots = (max_total_oproj + tp_size - 1) // tp_size
        max_owner_words = (max_owner_slots + 63) // 64

        def _u32(n): return torch.zeros(n, dtype=torch.uint32, device=dev)

        # Symmetric allocations (rendezvous once). For tp_size == 1 the AR is identity,
        # so plain local tensors suffice and the baked pointers stay 0.
        if tp_size > 1:
            csym = symm_mem.empty(max_num_row_tiles, 128, num_out, N_TILE,
                                  device=dev, dtype=dtype); csym.zero_()
            hC = symm_mem.rendezvous(csym, group_name)
            assert hC.has_multicast_support
            rco = symm_mem.empty(max_owner_slots, device=dev, dtype=torch.uint32); rco.zero_()
            hRC = symm_mem.rendezvous(rco, group_name)
            rbits = symm_mem.empty(max_owner_words, device=dev, dtype=torch.int64); rbits.zero_()
            hRB = symm_mem.rendezvous(rbits, group_name)
            nvl = symm_mem.empty(2, device=dev, dtype=torch.int32); nvl.zero_()
            hN = symm_mem.rendezvous(nvl, group_name)
            csym_mc_ptr = hC.multicast_ptr
            nvl_local_ptr = hN.buffer_ptrs[rank]
            nvl_peer_ptrs = [hN.buffer_ptrs[r] for r in range(tp_size)]
            rc_ptrs = [hRC.buffer_ptrs[r] for r in range(tp_size)]
            rb_ptrs = [hRB.buffer_ptrs[r] for r in range(tp_size)]
        else:
            csym = torch.zeros(max_num_row_tiles, 128, num_out, N_TILE, device=dev, dtype=dtype)
            rco = _u32(max_owner_slots)
            rbits = torch.zeros(max_owner_words, dtype=torch.int64, device=dev)
            nvl = torch.zeros(2, dtype=torch.int32, device=dev)
            csym_mc_ptr = 0; nvl_local_ptr = 0
            nvl_peer_ptrs = []; rc_ptrs = []; rb_ptrs = []

        ws = cls(
            dev=dev, dtype=dtype, tp_size=tp_size, rank=rank,
            H_local=H_local, H_kv=H_kv, D=D, K_local=K_local, hidden=hidden,
            hidden_pad=hidden_pad, N_TILE=N_TILE, super_group_n_tiles=super_group_n_tiles,
            q_per_kv=q_per_kv, num_out=num_out, num_super_groups=num_super_groups,
            max_num_row_tiles=max_num_row_tiles, max_num_batch=max_num_batch,
            max_total_oproj=max_total_oproj, tot_cap=tot_cap, max_tot_k=max_tot_k,
            # local control + metadata (all capacity-shaped; active fills prefixes)
            ctrl=_u32(NUM_CTRL), sync_ctrl=_u32(NUM_SYNC),
            actv=torch.zeros(6, dtype=torch.int32, device=dev),
            head_ready=_u32(max_num_row_tiles), oproj_queue=_u32(max_total_oproj),
            ar_done_bits=torch.zeros(max_owner_words, dtype=torch.int64, device=dev),
            cu_q=torch.zeros(max_num_batch + 1, dtype=torch.int32, device=dev),
            cu_k=torch.zeros(max_num_batch + 1, dtype=torch.int32, device=dev),
            fa_b=torch.zeros(max_num_row_tiles, dtype=torch.int32, device=dev),
            fa_mb=torch.zeros(max_num_row_tiles, dtype=torch.int32, device=dev),
            # input activation buffers (capacity-shaped; fixed shape across layers)
            Q=torch.zeros(tot_cap, H_local, D, device=dev, dtype=dtype),
            K=torch.zeros(max_tot_k, H_kv, D, device=dev, dtype=dtype),
            V=torch.zeros(max_tot_k, H_kv, D, device=dev, dtype=dtype),
            W_o=torch.zeros(K_local, hidden_pad, device=dev, dtype=dtype),
            Oscr=torch.zeros(max_num_row_tiles, 128, H_local, D, device=dev, dtype=dtype),
            csym=csym, rco=rco, rbits=rbits, nvl=nvl,
            csym_mc_ptr=csym_mc_ptr, nvl_local_ptr=nvl_local_ptr,
            nvl_peer_ptrs=nvl_peer_ptrs, rc_ptrs=rc_ptrs, rb_ptrs=rb_ptrs,
        )
        # One-time global zero barrier: the kernel has no start cleaner / init barrier,
        # so the first layer relies on this full zero being complete and cross-rank
        # visible before any rank issues a remote control write.
        torch.cuda.synchronize()
        if tp_size > 1 and dist.is_initialized():
            dist.barrier()
        return ws

    # ---------------------------------------------------------------- compile
    def _cts(self):
        a4 = (self.ctrl, self.sync_ctrl, self.actv, self.head_ready,
              self.oproj_queue, self.rco)
        a8 = (self.rbits, self.ar_done_bits)
        a16 = (self.Q, self.K, self.V, self.Oscr, self.W_o, self.csym,
               self.cu_q, self.cu_k, self.fa_b, self.fa_mb)
        cts = [from_dlpack(t, assumed_align=4) for t in a4]
        cts += [from_dlpack(t, assumed_align=8) for t in a8]
        cts += [from_dlpack(t, assumed_align=16) for t in a16]
        return cts

    def compile(self, w_fa=4, w_oproj=1, w_ar=1, num_ctas=132, kv_stages=2,
                K_CHUNK=64, oproj_stages=4, softmax_scale=None):
        ker = FusedFaOprojAr(
            num_fa=self.max_num_row_tiles * self.H_local,
            num_row_tiles=self.max_num_row_tiles, H_local=self.H_local, D=self.D,
            num_super_groups=self.num_super_groups, total_oproj=self.max_total_oproj,
            num_ctas=num_ctas, hidden=self.hidden, tp_size=self.tp_size, rank=self.rank,
            q_per_kv=self.q_per_kv, N_TILE=self.N_TILE,
            super_group_n_tiles=self.super_group_n_tiles, kv_stages=kv_stages,
            K_CHUNK=K_CHUNK, oproj_stages=oproj_stages,
            csym_mc_ptr=self.csym_mc_ptr, nvl_local_ptr=self.nvl_local_ptr,
            nvl_peer_ptrs=self.nvl_peer_ptrs, rc_ptrs=self.rc_ptrs, rb_ptrs=self.rb_ptrs,
            softmax_scale=softmax_scale, w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar)
        self._ts = torch.cuda.current_stream()
        self._st = cuda.CUstream(self._ts.cuda_stream)
        self._cts_cache = self._cts()
        self._compiled = cute.compile(ker, *self._cts_cache, self._st)
        return self

    # ---------------------------------------------------------------- per layer
    def set_layer(self, meta, Q, K, V, W_o):
        """Fill active input prefixes + varlen metadata + runtime active counts.

        Q:[tot_q,H_local,D]  K/V:[tot_k,H_kv,D]  W_o:[K_local,hidden]. Everything is
        copied into the capacity buffers' active prefix; the rest is stale but unread.
        """
        R = meta.num_row_tiles
        nb = meta.num_batch
        assert R <= self.max_num_row_tiles and nb <= self.max_num_batch
        tq, tk = Q.shape[0], K.shape[0]
        assert tq <= self.tot_cap and tk <= self.max_tot_k
        self.Q[:tq].copy_(Q); self.K[:tk].copy_(K); self.V[:tk].copy_(V)
        self.W_o[:, :self.hidden].copy_(W_o)
        d = self.dev
        self.cu_q[:nb + 1].copy_(torch.as_tensor(meta.cu_seqlens_q, device=d, dtype=torch.int32))
        self.cu_k[:nb + 1].copy_(torch.as_tensor(meta.cu_seqlens_k, device=d, dtype=torch.int32))
        self.fa_b[:R].copy_(torch.as_tensor(meta.batch_idx, device=d, dtype=torch.int32))
        self.fa_mb[:R].copy_(torch.as_tensor(meta.m_block, device=d, dtype=torch.int32))
        self.actv.copy_(torch.as_tensor(
            active_counts(R, self.H_local, self.num_super_groups, self.tp_size, self.rank),
            device=d))
        self._meta = meta

    def launch(self):
        assert self._compiled is not None, "call compile() first"
        with torch.cuda.stream(self._ts):
            self._compiled(*self._cts_cache, self._st)

    # ---------------------------------------------------------------- debug
    def debug_zero_all(self):
        """Fault-localization only. Not used on the production / bench / test path."""
        for t in (self.ctrl, self.sync_ctrl, self.actv, self.head_ready,
                  self.oproj_queue, self.rco, self.rbits, self.ar_done_bits,
                  self.nvl, self.csym, self.Oscr):
            t.zero_()
