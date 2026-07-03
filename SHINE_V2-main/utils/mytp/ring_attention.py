"""
Ring Flash Attention wrapper for Sequence Parallel (SP).

Self-contained implementation of zigzag ring flash attention that is compatible
with flash_attn 2.8.3 (which returns 5 values from _flash_attn_forward).

Provides:
  - zigzag_split / zigzag_unsplit: partition/restore a full sequence into
    zigzag-balanced chunks across SP ranks.
  - sp_flash_attention: unified interface that performs zigzag ring attention
    with proper process group, handling GQA and causal mask automatically.
  - sp_flash_attention_alltoall_zigzag: **RECOMMENDED DEFAULT** for training.
    Accepts contiguous chunks, internally does all-to-all + zigzag ring + all-to-all.
    31% faster than contiguous ring with only 1-3% more memory (with grad ckpt).
  - sp_flash_attention_contiguous: contiguous ring attention (no zigzag).
    Simpler but slower due to causal load imbalance.

Expected tensor layouts (matching flash_attn convention):
  q: [B, S_local, H_q, D]
  k: [B, S_local, H_kv, D]
  v: [B, S_local, H_kv, D]
  output: [B, S_local, H_q, D]
"""

from __future__ import annotations

import os
# Disable flash_attn's "precision enhancement" which causes non-determinism
# on H20 GPUs with head_dim=128. Must be set before importing flash_attn.
os.environ.setdefault("PRECISION_ENHENCEMENT_FA2", "0")

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import _flash_attn_backward

__all__ = [
    "zigzag_split",
    "zigzag_unsplit",
    "sp_flash_attention",
    "sp_flash_attention_contiguous",
    "sp_flash_attention_alltoall_zigzag",
    "get_sp_attention_fn",
]


# ---------------------------------------------------------------------------
# Ring communication helper
# ---------------------------------------------------------------------------

class RingComm:
    """Point-to-point ring communication for KV exchange."""

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self, k: torch.Tensor, v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


# ---------------------------------------------------------------------------
# LSE merge utilities
# ---------------------------------------------------------------------------

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online softmax merge of two attention blocks."""
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


# ---------------------------------------------------------------------------
# Flash attention forward/backward wrappers (compatible with flash_attn 2.8.3)
# ---------------------------------------------------------------------------

def _fa_forward(q, k, v, dropout_p, softmax_scale, causal, window_size):
    """Call flash_attn_func and return (out, lse).

    Uses the high-level flash_attn_func API which is deterministic
    (unlike the low-level _flash_attn_forward which has non-determinism
    issues with D>=128 due to split-K atomicAdd).
    """
    # Use return_attn_probs=True to get softmax_lse
    result = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        return_attn_probs=True,
    )
    # Returns (out, softmax_lse, S_dmask) when return_attn_probs=True
    block_out = result[0].to(torch.float32)
    block_lse = result[1]  # [B, H, S_q]
    return block_out, block_lse


def _fa_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv,
                 dropout_p, softmax_scale, causal, window_size, deterministic):
    """Call _flash_attn_backward compatible with flash_attn 2.8.3."""
    _flash_attn_backward(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=0.0,
        alibi_slopes=None,
        deterministic=deterministic,
    )


# ---------------------------------------------------------------------------
# Zigzag Ring Flash Attention Forward
# ---------------------------------------------------------------------------

def zigzag_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size=(-1, -1),
    deterministic: bool = False,
):
    """Zigzag ring flash attention forward pass."""
    assert causal, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_lse = _fa_forward(q, k, v, dropout_p, softmax_scale, causal=True, window_size=window_size)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = _fa_forward(q, k0, v0, dropout_p, softmax_scale, causal=False, window_size=window_size)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = _fa_forward(q1, k, v, dropout_p, softmax_scale, causal=False, window_size=window_size)
            out, lse = update_out_and_lse(
                out, lse, block_out, block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


# ---------------------------------------------------------------------------
# Zigzag Ring Flash Attention Backward
# ---------------------------------------------------------------------------

def zigzag_ring_flash_attn_backward(
    process_group,
    dout,
    q, k, v,
    out,
    softmax_lse,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size=(-1, -1),
    deterministic: bool = False,
):
    """Zigzag ring flash attention backward pass."""
    assert causal, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dout1 = dout.chunk(2, dim=1)[1]
    q1 = q.chunk(2, dim=1)[1]
    out1 = out.chunk(2, dim=1)[1]
    softmax_lse1 = softmax_lse.chunk(2, dim=2)[1].contiguous()
    block_seq_len = q.shape[1] // 2

    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            _fa_backward(
                dout, q, k, v, out, softmax_lse,
                dq_buffer, dk_buffer, dv_buffer,
                dropout_p, softmax_scale, causal=True, window_size=window_size,
                deterministic=deterministic,
            )
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                _fa_backward(
                    dout, q, k0, v0, out, softmax_lse,
                    dq_buffer, dk_buffer[:, :block_seq_len], dv_buffer[:, :block_seq_len],
                    dropout_p, softmax_scale, causal=False, window_size=window_size,
                    deterministic=deterministic,
                )
                dq += dq_buffer
            else:
                _fa_backward(
                    dout1, q1, k, v, out1, softmax_lse1,
                    dq_buffer[:, :block_seq_len], dk_buffer, dv_buffer,
                    dropout_p, softmax_scale, causal=False, window_size=window_size,
                    deterministic=deterministic,
                )
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class ZigZagRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        deterministic,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = zigzag_ring_flash_attn_forward(
            group, q, k, v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_backward(
            ctx.group,
            dout, q, k, v, out, softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Contiguous Ring Flash Attention Forward (Scheme A - no zigzag)
# ---------------------------------------------------------------------------

def contiguous_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size=(-1, -1),
    deterministic: bool = False,
):
    """Contiguous ring flash attention forward pass (Scheme A).
    
    Each rank holds a contiguous chunk of the sequence:
      rank 0: tokens [0, ..., chunk_size-1]
      rank 1: tokens [chunk_size, ..., 2*chunk_size-1]
      ...
    
    For causal attention, rank r only needs KV from ranks 0..r (not future ranks).
    This leads to load imbalance but avoids any extra communication beyond ring KV passing.
    """
    assert causal, "contiguous ring attention only supports causal=True"
    comm = RingComm(process_group)

    seq_len = q.shape[1]
    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        # Determine which rank's KV we currently have.
        # After `step` ring passes, we have KV from rank (our_rank - step) % world_size
        kv_rank = (comm.rank - step) % comm.world_size

        if kv_rank == comm.rank:
            # Self-attention: use causal mask
            block_out, block_lse = _fa_forward(
                q, k, v, dropout_p, softmax_scale, causal=True, window_size=window_size
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif kv_rank < comm.rank:
            # KV from an earlier rank: all tokens are visible (no causal mask needed)
            block_out, block_lse = _fa_forward(
                q, k, v, dropout_p, softmax_scale, causal=False, window_size=window_size
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            # KV from a later rank: all tokens are masked (skip computation)
            pass

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


# ---------------------------------------------------------------------------
# Contiguous Ring Flash Attention Backward (Scheme A)
# ---------------------------------------------------------------------------

def contiguous_ring_flash_attn_backward(
    process_group,
    dout,
    q, k, v,
    out,
    softmax_lse,
    softmax_scale: float,
    dropout_p: float = 0,
    causal: bool = True,
    window_size=(-1, -1),
    deterministic: bool = False,
):
    """Contiguous ring flash attention backward pass (Scheme A).
    
    For contiguous ring attention, rank r computes with KV from ranks 0..r.
    The backward:
    - dQ: accumulate contributions from all KV blocks used (ranks 0..r)
    - dK/dV: each KV block gets contributions from all Q ranks that used it.
      For rank s's KV, contributions come from ranks s..SP-1.
      The dK/dV ring accumulates these and delivers them back to the owner.
    """
    assert causal, "contiguous ring attention only supports causal=True"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        kv_rank = (kv_comm.rank - step) % kv_comm.world_size

        if step == 0:
            # Step 0: self-attention with causal mask
            _fa_backward(
                dout, q, k, v, out, softmax_lse,
                dq_buffer, dk_buffer, dv_buffer,
                dropout_p, softmax_scale, causal=True, window_size=window_size,
                deterministic=deterministic,
            )
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if kv_rank < kv_comm.rank:
                # KV from earlier rank: compute non-causal backward
                _fa_backward(
                    dout, q, k, v, out, softmax_lse,
                    dq_buffer, dk_buffer, dv_buffer,
                    dropout_p, softmax_scale, causal=False, window_size=window_size,
                    deterministic=deterministic,
                )
                dq += dq_buffer

                # Wait for previous d_kv communication, then accumulate
                d_kv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv
                dk += dk_buffer
                dv += dv_buffer
            else:
                # KV from later rank: no computation, just pass dk/dv through ring
                d_kv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)

# ---------------------------------------------------------------------------
# Autograd Function for Contiguous Ring (Scheme A)
# ---------------------------------------------------------------------------

class ContiguousRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, window_size, deterministic, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = contiguous_ring_flash_attn_forward(
            group, q, k, v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = contiguous_ring_flash_attn_backward(
            ctx.group,
            dout, q, k, v, out, softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# All-to-all reshuffle utilities (Scheme B)
# ---------------------------------------------------------------------------

def _contiguous_to_zigzag_alltoall(x, sp_group, sp_rank, sp_size, seq_dim=1):
    """Convert contiguous chunk to zigzag layout via all-to-all communication.
    
    Args:
        x: [B, S_local, H, D] tensor in contiguous layout
        sp_group: SP process group
        sp_rank: this rank's SP position
        sp_size: SP world size
        seq_dim: sequence dimension (default 1)
    
    Returns:
        [B, S_local, H, D] tensor in zigzag layout
    
    Note: The token distribution between contiguous and zigzag is NOT uniform.
    For SP=4, S=16: src rank 0 sends [2,2,0,0] tokens to dst ranks.
    We use individual send/recv pairs instead of all_to_all.
    """
    seq_len_local = x.shape[seq_dim]
    full_seq_len = seq_len_local * sp_size

    # Compute zigzag indices
    zigzag_idx = _zigzag_indices(full_seq_len, sp_size)  # [sp_size, chunk_size]

    # My global token range
    my_start = sp_rank * seq_len_local

    # For each destination rank d, find which of my tokens it needs
    # and in what order (matching the zigzag order for that rank)
    send_list = []
    send_counts = []
    for d in range(sp_size):
        wanted = zigzag_idx[d]  # global indices rank d wants
        mask = (wanted >= my_start) & (wanted < my_start + seq_len_local)
        local_idx = (wanted[mask] - my_start).to(x.device)
        chunk = x.index_select(seq_dim, local_idx) if len(local_idx) > 0 else None
        send_list.append(chunk)
        send_counts.append(len(local_idx))

    # Determine what I will receive from each source rank
    recv_counts = []
    my_zigzag = zigzag_idx[sp_rank]  # global indices I should hold in zigzag
    for s in range(sp_size):
        s_start = s * seq_len_local
        s_end = s_start + seq_len_local
        mask = (my_zigzag >= s_start) & (my_zigzag < s_end)
        recv_counts.append(mask.sum().item())

    # Exchange data using point-to-point communication
    # (since all_to_all requires equal sizes which we don't have)
    recv_list = [None] * sp_size
    ops = []
    
    for peer in range(sp_size):
        if peer == sp_rank:
            # Local copy
            if send_counts[peer] > 0:
                recv_list[peer] = send_list[peer]
        else:
            # Send to peer
            if send_counts[peer] > 0:
                ops.append(dist.P2POp(dist.isend, send_list[peer].contiguous(),
                                      dist.get_global_rank(sp_group, peer), group=sp_group))
            # Recv from peer
            if recv_counts[peer] > 0:
                # Determine shape for recv buffer
                recv_shape = list(x.shape)
                recv_shape[seq_dim] = recv_counts[peer]
                recv_buf = torch.empty(recv_shape, dtype=x.dtype, device=x.device)
                recv_list[peer] = recv_buf
                ops.append(dist.P2POp(dist.irecv, recv_buf,
                                      dist.get_global_rank(sp_group, peer), group=sp_group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Concatenate received chunks in source rank order
    # The tokens from each source rank are already in the correct zigzag sub-order
    parts = [recv_list[s] for s in range(sp_size) if recv_list[s] is not None]
    out = torch.cat(parts, dim=seq_dim)
    return out.contiguous()


def _zigzag_to_contiguous_alltoall(x, sp_group, sp_rank, sp_size, seq_dim=1):
    """Convert zigzag layout back to contiguous chunk via all-to-all communication.
    
    Args:
        x: [B, S_local, H, D] tensor in zigzag layout
        sp_group: SP process group
        sp_rank: this rank's SP position
        sp_size: SP world size
        seq_dim: sequence dimension (default 1)
    
    Returns:
        [B, S_local, H, D] tensor in contiguous layout
    """
    seq_len_local = x.shape[seq_dim]
    full_seq_len = seq_len_local * sp_size

    # Compute zigzag indices
    zigzag_idx = _zigzag_indices(full_seq_len, sp_size)  # [sp_size, chunk_size]

    # My zigzag tokens' global positions
    my_zigzag = zigzag_idx[sp_rank]  # [chunk_size]

    # For each destination rank d (which holds contiguous range [d*chunk, (d+1)*chunk)),
    # find which of my zigzag tokens belong to that range
    send_list = []
    send_counts = []
    for d in range(sp_size):
        d_start = d * seq_len_local
        d_end = d_start + seq_len_local
        mask = (my_zigzag >= d_start) & (my_zigzag < d_end)
        local_idx = torch.where(mask)[0].to(x.device)
        chunk = x.index_select(seq_dim, local_idx) if len(local_idx) > 0 else None
        send_list.append(chunk)
        send_counts.append(len(local_idx))

    # Determine what I will receive from each source rank
    recv_counts = []
    my_start = sp_rank * seq_len_local
    for s in range(sp_size):
        s_zigzag = zigzag_idx[s]
        mask = (s_zigzag >= my_start) & (s_zigzag < my_start + seq_len_local)
        recv_counts.append(mask.sum().item())

    # Exchange data using point-to-point communication
    recv_list = [None] * sp_size
    ops = []

    for peer in range(sp_size):
        if peer == sp_rank:
            if send_counts[peer] > 0:
                recv_list[peer] = send_list[peer]
        else:
            if send_counts[peer] > 0:
                ops.append(dist.P2POp(dist.isend, send_list[peer].contiguous(),
                                      dist.get_global_rank(sp_group, peer), group=sp_group))
            if recv_counts[peer] > 0:
                recv_shape = list(x.shape)
                recv_shape[seq_dim] = recv_counts[peer]
                recv_buf = torch.empty(recv_shape, dtype=x.dtype, device=x.device)
                recv_list[peer] = recv_buf
                ops.append(dist.P2POp(dist.irecv, recv_buf,
                                      dist.get_global_rank(sp_group, peer), group=sp_group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Now scatter received tokens into correct contiguous positions
    # From rank s, I receive tokens whose global positions are the subset of
    # zigzag_idx[s] that fall in [my_start, my_start + seq_len_local).
    # I need to place them at local positions (global_pos - my_start).
    out = torch.empty_like(x)

    for s in range(sp_size):
        if recv_list[s] is None:
            continue
        s_zigzag = zigzag_idx[s]
        mask = (s_zigzag >= my_start) & (s_zigzag < my_start + seq_len_local)
        target_local_pos = (s_zigzag[mask] - my_start).to(x.device)
        
        if seq_dim == 1:
            for b in range(x.shape[0]):
                out[b].index_copy_(0, target_local_pos, recv_list[s][b])
        else:
            raise NotImplementedError("Only seq_dim=1 is supported")

    return out.contiguous()


class AllToAllContiguousToZigzag(torch.autograd.Function):
    """Autograd-compatible all-to-all: contiguous → zigzag."""

    @staticmethod
    def forward(ctx, x, sp_group, sp_rank, sp_size, seq_dim):
        ctx.sp_group = sp_group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size
        ctx.seq_dim = seq_dim
        return _contiguous_to_zigzag_alltoall(x, sp_group, sp_rank, sp_size, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward of contiguous→zigzag is zigzag→contiguous
        grad_input = _zigzag_to_contiguous_alltoall(
            grad_output, ctx.sp_group, ctx.sp_rank, ctx.sp_size, ctx.seq_dim
        )
        return grad_input, None, None, None, None


class AllToAllZigzagToContiguous(torch.autograd.Function):
    """Autograd-compatible all-to-all: zigzag → contiguous."""

    @staticmethod
    def forward(ctx, x, sp_group, sp_rank, sp_size, seq_dim):
        ctx.sp_group = sp_group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size
        ctx.seq_dim = seq_dim
        return _zigzag_to_contiguous_alltoall(x, sp_group, sp_rank, sp_size, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward of zigzag→contiguous is contiguous→zigzag
        grad_input = _contiguous_to_zigzag_alltoall(
            grad_output, ctx.sp_group, ctx.sp_rank, ctx.sp_size, ctx.seq_dim
        )
        return grad_input, None, None, None, None


# ---------------------------------------------------------------------------
# Main SP attention interfaces
# ---------------------------------------------------------------------------

@torch.compiler.disable
def sp_flash_attention_contiguous(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sp_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dropout_p: float = 0.0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Scheme A: Contiguous ring flash attention (no zigzag, has load imbalance).

    Each rank holds a contiguous chunk of the sequence. No extra communication
    beyond the ring KV passing. Has causal load imbalance.

    Args:
        q: [B, S_local, H_q, D] - query states (contiguous chunk)
        k: [B, S_local, H_kv, D] - key states (contiguous chunk)
        v: [B, S_local, H_kv, D] - value states (contiguous chunk)
        sp_group: process group for SP communication
        softmax_scale: scaling factor (default: 1/sqrt(D))
        causal: whether to use causal mask (default True)
        dropout_p: dropout probability (default 0.0)
        deterministic: whether to use deterministic backward (default False)

    Returns:
        output: [B, S_local, H_q, D] - attention output
    """
    assert causal, "contiguous ring attention only supports causal=True"

    out = ContiguousRingFlashAttnFunc.apply(
        q, k, v,
        dropout_p,
        softmax_scale,
        causal,
        (-1, -1),
        deterministic,
        sp_group,
    )
    return out


@torch.compiler.disable
def sp_flash_attention_alltoall_zigzag(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sp_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dropout_p: float = 0.0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Scheme B: All-to-all reshuffle + zigzag ring flash attention.

    Each rank holds a contiguous chunk. This function:
    1. All-to-all to convert contiguous → zigzag layout
    2. Run zigzag ring flash attention (load balanced)
    3. All-to-all to convert zigzag → contiguous layout

    Args:
        q: [B, S_local, H_q, D] - query states (contiguous chunk)
        k: [B, S_local, H_kv, D] - key states (contiguous chunk)
        v: [B, S_local, H_kv, D] - value states (contiguous chunk)
        sp_group: process group for SP communication
        softmax_scale: scaling factor (default: 1/sqrt(D))
        causal: whether to use causal mask (default True)
        dropout_p: dropout probability (default 0.0)
        deterministic: whether to use deterministic backward (default False)

    Returns:
        output: [B, S_local, H_q, D] - attention output
    """
    assert causal, "alltoall zigzag ring attention only supports causal=True"

    sp_rank = dist.get_rank(sp_group)
    sp_size = dist.get_world_size(sp_group)

    # Step 1: contiguous → zigzag (all-to-all)
    q_zigzag = AllToAllContiguousToZigzag.apply(q, sp_group, sp_rank, sp_size, 1)
    k_zigzag = AllToAllContiguousToZigzag.apply(k, sp_group, sp_rank, sp_size, 1)
    v_zigzag = AllToAllContiguousToZigzag.apply(v, sp_group, sp_rank, sp_size, 1)

    # Step 2: zigzag ring flash attention
    out_zigzag = ZigZagRingFlashAttnFunc.apply(
        q_zigzag, k_zigzag, v_zigzag,
        dropout_p,
        softmax_scale,
        causal,
        (-1, -1),
        deterministic,
        sp_group,
    )

    # Step 3: zigzag → contiguous (all-to-all)
    out = AllToAllZigzagToContiguous.apply(out_zigzag, sp_group, sp_rank, sp_size, 1)

    return out


# ---------------------------------------------------------------------------
# Zigzag sequence partition utilities
# ---------------------------------------------------------------------------

def _zigzag_indices(seq_len: int, sp_size: int) -> torch.Tensor:
    """Generate the global token indices for zigzag partition.

    For SP=4 and seq_len=16, the zigzag pattern is:
      rank 0: [0, 7, 8, 15]   (first half ascending, second half descending)
      rank 1: [1, 6, 9, 14]
      rank 2: [2, 5, 10, 13]
      rank 3: [3, 4, 11, 12]

    Each rank gets seq_len // sp_size tokens, split into two halves:
      - First half: tokens from the first half of the sequence (ascending)
      - Second half: tokens from the second half of the sequence (descending
        within each rank's allocation, but ascending globally per rank)

    Returns: [sp_size, chunk_size] tensor of global indices.
    """
    assert seq_len % (2 * sp_size) == 0, (
        f"seq_len={seq_len} must be divisible by 2*sp_size={2*sp_size}"
    )
    chunk_size = seq_len // sp_size
    half_chunk = chunk_size // 2

    indices = torch.empty(sp_size, chunk_size, dtype=torch.long)
    for rank in range(sp_size):
        # First half of the sequence (positions 0..seq_len//2-1):
        # rank r gets positions [r*half_chunk, ..., (r+1)*half_chunk - 1]
        first_half_start = rank * half_chunk
        indices[rank, :half_chunk] = torch.arange(
            first_half_start, first_half_start + half_chunk
        )

        # Second half of the sequence (positions seq_len//2..seq_len-1):
        # rank r gets positions in REVERSE order to balance causal load
        # rank 0 gets the LAST half_chunk of the second half
        # rank (sp_size-1) gets the FIRST half_chunk of the second half
        second_half_start = seq_len // 2 + (sp_size - 1 - rank) * half_chunk
        indices[rank, half_chunk:] = torch.arange(
            second_half_start, second_half_start + half_chunk
        )

    return indices


def zigzag_split(
    x: torch.Tensor,
    sp_rank: int,
    sp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Split a full-length tensor into this rank's zigzag chunk.

    Args:
        x: tensor with shape [..., S, ...] where S is at seq_dim.
        sp_rank: this rank's position in the SP group.
        sp_size: total number of SP ranks.
        seq_dim: which dimension is the sequence dimension (default 1).

    Returns:
        Tensor with shape [..., S//sp_size, ...] containing this rank's
        zigzag-selected tokens.
    """
    seq_len = x.shape[seq_dim]
    indices = _zigzag_indices(seq_len, sp_size)
    my_indices = indices[sp_rank].to(x.device)
    return x.index_select(seq_dim, my_indices).contiguous()


def zigzag_unsplit(
    x_local: torch.Tensor,
    sp_rank: int,
    sp_size: int,
    seq_dim: int = 1,
    full_seq_len: Optional[int] = None,
    sp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Gather zigzag chunks from all SP ranks and restore original order.

    Args:
        x_local: this rank's local chunk [..., S_local, ...].
        sp_rank: this rank's position in the SP group.
        sp_size: total number of SP ranks.
        seq_dim: which dimension is the sequence dimension.
        full_seq_len: the original full sequence length (default: S_local * sp_size).
        sp_group: the process group for SP communication.

    Returns:
        Full tensor [..., S, ...] with tokens in original sequential order.
    """
    chunk_size = x_local.shape[seq_dim]
    if full_seq_len is None:
        full_seq_len = chunk_size * sp_size

    # All-gather along the sequence dimension
    gathered_list = [torch.empty_like(x_local) for _ in range(sp_size)]
    dist.all_gather(gathered_list, x_local.contiguous(), group=sp_group)

    # Concatenate: gathered_list[r] has rank r's zigzag chunk
    gathered = torch.cat(gathered_list, dim=seq_dim)

    # Build the inverse permutation to restore original order
    indices = _zigzag_indices(full_seq_len, sp_size)
    flat_indices = indices.reshape(-1).to(x_local.device)
    inv_indices = torch.empty_like(flat_indices)
    inv_indices[flat_indices] = torch.arange(full_seq_len, device=x_local.device)

    return gathered.index_select(seq_dim, inv_indices).contiguous()


# ---------------------------------------------------------------------------
# Main SP attention interface
# ---------------------------------------------------------------------------

@torch.compiler.disable
def sp_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sp_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dropout_p: float = 0.0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Compute attention with zigzag ring flash attention over SP group.

    This function expects that q, k, v have ALREADY been zigzag-partitioned
    (i.e., each rank holds its zigzag chunk of the sequence).

    Args:
        q: [B, S_local, H_q, D] - query states (already zigzag partitioned)
        k: [B, S_local, H_kv, D] - key states (already zigzag partitioned)
        v: [B, S_local, H_kv, D] - value states (already zigzag partitioned)
        sp_group: process group for SP communication
        softmax_scale: scaling factor (default: 1/sqrt(D))
        causal: whether to use causal mask (default True)
        dropout_p: dropout probability (default 0.0)
        deterministic: whether to use deterministic backward (default False)

    Returns:
        output: [B, S_local, H_q, D] - attention output
    """
    assert causal, "zigzag ring attention only supports causal=True"

    out = ZigZagRingFlashAttnFunc.apply(
        q, k, v,
        dropout_p,
        softmax_scale,
        causal,
        (-1, -1),  # window_size
        deterministic,
        sp_group,
    )
    return out


# ---------------------------------------------------------------------------
# Factory function: select SP attention scheme at init time (no forward overhead)
# ---------------------------------------------------------------------------

def get_sp_attention_fn(mode: str = "alltoall_zigzag"):
    """Return the SP attention function for the given mode.

    This should be called at model __init__ time. The returned function is
    stored as an attribute and called directly in forward, avoiding any
    if-else branching at runtime.

    Args:
        mode: One of:
            - "alltoall_zigzag" (default, recommended): Scheme B.
              All-to-all reshuffle + zigzag ring attention. Best for long
              sequences (>=16k). 31% faster than contiguous with only 1-3%
              more memory under gradient checkpointing.
            - "contiguous": Scheme A.
              Contiguous ring attention without zigzag. Simpler but has
              causal load imbalance. Better for short sequences (<8k).
            - "zigzag": Raw zigzag ring attention.
              Expects pre-zigzag-partitioned input. Low-level interface,
              not recommended for direct use in model layers.

    Returns:
        A callable with signature:
            fn(q, k, v, sp_group, causal=True, softmax_scale=None,
               dropout_p=0.0, deterministic=False) -> output

    Example:
        # In model __init__:
        self.sp_attn_fn = get_sp_attention_fn("alltoall_zigzag")

        # In model forward (no branching):
        out = self.sp_attn_fn(q, k, v, sp_group=self.sp_group, causal=True)
    """
    if mode == "alltoall_zigzag":
        return sp_flash_attention_alltoall_zigzag
    elif mode == "contiguous":
        return sp_flash_attention_contiguous
    elif mode == "zigzag":
        return sp_flash_attention
    else:
        raise ValueError(
            f"Unknown SP attention mode: '{mode}'. "
            f"Choose from: 'alltoall_zigzag' (default), 'contiguous', 'zigzag'."
        )
