"""
Sequence Parallelism (SP) utility functions.

These functions handle the mechanics of splitting sequences across SP ranks
and correctly computing loss at boundaries.

Key invariants:
  - Total sequence length must be divisible by sp_world.
  - Each SP rank holds a contiguous chunk of the full sequence.
  - position_ids are global (not local), ensuring correct RoPE.
  - Loss at SP boundaries is handled by exchanging the boundary label
    from the next rank (for next-token prediction shift).
  - Loss is all-reduced across SP ranks to get the correct mean.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


__all__ = [
    "sp_pad_to_divisible",
    "sp_split_sequence",
    "sp_make_position_ids",
    "sp_exchange_boundary_labels",
    "sp_reduce_loss",
]


def sp_pad_to_divisible(
    tensor: torch.Tensor,
    sp_world: int,
    pad_value: int = 0,
) -> Tuple[torch.Tensor, int]:
    """Pad tensor's seq dimension (dim=1) to be divisible by 2*sp_world.

    Zigzag ring attention requires the total sequence length to be divisible
    by 2*sp_world. This function pads the sequence at the end with pad_value
    to satisfy that constraint.

    Args:
        tensor: [B, S, ...] tensor to pad along dim=1.
        sp_world: Total number of SP ranks.
        pad_value: Value to use for padding (0 for input_ids, -100 for labels).

    Returns:
        Tuple of (padded_tensor, original_seq_len).
        If no padding is needed, returns (tensor, S) without copying.
    """
    S = tensor.shape[1]
    divisor = 2 * sp_world
    remainder = S % divisor
    if remainder == 0:
        return tensor, S
    pad_len = divisor - remainder
    # Build padding shape: same as tensor but with pad_len in dim=1
    pad_shape = list(tensor.shape)
    pad_shape[1] = pad_len
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=1), S


def sp_split_sequence(
    input_ids: torch.LongTensor,
    sp_rank: int,
    sp_world: int,
) -> torch.LongTensor:
    """Split input_ids along the sequence dimension for this SP rank.

    Args:
        input_ids: [B, S_full] full sequence token IDs.
        sp_rank: This rank's position in the SP group (0-indexed).
        sp_world: Total number of SP ranks.

    Returns:
        [B, S_local] local chunk of input_ids for this rank.

    Raises:
        ValueError: If S_full is not divisible by sp_world.
    """
    B, S_full = input_ids.shape
    if S_full % sp_world != 0:
        raise ValueError(
            f"Sequence length {S_full} is not divisible by sp_world={sp_world}. "
            f"SP requires total sequence length to be evenly divisible."
        )
    S_local = S_full // sp_world
    start = sp_rank * S_local
    end = start + S_local
    return input_ids[:, start:end].contiguous()


def sp_make_position_ids(
    S_local: int,
    sp_rank: int,
    sp_world: int,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """Create global position_ids for this SP rank's local chunk.

    For SP rank i with S_local tokens, position_ids are:
        [i * S_local, i * S_local + 1, ..., (i+1) * S_local - 1]

    This ensures RoPE uses the correct global positions.

    Args:
        S_local: Local sequence length on this rank.
        sp_rank: This rank's position in the SP group.
        sp_world: Total number of SP ranks.
        batch_size: Batch size (typically 1 for SP training).
        device: Device for the output tensor.

    Returns:
        position_ids: [B, S_local] with global positions.
    """
    offset = sp_rank * S_local
    pos = torch.arange(offset, offset + S_local, device=device)
    # Expand to [B, S_local]
    return pos.unsqueeze(0).expand(batch_size, -1)


@torch.compiler.disable
def sp_exchange_boundary_labels(
    labels: torch.LongTensor,
    sp_group: ProcessGroup,
) -> torch.LongTensor:
    """Exchange boundary labels for correct next-token prediction loss.

    In next-token prediction, the loss for position i uses label[i+1].
    At SP boundaries, rank k's last position needs the label from rank k+1's
    first position. This function appends that boundary label.

    After this call, the returned labels tensor has shape [B, S_local + 1]:
      - labels[:, 0:S_local] are the original local labels
      - labels[:, S_local] is the first label from the next rank
        (or -100 for the last rank, since there's no next token)

    The caller should then do:
      shift_labels = returned_labels[:, 1:]  # [B, S_local]
    which gives the correct shifted labels for all positions including
    the boundary.

    Args:
        labels: [B, S_local] local labels for this rank.
        sp_group: SP process group.

    Returns:
        [B, S_local + 1] labels with boundary label appended.
    """
    sp_rank = dist.get_rank(sp_group)
    sp_world = dist.get_world_size(sp_group)
    B = labels.shape[0]
    device = labels.device

    # The boundary label we need: first label of the next rank
    # We send our first label to the previous rank, and receive from next rank.
    send_buf = labels[:, 0].contiguous()  # [B] - our first label (sent to prev rank)
    recv_buf = torch.full((B,), -100, dtype=labels.dtype, device=device)  # default: ignore

    # Non-blocking send/recv for pipeline efficiency
    if sp_world == 1:
        # No SP, just append -100
        return torch.cat([labels, recv_buf.unsqueeze(1)], dim=1)

    # Use all_gather to exchange boundary labels (ensures all ranks participate
    # in the same collective, avoiding NCCL ordering issues with asymmetric P2P).
    # Each rank contributes its first label; we then pick the next rank's value.
    gathered = [torch.empty_like(send_buf) for _ in range(sp_world)]
    dist.all_gather(gathered, send_buf, group=sp_group)

    if sp_rank < sp_world - 1:
        recv_buf = gathered[sp_rank + 1]
    # else: recv_buf stays as -100 (last rank has no next)

    # Append boundary label
    return torch.cat([labels, recv_buf.unsqueeze(1)], dim=1)


@torch.compiler.disable
def sp_reduce_loss(
    local_loss_sum: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    sp_group: ProcessGroup,
) -> torch.Tensor:
    """Compute global mean loss across SP ranks (autograd-compatible).

    Each SP rank has a local loss sum (with grad) and valid token count (no grad).
    The global mean is:
        global_loss = sum_all_ranks(local_sum) / sum_all_ranks(local_count)

    For correct backprop, we only all-reduce the count (detached), then each rank
    computes local_sum / global_count. The backward pass naturally produces
    correct gradients: each rank's grad is scaled by 1/global_count, and when
    summed across SP ranks (via gradient all-reduce), gives the correct total.

    Args:
        local_loss_sum: Scalar local SUM of losses on this rank (reduction="sum").
            Must be differentiable (requires_grad=True in the computation graph).
        num_valid_tokens: Number of valid (non-ignored) tokens on this rank.
            Does not need gradients.
        sp_group: SP process group.

    Returns:
        Scalar loss = local_sum / global_count. Each rank returns a DIFFERENT
        value (its own local_sum divided by the shared global_count). After
        backward + gradient all-reduce across SP, the parameter gradients will
        be correct.
    """
    if sp_group is None or dist.get_world_size(sp_group) <= 1:
        return local_loss_sum / num_valid_tokens.float().clamp(min=1)

    # All-reduce only the count (no gradient needed)
    global_count = num_valid_tokens.float().clone().detach()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=sp_group)

    # Each rank computes local_sum / global_count (differentiable w.r.t. local_sum)
    return local_loss_sum / global_count.clamp(min=1)
