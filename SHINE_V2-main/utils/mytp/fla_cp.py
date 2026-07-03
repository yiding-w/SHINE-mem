"""
FLA Context Parallel (CP) utilities for Sequence Parallelism.

This module wraps FLA's native CP support for use in our SP training pipeline.
FLA's CP handles:
  - Conv1d halo exchange (forward + backward)
  - Recurrent state merge across SP ranks (forward + backward)

We only need to construct the FLACPContext correctly; FLA handles all
communication internally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from fla.ops.cp.context import FLACPContext, build_cp_context


__all__ = ["build_sp_cp_context"]


def build_sp_cp_context(
    seq_len_local: int,
    sp_group: ProcessGroup,
    conv1d_kernel_size: int = 4,
    device: Optional[torch.device] = None,
) -> FLACPContext:
    """Build an FLACPContext for single-sequence SP training.

    Our training scenario: batch_size=1, one full sequence is evenly split
    across all SP ranks. Each rank holds a contiguous chunk of length
    ``seq_len_local``.

    Args:
        seq_len_local: Length of the local sequence chunk on this rank.
        sp_group: The SP process group (ranks that share the same TP rank
            but different SP positions).
        conv1d_kernel_size: Kernel size for causal conv1d (default 4 for
            Qwen3.6-27B). This tells FLA how many tokens to exchange as
            halo for conv1d.
        device: Device for the cu_seqlens tensor.

    Returns:
        FLACPContext that can be passed to both ``fla.modules.conv.causal_conv1d``
        and ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.
    """
    sp_world = dist.get_world_size(sp_group)
    seq_len_full = seq_len_local * sp_world

    # cu_seqlens = [0, S_full] means one single sequence of length S_full
    cu_seqlens = torch.tensor([0, seq_len_full], dtype=torch.long, device=device)

    return build_cp_context(
        cu_seqlens=cu_seqlens,
        group=sp_group,
        conv1d_kernel_size=conv1d_kernel_size,
    )
