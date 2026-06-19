"""Policy loss for on-policy REINFORCE / GRPO.

We don't use verl's ``compute_policy_loss_vanilla`` because it (a) requires a
verl ``ActorConfig`` with clip ratios, (b) assumes an importance-weighted PPO
ratio (``ratio = exp(log_prob - old_log_prob)``), and (c) needs a
``global_batch_info`` dict for FSDP-aware aggregation we don't have.

On-policy REINFORCE is the ratio=1 special case. The math reduces to
``loss = -advantage * log_prob`` masked and aggregated. We use our vendored
``agg_loss`` (:mod:`meta_past.rl._verl_kernels`) so the aggregation mode
(``token-mean`` / ``seq-mean-token-sum`` / ``seq-mean-token-mean``) matches
what verl's trainer would produce.
"""

from __future__ import annotations

import torch

from ._verl_kernels import agg_loss


def policy_loss(
    log_prob: torch.Tensor,       # [bs, T] current-policy logπ
    advantages: torch.Tensor,      # [bs, T] broadcast advantages
    response_mask: torch.Tensor,   # [bs, T] 1 on response positions
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Scalar REINFORCE loss: ``agg(−A · logπ, mask)``.

    On-policy: no ratio, no clipping.
    """
    if log_prob.shape != advantages.shape or log_prob.shape != response_mask.shape:
        raise ValueError(
            f"shape mismatch: log_prob={tuple(log_prob.shape)} "
            f"advantages={tuple(advantages.shape)} mask={tuple(response_mask.shape)}"
        )
    loss_mat = -advantages * log_prob
    return agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask.to(loss_mat.dtype),
        loss_agg_mode=loss_agg_mode,
    )
