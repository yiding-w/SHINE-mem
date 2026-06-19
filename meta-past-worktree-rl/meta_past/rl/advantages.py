"""Advantage computation — uses our vendored verl kernels.

Our rollouts produce one scalar reward per sample. The kernels expect
``token_level_rewards`` of shape ``[bs, T]`` with the reward placed at
some token position and the advantage broadcast back across the response.
We wrap that convention here so the trainer only sees ``(rewards[bs],
response_mask[bs, T], group_ids[bs]) -> advantages[bs, T]``.

Source: :mod:`meta_past.rl._verl_kernels` (vendored from verl 0.6.x).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import _verl_kernels as core_algos


def compute_advantages(
    rewards: torch.Tensor,            # [bs] — scalar reward per sample
    response_mask: torch.Tensor,      # [bs, T] — 1 on response tokens
    group_ids: Sequence[int] | np.ndarray | torch.Tensor,
    kind: str = "grpo",               # "grpo" | "rloo" | "reinforce_plus_plus"
    norm_adv_by_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return verl-computed advantages, shape ``[bs, T]``.

    ``kind``:
      - "grpo"  : ``(r - mean_g) / (std_g + eps)`` if ``norm_adv_by_std`` else centered only.
      - "rloo"  : leave-one-out baseline per group.
      - "reinforce_plus_plus": verl's R++ variant.
    """
    if rewards.dim() != 1:
        raise ValueError(f"rewards must be 1-D [bs]; got {tuple(rewards.shape)}")
    if response_mask.dim() != 2 or response_mask.shape[0] != rewards.shape[0]:
        raise ValueError(
            f"response_mask must be [bs, T] matching rewards[bs]; "
            f"got rewards={tuple(rewards.shape)} mask={tuple(response_mask.shape)}"
        )

    bs, T = response_mask.shape
    # Build token_level_rewards: scalar at position 0, zeros elsewhere.
    # verl's GRPO/RLOO kernels do ``scores = token_level_rewards.sum(-1)`` so
    # placement is irrelevant as long as it sums to the scalar.
    token_level_rewards = torch.zeros_like(response_mask, dtype=rewards.dtype)
    token_level_rewards[:, 0] = rewards

    if not isinstance(group_ids, np.ndarray):
        group_ids = np.asarray(
            group_ids.tolist() if isinstance(group_ids, torch.Tensor) else list(group_ids)
        )

    if kind == "grpo":
        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask.to(token_level_rewards.dtype),
            index=group_ids,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std,
        )
    elif kind == "rloo":
        advantages, _ = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask.to(token_level_rewards.dtype),
            index=group_ids,
            epsilon=epsilon,
        )
    elif kind == "reinforce_plus_plus":
        advantages, _ = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask.to(token_level_rewards.dtype),
            gamma=1.0,
        )
    else:
        raise ValueError(
            f"Unknown advantage kind {kind!r}. "
            "Supported: grpo | rloo | reinforce_plus_plus."
        )
    return advantages
