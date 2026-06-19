"""Self-contained vendored copies of the verl kernels we use.

Originally imported from ``verl.trainer.ppo.core_algos`` (advantage estimators)
and ``verl.utils.torch_functional`` (masked reductions). We vendor them here
so the RL stack has no external verl dependency — verl pulls a heavy
RLHF training framework that's unnecessary for our single-machine setup,
and conflicting with our env policy.

Source pinned to verl ``0.6.x`` (commit hash recorded in PR description).
Numerical behavior is identical; only the import path changes.

Functions:
- ``compute_grpo_outcome_advantage``  — GRPO group-z-score (or Dr.GRPO)
- ``compute_rloo_outcome_advantage``  — leave-one-out baseline
- ``compute_reinforce_plus_plus_outcome_advantage`` — discounted-return + whiten
- ``agg_loss``                         — token/seq-mean masked aggregations
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


# --- masked reductions ------------------------------------------------------


def masked_sum(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis=None) -> torch.Tensor:
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask must be 1.")
        if mask_sum == 1:
            raise ValueError("Mask sum == 1; variance is undefined.")
        bessel = mask_sum / (mask_sum - 1)
        variance = variance * bessel
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True,
) -> torch.Tensor:
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


# --- advantage estimators ---------------------------------------------------


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO outcome-advantage: per-group z-score (or Dr.GRPO if no std)."""
    scores = token_level_rewards.sum(dim=-1)

    id2score: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                t = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(t)
                id2std[idx] = torch.std(t)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RLOO leave-one-out baseline."""
    scores = token_level_rewards.sum(dim=-1)
    id2score: dict = defaultdict(list)
    id2mean: dict = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = (
                    scores[i] * response_num / (response_num - 1)
                    - id2mean[index[i]] * response_num / (response_num - 1)
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE++ : discounted return + whitening."""
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device)
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]
        advantages = masked_whiten(returns, response_mask)
        advantages = advantages * response_mask
    return advantages, returns


# --- loss aggregation -------------------------------------------------------


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
) -> torch.Tensor:
    """token-mean / seq-mean-token-sum / seq-mean-token-mean loss aggregation."""
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                loss_scale_factor = loss_mask.shape[-1]
            loss = loss / loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask_count = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask_count + 1e-8)
        seq_mask = (seq_mask_count > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    return loss
