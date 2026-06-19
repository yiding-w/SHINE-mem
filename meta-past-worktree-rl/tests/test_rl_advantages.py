"""Tests for verl-backed advantage + loss wrappers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from meta_past.rl.advantages import compute_advantages
from meta_past.rl.losses import policy_loss


def _mask_like(rewards: torch.Tensor, T: int = 3) -> torch.Tensor:
    return torch.ones(rewards.shape[0], T, dtype=torch.float32)


def test_grpo_normalizes_within_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = _mask_like(rewards)
    group_ids = np.array([0, 0, 0, 0])

    adv = compute_advantages(rewards, mask, group_ids, kind="grpo")
    # Advantages are broadcast token-wise → same scalar on every masked position.
    scalar = adv[:, 0]
    # verl uses torch.std() with default unbiased=True (Bessel-corrected).
    mean, std = rewards.mean(), rewards.std()
    expected = (rewards - mean) / (std + 1e-6)
    assert torch.allclose(scalar, expected, atol=1e-4)
    # Same scalar replicated across T.
    assert torch.allclose(adv, scalar.unsqueeze(-1) * mask)


def test_grpo_groups_independent():
    rewards = torch.tensor([1.0, 3.0, 10.0, 20.0])
    mask = _mask_like(rewards)
    group_ids = np.array([0, 0, 1, 1])

    adv = compute_advantages(rewards, mask, group_ids, kind="grpo",
                             norm_adv_by_std=False)
    # Dr.GRPO (norm_adv_by_std=False) = centering only.
    # Group 0: mean=2, scalars=[-1, 1]. Group 1: mean=15, scalars=[-5, 5].
    assert torch.allclose(adv[0, 0], torch.tensor(-1.0), atol=1e-4)
    assert torch.allclose(adv[1, 0], torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(adv[2, 0], torch.tensor(-5.0), atol=1e-4)
    assert torch.allclose(adv[3, 0], torch.tensor(5.0), atol=1e-4)


def test_rloo_leave_one_out():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = _mask_like(rewards)
    group_ids = np.array([0, 0, 0, 0])

    adv = compute_advantages(rewards, mask, group_ids, kind="rloo")
    # verl's RLOO: A_i = r_i - mean(r_{j != i}).
    # i=0: 1 - 0 = 1; i=1..3: 0 - 1/3.
    scalars = adv[:, 0]
    expected = torch.tensor([1.0, -1/3, -1/3, -1/3])
    assert torch.allclose(scalars, expected, atol=1e-4)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        compute_advantages(torch.tensor([1.0, 2.0]), _mask_like(torch.tensor([1.0, 2.0])),
                           np.array([0, 0]), kind="not-a-kind")


def test_policy_loss_gradient_flow():
    torch.manual_seed(0)
    N, T = 3, 5
    log_prob = torch.randn(N, T, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    advantages = torch.tensor([[1.0], [-2.0], [0.5]]) * mask  # broadcast

    loss = policy_loss(log_prob, advantages, mask, loss_agg_mode="token-mean")
    loss.backward()

    # d/d log_prob of -(A · log_prob) summed & normalized by total tokens.
    total_tokens = mask.sum()
    expected_grad = -advantages * mask / total_tokens
    assert torch.allclose(log_prob.grad, expected_grad, atol=1e-6)


def test_policy_loss_rejects_shape_mismatch():
    lp = torch.zeros(2, 3)
    adv = torch.zeros(2, 4)
    mask = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        policy_loss(lp, adv, mask)


def test_verl_kernels_wired_through():
    """Make sure both advantages and losses route through the vendored kernels."""
    import meta_past.rl.advantages as A
    import meta_past.rl.losses as L
    import inspect

    src_a = inspect.getsource(A)
    src_l = inspect.getsource(L)
    assert "_verl_kernels" in src_a
    assert "_verl_kernels" in src_l
