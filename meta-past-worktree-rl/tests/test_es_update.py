"""Sanity tests for antithetic_es_grad on a 2D toy objective.

Objective: f(x) = -||x - x*||^2 for a fixed target x*. ES with antithetic
pairing and z-score normalization should drive x toward x* within a handful
of steps.
"""

from __future__ import annotations

import numpy as np
import torch

from meta_past.es.perturb import InPlacePerturber
from meta_past.es.update import antithetic_es_grad, one_sided_es_grad


def _eval(x: torch.Tensor, target: torch.Tensor) -> float:
    return -float(((x - target) ** 2).sum())


def test_2d_quadratic_converges():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    target = torch.tensor([2.0, -1.5])
    x = torch.zeros(2, requires_grad=False)
    params = [("x", x)]
    sigma = 0.1
    lr = 0.05
    n = 32

    perturber = InPlacePerturber(params, sigma=sigma)

    # Use normalize="none" here so the assertion is about the unbiased gradient
    # estimate, not the sign-SGD-like behavior of zscore normalization. zscore
    # behavior is covered by test_rank_normalization_matches_zscore_sign below.
    for _step in range(100):
        seeds = rng.integers(0, 2**31, size=n).tolist()
        R_plus, R_minus = [], []
        for s in seeds:
            perturber.apply(s, +1)
            R_plus.append(_eval(x, target))
            perturber.restore(s, +1)
            perturber.apply(s, -1)
            R_minus.append(_eval(x, target))
            perturber.restore(s, -1)
        grads = antithetic_es_grad(
            R_plus, R_minus, seeds, params, sigma, normalize="none"
        )
        x.data.add_(lr * grads[0])

    err = float((x - target).norm())
    assert err < 0.25, f"ES did not converge: ||x-x*|| = {err:.4f}"


def test_zero_reward_signal_gives_zero_update():
    # When R_plus == R_minus for every seed, the gradient must be zero
    # regardless of sigma or normalization.
    x = torch.zeros(3)
    params = [("x", x)]
    seeds = [1, 2, 3, 4]
    grads = antithetic_es_grad(
        rewards_plus=[0.0] * 4,
        rewards_minus=[0.0] * 4,
        base_seeds=seeds,
        params=params,
        sigma=1e-2,
        normalize="zscore",
    )
    assert torch.all(grads[0] == 0)


def test_one_sided_converges():
    """ES at Scale-style: N one-sided perturbations, zscore reward, update."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    target = torch.tensor([2.0, -1.5])
    x = torch.zeros(2)
    params = [("x", x)]
    sigma = 0.1
    lr = 0.1
    n = 30

    perturber = InPlacePerturber(params, sigma=sigma)

    for _step in range(200):
        seeds = rng.integers(0, 2**31, size=n).tolist()
        rewards = []
        for s in seeds:
            perturber.apply(s, +1)
            rewards.append(-float(((x - target) ** 2).sum()))
            perturber.restore(s, +1)
        grads = one_sided_es_grad(
            rewards, seeds, params, sigma, normalize="zscore"
        )
        x.data.add_(grads[0], alpha=lr)

    err = float((x - target).norm())
    assert err < 0.3, f"one-sided ES did not converge: ||x-x*||={err:.4f}"


def test_rank_normalization_matches_zscore_sign():
    # With monotonic rewards the two normalizations should agree on the
    # gradient direction, even if magnitudes differ.
    x = torch.zeros(2)
    params = [("x", x)]
    seeds = [10, 20, 30, 40]
    Rp = [1.0, 2.0, 3.0, 4.0]
    Rm = [0.0, 0.0, 0.0, 0.0]

    g_z = antithetic_es_grad(Rp, Rm, seeds, params, sigma=0.1, normalize="zscore")[0]
    g_r = antithetic_es_grad(Rp, Rm, seeds, params, sigma=0.1, normalize="rank")[0]
    # Direction must agree on every coordinate where zscore is non-trivial.
    non_trivial = g_z.abs() > 1e-6
    same_sign = torch.sign(g_z[non_trivial]) == torch.sign(g_r[non_trivial])
    assert bool(same_sign.all())
