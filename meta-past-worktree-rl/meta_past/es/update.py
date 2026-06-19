"""ES gradient estimate ported from ES at Scale
(https://github.com/VsonicV/es-fine-tuning-paper,
 see `es_fine-tuning_conciseness.py:283-310`).

Two modes, selected by how the caller passes rewards:

* **one_sided_es_grad**  — ES at Scale's exact kernel. POPULATION_SIZE = N
  one-sided perturbations, rewards z-scored across the population, update
  ``param += α·(1/N)·Σ r_norm_i·ε_i``. Matches their algorithm line for line
  except that we keep per-tensor seed shift (iid noise) instead of the
  layer-correlated default, and we generate noise in fp32 then cast.

* **antithetic_es_grad** — the paired variant we were using. Still exposed
  because some experiments benefit from the 2× variance reduction; ES at
  Scale itself does NOT use antithetic.

Both functions treat ``sigma`` as a scalar. Under ``sigma_mode="rms_relative"``
pass σ_rel (not per-tensor σ_k) — see the docstring on ``one_sided_es_grad``
for why.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .perturb import tensor_seed
from .noise import make_noise


def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "zscore":
        # ES at Scale uses `(r - mean) / (std + 1e-8)` — preserves scale when
        # std is tiny instead of returning zeros (as our earlier variant did
        # when all rewards were identical).
        return (values - float(values.mean())) / (float(values.std()) + 1e-8)
    if mode == "rank":
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        # Centered, unit-variance-ish mapping.
        ranks = ranks / max(len(values) - 1, 1) - 0.5
        return ranks
    raise ValueError(f"Unknown reward normalization mode: {mode!r}")


def one_sided_es_grad(
    rewards: Sequence[float],
    base_seeds: Sequence[int],
    params: Sequence[tuple[str, torch.Tensor]],
    sigma: float,
    normalize: str = "zscore",
    noise_dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """ES at Scale kernel: one-sided perturbations, zscore-normalized rewards.

    The returned tensor ``g_k`` is ready to be added to ``params[k]`` via a
    ``p += α * g_k`` step. With ``sigma_mode="rms_relative"`` pass σ_rel — the
    1/σ_rel factor keeps the update scaled consistently across tensors
    (ĝ_k ∝ RMS_k · g_k when the perturbation was σ_rel · RMS_k · ε).

    Matches ES at Scale's main update formula:
        update_k = (1 / N) * Σ_i r_norm_i * ε_i,k
        param_k += α * update_k
    but with ε generated in fp32 and the per-tensor seed shift from our
    ``tensor_seed``.
    """
    n = len(rewards)
    if n == 0:
        raise ValueError("Need at least one population sample.")
    if len(rewards) != len(base_seeds):
        raise ValueError("rewards and base_seeds must have the same length.")

    r_norm = _normalize(np.asarray(rewards, dtype=np.float64), normalize)
    coeff = 1.0 / n

    grads: list[torch.Tensor] = []
    for k, (_, p) in enumerate(params):
        acc = torch.zeros(p.shape, dtype=noise_dtype, device=p.device)
        for i, s in enumerate(base_seeds):
            eps = make_noise(
                p.shape, tensor_seed(int(s), k), p.device, noise_dtype
            )
            acc.add_(eps, alpha=float(r_norm[i]))
        acc.mul_(coeff)
        grads.append(acc.to(p.dtype))
    return grads


def antithetic_es_grad(
    rewards_plus: Sequence[float],
    rewards_minus: Sequence[float],
    base_seeds: Sequence[int],
    params: Sequence[tuple[str, torch.Tensor]],
    sigma: float,
    normalize: str = "zscore",
    noise_dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Return a per-parameter gradient tensor (same shape/device/dtype as p).

    ``sigma`` is a SCALAR — under ``sigma_mode="rms_relative"`` pass the
    σ_rel used at perturb time (NOT the per-tensor σ_k). Rationale:

      r̂ = (R+ - R-)/2 ≈ σ_rel · Σ_k RMS_k · (ε_k · g_k)
      ĝ_k = (1/(N σ_rel)) Σ_i r̂_i ε_i,k ≈ RMS_k · g_k

    So the returned gradient automatically scales with each tensor's RMS.
    The update ``p += lr · ĝ_k`` then moves every tensor by roughly the
    same *relative* magnitude — which is the property we want from a
    preconditioner, and which was what broke in run #4 (dividing by per-
    tensor σ_k unscaled the RMS weighting, and small-RMS tensors got
    catastrophic raw updates).

    Under ``sigma_mode="absolute"`` (σ_k = σ for all k) both conventions
    collapse to the same formula.

    N = len(base_seeds); total population = 2N (antithetic pairs).
    """
    assert len(rewards_plus) == len(rewards_minus) == len(base_seeds), (
        "rewards_plus, rewards_minus, base_seeds must all have length N."
    )
    n = len(base_seeds)
    if n == 0:
        raise ValueError("Need at least one antithetic pair.")

    diff = (np.asarray(rewards_plus, dtype=np.float64)
            - np.asarray(rewards_minus, dtype=np.float64)) / 2.0
    r_hat = _normalize(diff, normalize)
    coeff = 1.0 / (n * float(sigma)) if sigma > 0 else 0.0

    grads: list[torch.Tensor] = []
    for k, (_, p) in enumerate(params):
        acc = torch.zeros(p.shape, dtype=noise_dtype, device=p.device)
        for i, s in enumerate(base_seeds):
            eps = make_noise(
                p.shape, tensor_seed(int(s), k), p.device, noise_dtype
            )
            acc.add_(eps, alpha=float(r_hat[i]))
        acc.mul_(coeff)
        grads.append(acc.to(p.dtype))
    return grads
