"""In-place +/- sigma*epsilon perturbation with exact restore.

Each tensor gets its own seed derived from the outer ``base_seed`` via a stable
shift, so noise is IID across tensors (avoiding the layer-correlated noise
quirk in ES at Scale's default variant — see their `_iid.py`).

Two sigma modes:

* ``"absolute"`` — the classical σ·ε: all tensors share one σ. Use when you've
  verified that every tensor operates at roughly the same scale.
* ``"rms_relative"`` — σ_k = σ · RMS(p_k), snapshotted at construction time.
  Tensors keep the same *relative* perturbation regardless of element scale.
  Required when perturbing heterogeneous scopes (the SHINE m2p+metalora mix
  has RMS spanning 5e-3 to 1.0; uniform σ·ε gives small-RMS tensors hugely
  disproportionate kicks, which accumulated destroys the model in ~5 ES
  steps — see scripts/debug_m2p_fine.py and the collapse in run #3).

The RMS snapshot is taken once; using a live RMS inside apply() would make
restore() non-invertible because post-perturbation RMS differs from
pre-perturbation RMS.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch

from .noise import make_noise


_SEED_MULT = 1_000_003  # prime; overflow-safe mask applied below
_SEED_MASK = 0x7FFFFFFF


def tensor_seed(base_seed: int, tensor_idx: int) -> int:
    """Stable per-tensor seed shift, bounded to 31 bits."""
    return (int(base_seed) * _SEED_MULT + tensor_idx) & _SEED_MASK


def _tensor_rms(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 1.0
    return float(t.detach().float().pow(2).mean().sqrt().item())


class InPlacePerturber:
    """Applies per-tensor sigma to a list of (name, tensor) pairs, reversibly.

    Caller owns the tensors. The perturber only mutates in place, so the same
    param references are valid across apply/restore cycles.

    Per-tensor sigma layout:
      sigma_mode="absolute"      -> sigma_k = sigma
      sigma_mode="rms_relative"  -> sigma_k = sigma * max(RMS(p_k), rms_floor)
    """

    def __init__(
        self,
        params: Sequence[tuple[str, torch.Tensor]],
        sigma: float,
        noise_dtype: torch.dtype = torch.float32,
        sigma_mode: Literal["absolute", "rms_relative"] = "absolute",
        rms_floor: float = 1e-3,
    ):
        self.params = list(params)
        self.sigma = float(sigma)
        self.noise_dtype = noise_dtype
        self.sigma_mode = sigma_mode
        self.rms_floor = float(rms_floor)

        # Snapshot per-tensor sigma at construction. This is what makes
        # apply/restore exactly invertible even in rms_relative mode.
        if sigma_mode == "absolute":
            self._sigma_k: list[float] = [self.sigma] * len(self.params)
        elif sigma_mode == "rms_relative":
            self._sigma_k = [
                self.sigma * max(_tensor_rms(p), self.rms_floor)
                for _, p in self.params
            ]
        else:
            raise ValueError(f"Unknown sigma_mode {sigma_mode!r}")

    def sigma_k(self, k: int) -> float:
        """Return the effective sigma used for the k-th tensor."""
        return self._sigma_k[k]

    def _step(self, base_seed: int, sign: int) -> None:
        for k, (_, p) in enumerate(self.params):
            eps = make_noise(
                p.shape,
                tensor_seed(base_seed, k),
                p.device,
                self.noise_dtype,
            )
            p.data.add_((sign * self._sigma_k[k]) * eps.to(p.dtype))

    def apply(self, base_seed: int, sign: int) -> None:
        self._step(base_seed, sign)

    def restore(self, base_seed: int, sign: int) -> None:
        self._step(base_seed, -sign)
