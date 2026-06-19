"""Seed-based deterministic Gaussian noise for ES perturbation.

Defaults to fp32 regardless of the target parameter dtype: many SHINE params
live in bf16/fp16, and bf16 at sigma=1e-3 quantizes tiny tensors (e.g.
``mem_tokens`` with magnitudes ~1e-3) to zero. We cast fp32 noise to the param
dtype only at the add step in ``perturb.py``.
"""

from __future__ import annotations

import torch


def make_noise(
    shape: tuple[int, ...] | torch.Size,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=gen, dtype=dtype, device=device)
