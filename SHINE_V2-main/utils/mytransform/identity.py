"""
Identity W-Transform: passes wdict through unchanged.

This is the default transform when no learned transformation is needed.
It has no parameters and zero computational overhead.
"""
from __future__ import annotations

import torch.nn as nn
from typing import Optional


class IdentityTransform(nn.Module):
    """Identity transform — returns wdict unchanged.

    This module exists so that the interface is uniform: callers always
    invoke ``transform(layer_wdict, layer_idx)`` regardless of whether
    a learned transform is active.
    """

    def __init__(self):
        super().__init__()

    def forward(self, layer_wdict: dict, layer_idx: int) -> dict:
        """Pass through unchanged."""
        return layer_wdict
