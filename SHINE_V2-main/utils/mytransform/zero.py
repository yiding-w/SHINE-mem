"""
Zero W-Transform: returns None, effectively disabling wdict injection.

When used as w_transform_context, this is equivalent to the old
``dynamic_hypernetwork: false`` behavior — the context forward pass
will not see any accumulated wdict, so the hypernetwork generation
is unaffected by the accumulated state.
"""
from __future__ import annotations

import torch.nn as nn
from typing import Optional


class ZeroTransform(nn.Module):
    """Zero transform — returns None, signaling that wdict should not be injected.

    This module has no parameters. When the layer wrapper receives None
    from the transform, it skips wdict injection entirely.
    """

    def __init__(self):
        super().__init__()

    def forward(self, layer_wdict: dict, layer_idx: int) -> Optional[dict]:
        """Always returns None, disabling wdict injection for this phase."""
        return None
