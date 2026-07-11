from __future__ import annotations

from typing import Optional

import torch

from utils.mytp.tp_linear import ColwiseLoraLinear, RowwiseLoraLinear


def split_wrapped_loradict(loradict, nograd_wdict=None):
    """Allow v1 callers to pass {"grad": loradict, "state": wdict} leaves.

    The v2 TP linear API normally receives trainable LoRA and detached W as
    separate arguments. V1 model forwards only one loradict argument, so the
    V1 adapter wraps them at the leaf and these linears unwrap them.
    """
    if isinstance(loradict, dict) and ("grad" in loradict or "state" in loradict):
        grad = loradict.get("grad", None)
        state = loradict.get("state", None)
        if nograd_wdict is None:
            nograd_wdict = state
        return grad, nograd_wdict
    return loradict, nograd_wdict


def _cast_leaf_to_input(leaf, input):
    if not isinstance(leaf, dict):
        return leaf
    result = {}
    changed = False
    for key, value in leaf.items():
        if torch.is_tensor(value) and (value.dtype != input.dtype or value.device != input.device):
            result[key] = value.to(device=input.device, dtype=input.dtype)
            changed = True
        else:
            result[key] = value
    return result if changed else leaf


class V1ColwiseLoraLinear(ColwiseLoraLinear):
    def init_lora_dict(self, r: int, scale: float, device, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            dtype = self.weight.dtype
        return super().init_lora_dict(r, scale, device, dtype)

    def forward(self, input, loradict=None, nograd_loradict=None, nograd_wdict=None):
        loradict, nograd_wdict = split_wrapped_loradict(loradict, nograd_wdict)
        loradict = _cast_leaf_to_input(loradict, input)
        nograd_loradict = _cast_leaf_to_input(nograd_loradict, input)
        nograd_wdict = _cast_leaf_to_input(nograd_wdict, input)
        return super().forward(
            input,
            loradict=loradict,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
        )

    def divide_idx(self, r: int, idx_start: int):
        if self.bias is not None:
            raise NotImplementedError("V1 divide_idx currently assumes bias=False")
        a_numel = self.in_features * r
        b_numel = self.out_features_total * r
        return [idx_start, idx_start + a_numel], idx_start + a_numel + b_numel


class V1RowwiseLoraLinear(RowwiseLoraLinear):
    def init_lora_dict(self, r: int, scale: float, device, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            dtype = self.weight.dtype
        return super().init_lora_dict(r, scale, device, dtype)

    def forward(self, input, loradict=None, nograd_loradict=None, nograd_wdict=None):
        loradict, nograd_wdict = split_wrapped_loradict(loradict, nograd_wdict)
        loradict = _cast_leaf_to_input(loradict, input)
        nograd_loradict = _cast_leaf_to_input(nograd_loradict, input)
        nograd_wdict = _cast_leaf_to_input(nograd_wdict, input)
        return super().forward(
            input,
            loradict=loradict,
            nograd_loradict=nograd_loradict,
            nograd_wdict=nograd_wdict,
        )

    def divide_idx(self, r: int, idx_start: int):
        if self.bias is not None:
            raise NotImplementedError("V1 divide_idx currently assumes bias=False")
        a_numel = self.in_features_total * r
        b_numel = self.out_features * r
        return [idx_start, idx_start + a_numel], idx_start + a_numel + b_numel
