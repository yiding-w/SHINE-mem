"""Convert SHINE's batched in-memory loradict to per-context PEFT tensor dicts.

SHINE produces a nested loradict keyed by ``int layer_idx``, with each leaf
holding ``A: [Lb, in, r]`` and ``B: [Lb, r, out]`` (see
``third_party/SHINE/LoraQwen.py:42-77``). We split along ``Lb`` to get one
PEFT-named tensor dict per context, suitable for
``vllm.lora.models.LoRAModel.from_lora_tensors`` (signature at
``vllm/lora/models.py:108`` in 0.7.3).

vLLM's loader **transposes A and B internally** (``from_lora_tensors`` does
``tensor.to(...).t()``). vLLM stores ``lora_a`` with shape ``[in, r]`` and
``lora_b`` with shape ``[r, out]`` internally (verified at
``vllm/lora/lora.py:create_dummy_lora_weights``: ``zeros([input_dim, rank])``
and ``zeros([rank, output_dim])``). PEFT files canonically store
``lora_A: [r, in]`` and ``lora_B: [out, r]`` and rely on vLLM's ``.t()``
to flip them. SHINE's A/B come out of the hypernetwork *not* PEFT-canonical
— they are ``[in, r]`` and ``[r, out]`` directly. So we ``.transpose(-1, -2)``
each leaf before handing it to vLLM, restoring the PEFT-canonical layout
that vLLM's ``.t()`` undoes.

SHINE folds ``sqrt(scale)`` into both A and B at
``LoraQwen.py:88-97``, so the values are already scaled. We set
``lora_alpha = lora_r`` so vLLM's ``vllm_lora_scaling_factor = alpha / r = 1``
and doesn't double-scale (PEFTHelper.__post_init__,
``vllm/lora/peft_helper.py:54-59``).
"""

from __future__ import annotations

from typing import Sequence

import torch


# SHINE keys -> Qwen3 submodule names (PEFT-style).
_ATTN_KEYS = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
_MLP_KEYS = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}

# Default LoRA target modules for Qwen3 (matches SHINE's injection points).
QWEN3_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _peft_key(layer_i: int, submodule: str, proj_name: str, ab: str) -> str:
    """PEFT/HuggingFace-style key, parsed by vLLM's ``parse_fine_tuned_lora_name``.

    See ``vllm/lora/utils.py:parse_fine_tuned_lora_name`` (line 97) for the
    grammar: it expects ``base_model.model.<module_path>.lora_{A,B}.weight``.
    For Qwen3 wrapped by HF as ``Qwen3ForCausalLM``, the LoRA-injected linears
    are at ``model.layers.{i}.{self_attn|mlp}.{proj}``.
    """
    return (
        f"base_model.model.model.layers.{layer_i}."
        f"{submodule}.{proj_name}.lora_{ab}.weight"
    )


def _split_layer(
    layer_i: int,
    submodule: str,
    subdict: dict,
    keymap: dict,
    out_per_b: list[dict[str, torch.Tensor]],
) -> None:
    """Append ``A[b]`` / ``B[b]`` for each context ``b`` and each proj key."""
    for shine_key, proj_name in keymap.items():
        entry = subdict[shine_key]
        A = entry["A"]  # [Lb, in, r]
        B = entry["B"]  # [Lb, r, out]
        C = entry.get("C", None)
        if C is not None:
            raise ValueError(
                f"SHINE loradict layer {layer_i}/{submodule}/{shine_key} has a "
                f"non-None C term; not representable as PEFT LoRA."
            )
        Lb = A.shape[0]
        if B.shape[0] != Lb:
            raise ValueError(
                f"Mismatched Lb: A={tuple(A.shape)} B={tuple(B.shape)} at "
                f"layer {layer_i}/{submodule}/{shine_key}."
            )
        if len(out_per_b) != Lb:
            raise ValueError(
                f"out_per_b prebuilt with len {len(out_per_b)} but loradict has Lb={Lb}."
            )
        for b in range(Lb):
            # SHINE A: [in, r] → PEFT canonical [r, in].
            # SHINE B: [r, out] → PEFT canonical [out, r].
            # vLLM's .t() inside from_lora_tensors flips back to its internal
            # [in, r] / [r, out] convention.
            out_per_b[b][_peft_key(layer_i, submodule, proj_name, "A")] = (
                A[b].transpose(-1, -2).contiguous()
            )
            out_per_b[b][_peft_key(layer_i, submodule, proj_name, "B")] = (
                B[b].transpose(-1, -2).contiguous()
            )


def shine_loradict_to_peft_batch(
    loradict: dict,
) -> list[dict[str, torch.Tensor]]:
    """Split a batched SHINE loradict into ``Lb`` PEFT-named tensor dicts.

    Returns ``list[dict[name, tensor]]`` of length ``Lb``, where each dict
    has 7 * num_layers * 2 entries (q/k/v/o + gate/up/down, A and B).

    **Tensors are NOT detached** — the caller decides. For the vLLM transfer
    path you almost always want ``.detach().cpu()`` before sending; for
    sanity-check / unit-test paths you may want to keep grad.
    """
    if not loradict:
        raise ValueError("Empty loradict.")

    # Peek at the first leaf to learn Lb.
    first_layer = next(iter(loradict.values()))
    first_proj = next(iter(_ATTN_KEYS.values()))
    Lb = int(first_layer["attention"][next(iter(_ATTN_KEYS.keys()))]["A"].shape[0])

    out_per_b: list[dict[str, torch.Tensor]] = [{} for _ in range(Lb)]

    for layer_i, layer_dict in loradict.items():
        if not isinstance(layer_i, int):
            raise TypeError(
                f"Unexpected loradict key {layer_i!r} (type {type(layer_i)})."
            )
        _split_layer(layer_i, "self_attn", layer_dict["attention"],
                     _ATTN_KEYS, out_per_b)
        _split_layer(layer_i, "mlp", layer_dict["mlp"],
                     _MLP_KEYS, out_per_b)

    del first_proj  # silence linter; it was only used to confirm structure
    return out_per_b


def peft_meta_for_qwen3(
    lora_r: int,
    target_modules: Sequence[str] = QWEN3_TARGET_MODULES,
) -> dict:
    """Minimal config dict for ``vllm.lora.peft_helper.PEFTHelper.from_dict``.

    SHINE pre-folds ``sqrt(scale)`` into A and B, so we set
    ``lora_alpha = r`` and vLLM's scaling becomes 1.0. ``bias = "none"``
    matches Qwen3 (no bias on the projection linears).
    """
    return {
        "r": int(lora_r),
        "lora_alpha": int(lora_r),
        "target_modules": list(target_modules),
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "context_length": 0,
    }
