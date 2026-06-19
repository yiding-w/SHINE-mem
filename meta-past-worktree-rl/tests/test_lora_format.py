"""Unit tests for batched SHINE loradict -> PEFT-name tensor dicts.

Builds a synthetic Lb=B loradict and verifies ``shine_loradict_to_peft_batch``
produces ``B`` independent dicts with the expected PEFT keys, the correct
b-th slice (no transpose — vLLM transposes internally), and the right
``peft_meta_for_qwen3`` config.
"""

from __future__ import annotations

import pytest
import torch

from meta_past.rl.lora_format import (
    QWEN3_TARGET_MODULES,
    peft_meta_for_qwen3,
    shine_loradict_to_peft_batch,
)


def _leaf(in_f, out_f, r, Lb):
    torch.manual_seed(in_f * 1000 + out_f + Lb * 7)
    A = torch.randn(Lb, in_f, r)
    B = torch.randn(Lb, r, out_f)
    return {"A": A, "B": B, "C": None}


def _make_loradict(num_layers, hidden, head_dim, n_q_heads, n_kv_heads,
                   intermediate, r, Lb):
    loradict = {}
    for i in range(num_layers):
        loradict[i] = {
            "attention": {
                "q": _leaf(hidden, n_q_heads * head_dim, r, Lb),
                "k": _leaf(hidden, n_kv_heads * head_dim, r, Lb),
                "v": _leaf(hidden, n_kv_heads * head_dim, r, Lb),
                "o": _leaf(n_q_heads * head_dim, hidden, r, Lb),
            },
            "mlp": {
                "gate": _leaf(hidden, intermediate, r, Lb),
                "up":   _leaf(hidden, intermediate, r, Lb),
                "down": _leaf(intermediate, hidden, r, Lb),
            },
        }
    return loradict


def test_split_lb3_produces_3_dicts():
    r = 4
    Lb = 3
    loradict = _make_loradict(2, 32, 8, 4, 2, 64, r, Lb)
    out = shine_loradict_to_peft_batch(loradict)

    assert len(out) == Lb
    # 2 layers × 7 projections × 2 (A, B) = 28 keys per dict.
    for d in out:
        assert len(d) == 2 * 7 * 2

    # Spot-check: layer-0 q_proj entries on each context are the b-th slice
    # **transposed** to PEFT canonical [r, in] / [out, r]. vLLM transposes
    # them back to its internal [in, r] / [r, out] when loading.
    A_orig = loradict[0]["attention"]["q"]["A"]   # [Lb, in, r]
    B_orig = loradict[0]["attention"]["q"]["B"]   # [Lb, r, out]
    for b in range(Lb):
        kA = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        kB = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        assert out[b][kA].shape == (r, A_orig.shape[1])         # [r, in]
        assert out[b][kB].shape == (B_orig.shape[2], r)         # [out, r]
        assert torch.equal(out[b][kA], A_orig[b].transpose(-1, -2))
        assert torch.equal(out[b][kB], B_orig[b].transpose(-1, -2))
        # And the b'-th slice is genuinely *not* the same as another b'.
        if b > 0:
            assert not torch.equal(out[b][kA], out[0][kA])


def test_keys_cover_all_target_modules():
    r = 4
    loradict = _make_loradict(1, 32, 8, 4, 2, 64, r, Lb=1)
    [d] = shine_loradict_to_peft_batch(loradict)
    found = set()
    for k in d.keys():
        # base_model.model.model.layers.{i}.{submodule}.{proj}.lora_{X}.weight
        parts = k.split(".")
        proj = parts[-3]
        found.add(proj)
    assert found == set(QWEN3_TARGET_MODULES)


def test_rejects_nonzero_C():
    r = 4
    loradict = _make_loradict(1, 32, 8, 4, 2, 64, r, Lb=2)
    loradict[0]["attention"]["q"]["C"] = torch.randn(2, 32)
    with pytest.raises(ValueError, match="non-None C term"):
        shine_loradict_to_peft_batch(loradict)


def test_rejects_mismatched_lb():
    r = 4
    loradict = _make_loradict(1, 32, 8, 4, 2, 64, r, Lb=2)
    # Tamper: A is Lb=2, B is Lb=3.
    loradict[0]["attention"]["q"]["B"] = torch.randn(3, r, 32)
    with pytest.raises(ValueError, match="Mismatched Lb"):
        shine_loradict_to_peft_batch(loradict)


def test_peft_meta_for_qwen3():
    meta = peft_meta_for_qwen3(lora_r=8)
    assert meta["r"] == 8
    assert meta["lora_alpha"] == 8        # alpha == r ⇒ vLLM scaling 1.0
    assert meta["bias"] == "none"
    assert meta["use_rslora"] is False
    assert meta["use_dora"] is False
    assert sorted(meta["target_modules"]) == sorted(QWEN3_TARGET_MODULES)


def test_loradict_slice_and_concat_roundtrip():
    """split into chunks then concat → bit-exact original."""
    from meta_past.shine_adapter import (
        _concat_loradicts_lb,
        _slice_loradict_lb,
    )

    r, Lb = 4, 6
    loradict = _make_loradict(2, 32, 8, 4, 2, 64, r, Lb)

    # Split into 3 chunks of 2 contexts.
    chunks = [
        _slice_loradict_lb(loradict, 0, 2),
        _slice_loradict_lb(loradict, 2, 4),
        _slice_loradict_lb(loradict, 4, 6),
    ]
    for c in chunks:
        first_proj_A = c[0]["attention"]["q"]["A"]
        assert first_proj_A.shape[0] == 2

    rejoined = _concat_loradicts_lb(chunks)
    # Compare leaf-by-leaf against the original.
    for layer_i in loradict.keys():
        for sub in ("attention", "mlp"):
            for proj_key in loradict[layer_i][sub].keys():
                for ab in ("A", "B"):
                    assert torch.equal(
                        rejoined[layer_i][sub][proj_key][ab],
                        loradict[layer_i][sub][proj_key][ab],
                    )
