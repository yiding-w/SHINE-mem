"""Smoke tests for ShineHypernet.

These tests require a local Qwen3-8B backbone and the SHINE-ift_mqa_1qa
checkpoint. If either is missing, the tests are skipped rather than failed —
CI without GPU / weights should not red-flag the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from meta_past.shine_adapter import ShineHypernet


_DEFAULT_BACKBONE = os.environ.get(
    "META_PAST_QWEN3_PATH",
    str(Path.home() / "huggingfacemodels" / "Qwen3-8B"),
)
_DEFAULT_CKPT = os.environ.get(
    "META_PAST_SHINE_CKPT",
    str(Path.home() / "huggingfacemodels" / "SHINE-ift_mqa_1qa"),
)


def _weights_available() -> bool:
    return (
        Path(_DEFAULT_BACKBONE).is_dir()
        and Path(_DEFAULT_CKPT).is_dir()
        and (Path(_DEFAULT_CKPT) / "metanetwork.pth").is_file()
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _weights_available(),
    reason="Needs CUDA + Qwen3-8B + SHINE-ift_mqa_1qa weights on disk.",
)


@pytest.fixture(scope="module")
def hypernet() -> ShineHypernet:
    return ShineHypernet(
        ckpt_dir=_DEFAULT_CKPT,
        device="cuda:0",
        backbone=_DEFAULT_BACKBONE,
        lora_r=8,
        metalora_r=128,
    )


def test_load_ckpt(hypernet: ShineHypernet) -> None:
    """Ctor runs, mem token count matches SHINE paper, no backbone leak."""
    assert hypernet.num_mem_token == 148, (
        f"Expected num_mem_token=148 for Qwen3-8B with lora_r=8, "
        f"got {hypernet.num_mem_token}."
    )
    params = hypernet.all_perturbable_params()
    assert len(params) > 0, "all_perturbable_params() returned nothing."

    # Default scope: m2p + metalora; mem_tokens intentionally excluded (see
    # ShineHypernet.all_perturbable_params for why).
    names = {n for n, _ in params}
    assert any(n.startswith("m2p.") for n in names), "No M2P params found."
    assert "mem_tokens" not in names, (
        "mem_tokens should be excluded by default — see adapter docstring."
    )
    assert any(n.startswith("metalora.") for n in names), "No metalora leaves."

    # Opt-in: include_mem_tokens=True brings it back.
    with_mem = hypernet.all_perturbable_params(include_mem_tokens=True)
    assert any(n == "mem_tokens" for n, _ in with_mem), \
        "include_mem_tokens=True should include mem_tokens."

    # No tensor under the backbone besides mem_tokens should be trainable.
    hypernet.assert_only_hypernet_trainable()

    # Every listed tensor must live on CUDA.
    for n, t in params:
        assert t.device.type == "cuda", f"Param {n} is on {t.device}, not cuda."


def test_generate_lora_shape(hypernet: ShineHypernet) -> None:
    """generate_lora returns a nested dict keyed by layer index.

    Each layer's sub-dict holds the LoRA tensors the decoder layer expects;
    we check the top-level structure (one entry per Qwen3 layer) and that the
    leaves are tensors with a rank-dim matching lora_r (= 8).
    """
    tok = hypernet.tokenizer
    ctx = "The fox jumps over the lazy dog."
    enc = tok(
        ctx,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    evidence_ids = enc["input_ids"].to(hypernet.device)
    evidence_mask = enc["attention_mask"].to(hypernet.device)

    lora = hypernet.generate_lora(evidence_ids, evidence_mask)
    assert isinstance(lora, dict), f"Expected dict, got {type(lora)}."

    num_hidden = hypernet.metanetwork.metamodel.config.num_hidden_layers
    assert set(lora.keys()) == set(range(num_hidden)), (
        f"Expected keys {set(range(num_hidden))}, got {set(lora.keys())}."
    )

    # Walk to a leaf tensor and confirm lora_r appears as one of its dims.
    def _leaf_tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _leaf_tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _leaf_tensors(v)

    leaves = list(_leaf_tensors(lora))
    assert leaves, "LoRA dict had no tensor leaves."
    # At least one leaf should expose lora_r on a dimension.
    assert any(hypernet.lora_r in tuple(t.shape) for t in leaves), (
        f"No leaf tensor exposes lora_r={hypernet.lora_r}. "
        f"Sample shapes: {[tuple(t.shape) for t in leaves[:4]]}"
    )
