import pytest


torch = pytest.importorskip("torch")

from LoraQwen import LoraLinear
from metanetwork_family import (
    _ablate_loradict_rank_suffix,
    _merge_rank_loradicts,
    _rank_delta_diagnostics,
)


def test_lora_numel_cache_is_rank_aware():
    layer = LoraLinear(5, 7, bias=False)
    assert layer.lora_params_numel(8) == 96
    assert layer.lora_params_numel(24) == 288
    assert layer.lora_params_numel(32) == 384
    assert layer.lora_params_numel(8) == 96


def test_zero_gated_residual_is_function_preserving_and_has_gate_gradient():
    base = {
        "A": torch.randn(2, 5, 8),
        "B": torch.randn(2, 8, 7),
        "C": None,
    }
    residual = {
        "A": torch.randn(2, 5, 24),
        "B": torch.randn(2, 24, 7),
        "C": None,
    }
    gate = torch.zeros((), requires_grad=True)
    merged = _merge_rank_loradicts(base, residual, gate)
    inputs = torch.randn(2, 3, 5)

    expected = torch.matmul(torch.matmul(inputs, base["A"]), base["B"])
    actual = torch.matmul(torch.matmul(inputs, merged["A"]), merged["B"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.sum().backward()
    assert gate.grad is not None
    assert torch.isfinite(gate.grad)
    assert gate.grad.abs() > 0


def test_nested_loradict_concatenates_only_rank_dimensions():
    base_leaf = {
        "A": torch.randn(1, 3, 2),
        "B": torch.randn(1, 2, 4),
        "C": None,
    }
    residual_leaf = {
        "A": torch.randn(1, 3, 5),
        "B": torch.randn(1, 5, 4),
        "C": None,
    }
    merged = _merge_rank_loradicts(
        {0: {"attention": {"q": base_leaf}}},
        {0: {"attention": {"q": residual_leaf}}},
        torch.ones(()),
    )
    assert merged[0]["attention"]["q"]["A"].shape == (1, 3, 7)
    assert merged[0]["attention"]["q"]["B"].shape == (1, 7, 4)


def test_ablation_zeroes_only_residual_b_and_diagnostics_measure_effective_delta():
    base = {
        "A": torch.randn(1, 3, 2),
        "B": torch.randn(1, 2, 4),
        "C": None,
    }
    residual = {
        "A": torch.randn(1, 3, 5),
        "B": torch.randn(1, 5, 4),
        "C": None,
    }
    merged = _merge_rank_loradicts(base, residual, torch.ones(()))
    stats = _rank_delta_diagnostics(merged, 2, 5, torch.ones(()))
    assert stats["base_delta_rms"] > 0
    assert stats["residual_delta_rms"] > 0

    ablated = _ablate_loradict_rank_suffix(merged, 2)
    torch.testing.assert_close(ablated["B"][:, :2], merged["B"][:, :2])
    assert torch.count_nonzero(ablated["B"][:, 2:]) == 0
