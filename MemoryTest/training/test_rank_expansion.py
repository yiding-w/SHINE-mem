import pytest


torch = pytest.importorskip("torch")


def test_repeat_expansion_preserves_prefix_and_fills_target():
    source = torch.arange(12, dtype=torch.float32).view(3, 4)
    target_rows = 8
    repeats = (target_rows + source.shape[0] - 1) // source.shape[0]
    expanded = source.repeat((repeats, 1))[:target_rows].clone()
    expanded[: source.shape[0]].copy_(source)
    assert expanded.shape == (8, 4)
    assert torch.equal(expanded[:3], source)
    assert torch.equal(expanded[3:6], source)
    assert torch.equal(expanded[6:8], source[:2])
