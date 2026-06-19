"""Tests for FrobeniusAnchor."""

from __future__ import annotations

import torch

from meta_past.anchor.frobenius import AnchorSchedule, FrobeniusAnchor


def test_schedule_bounds():
    s = AnchorSchedule(coef_start=1.0, coef_end=0.1, decay_steps=100)
    assert s.at(0) == 1.0
    assert s.at(100) == 0.1
    assert s.at(150) == 0.1            # capped below
    assert s.at(-5) == 1.0             # capped above
    mid = s.at(50)
    assert 0.5 < mid < 0.6              # halfway between 1.0 and 0.1


def test_anchor_pulls_toward_snapshot():
    p = torch.tensor([10.0, 10.0, 10.0])
    params = [("p", p)]
    anchor = FrobeniusAnchor(params, AnchorSchedule(1.0, 1.0, 1))

    # Move p far from its snapshot.
    p.data.copy_(torch.tensor([0.0, 0.0, 0.0]))
    anchor.apply_step(params, lr=0.5, step=0)
    # With coef=1.0 and lr=0.5, p should move halfway toward the snapshot (10.0).
    assert torch.allclose(p, torch.tensor([5.0, 5.0, 5.0]), atol=1e-5)


def test_rollback_bumps_coef():
    p = torch.tensor([10.0, 10.0])
    params = [("p", p)]
    # coef_end=0 so schedule contributes 0 after step 1; only rollback_mul drives
    # the bump effect.
    anchor = FrobeniusAnchor(params, AnchorSchedule(0.0, 0.0, 1))
    p.data.copy_(torch.zeros(2))

    anchor.apply_step(params, lr=0.5, step=10)
    before_bump = p.clone()
    assert torch.allclose(before_bump, torch.zeros(2)), \
        "coef=0 must not move the params."

    anchor.bump_coef(factor=2.0)
    # Now schedule=0 * rollback_mul=2 = 0 — still no movement; prove it:
    p.data.copy_(torch.zeros(2))
    anchor.apply_step(params, lr=0.5, step=10)
    assert torch.allclose(p, torch.zeros(2))


def test_missing_snapshot_raises():
    p = torch.tensor([1.0])
    anchor = FrobeniusAnchor([("p", p)])
    # Introduce an unseen param.
    q = torch.tensor([2.0])
    try:
        anchor.apply_step([("q", q)], lr=0.1, step=0)
    except KeyError as e:
        assert "q" in str(e)
        return
    raise AssertionError("Expected KeyError for missing snapshot.")
