"""Roundtrip tests for InPlacePerturber: apply(+1) then restore(+1) must
return tensors to their original state bit-exactly in fp32 and ULP-close in bf16."""

from __future__ import annotations

import torch

from meta_past.es.perturb import InPlacePerturber


def _make_params(dtype: torch.dtype, n_tensors: int = 4, seed: int = 1234):
    torch.manual_seed(seed)
    params = []
    shapes = [(7,), (3, 5), (2, 4, 6), (8, 8)]
    for k in range(n_tensors):
        shape = shapes[k % len(shapes)]
        t = torch.randn(*shape, dtype=dtype)
        params.append((f"p{k}", t))
    return params


def test_roundtrip_fp32():
    params = _make_params(torch.float32)
    snapshots = [p.clone() for _, p in params]
    perturber = InPlacePerturber(params, sigma=5e-3, noise_dtype=torch.float32)

    for sign in (+1, -1):
        perturber.apply(base_seed=7, sign=sign)
        # Something must have changed.
        changed = any(
            not torch.equal(snap, p) for snap, (_, p) in zip(snapshots, params)
        )
        assert changed, "Perturbation had no effect."
        perturber.restore(base_seed=7, sign=sign)
        # fp32 round-trip is bit-exact when noise is drawn in fp32.
        for snap, (name, p) in zip(snapshots, params):
            assert torch.equal(snap, p), f"{name}: fp32 restore not bit-exact."


def test_roundtrip_bf16():
    params = _make_params(torch.bfloat16)
    snapshots = [p.clone() for _, p in params]
    perturber = InPlacePerturber(params, sigma=5e-3, noise_dtype=torch.float32)

    perturber.apply(base_seed=11, sign=+1)
    perturber.restore(base_seed=11, sign=+1)
    # bf16 has ~1e-2 relative precision; tolerance is generous but still a
    # regression gate if apply/restore diverge structurally.
    for snap, (name, p) in zip(snapshots, params):
        diff = (snap.float() - p.float()).abs().max().item()
        assert diff < 1e-2, f"{name}: bf16 roundtrip diff {diff} too large."


def test_different_seeds_produce_different_noise():
    # sigma is small enough that restore is bit-exact in fp32 (fp32 sum/diff
    # is exact whenever the addend magnitudes are similar to the base).
    params = _make_params(torch.float32, n_tensors=2)
    perturber = InPlacePerturber(params, sigma=5e-3)

    perturber.apply(base_seed=1, sign=+1)
    after_s1 = [p.clone() for _, p in params]
    perturber.restore(base_seed=1, sign=+1)

    perturber.apply(base_seed=2, sign=+1)
    after_s2 = [p.clone() for _, p in params]
    perturber.restore(base_seed=2, sign=+1)

    assert not torch.equal(after_s1[0], after_s2[0]), \
        "Same tensor under different base seeds got identical noise."


def test_per_tensor_seed_shift_makes_tensors_independent():
    # If two tensors of identical shape are perturbed, the applied noise
    # should differ — otherwise we have the layer-correlated-noise bug from
    # ES-at-Scale's default variant.
    a = torch.zeros(16)
    b = torch.zeros(16)
    params = [("a", a), ("b", b)]
    perturber = InPlacePerturber(params, sigma=1.0)
    perturber.apply(base_seed=42, sign=+1)
    assert not torch.equal(a, b), "Same-shape tensors got identical noise!"
