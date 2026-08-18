import pytest

from ControlSHINE.controlshine.splits import select_entity_disjoint_test


def _row(sample_id, entity, relation="r"):
    return {"sample_id": sample_id, "entity": entity, "relation": relation}


def test_entity_disjoint_split_is_deterministic_and_unique():
    pilot = [_row("p1", "Alpha")]
    candidates = [
        _row("p1", "Alpha"),
        _row("c1", " alpha "),
        _row("c2", "Beta"),
        _row("c3", "Beta"),
        _row("c4", "Gamma"),
    ]
    first, manifest = select_entity_disjoint_test(pilot, candidates, size=2, seed=7)
    second, _ = select_entity_disjoint_test(pilot, candidates, size=2, seed=7)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert {row["entity"] for row in first} in ({"Beta", "Gamma"},)
    assert manifest["test_unique_entities"] == 2


def test_split_fails_when_not_enough_disjoint_entities():
    with pytest.raises(ValueError, match="requested 2"):
        select_entity_disjoint_test(
            [_row("p1", "Alpha")], [_row("c1", "Beta")], size=2, seed=1
        )

