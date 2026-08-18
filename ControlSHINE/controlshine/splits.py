"""Deterministic entity-disjoint dataset splitting utilities."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any


def normalize_entity(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def select_entity_disjoint_test(
    pilot_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    one_per_entity: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if size < 1:
        raise ValueError("size must be positive")
    pilot_entities = {normalize_entity(row.get("entity")) for row in pilot_rows}
    pilot_entities.discard("")
    pilot_ids = {str(row["sample_id"]) for row in pilot_rows}

    eligible = []
    excluded = Counter()
    for row in candidate_rows:
        entity = normalize_entity(row.get("entity"))
        sample_id = str(row["sample_id"])
        if sample_id in pilot_ids:
            excluded["sample_id_overlap"] += 1
            continue
        if not entity:
            excluded["missing_entity"] += 1
            continue
        if entity in pilot_entities:
            excluded["entity_overlap"] += 1
            continue
        eligible.append(row)

    random.Random(seed).shuffle(eligible)
    selected = []
    selected_entities = set()
    for row in eligible:
        entity = normalize_entity(row.get("entity"))
        if one_per_entity and entity in selected_entities:
            excluded["duplicate_test_entity"] += 1
            continue
        selected.append(row)
        selected_entities.add(entity)
        if len(selected) >= size:
            break
    if len(selected) < size:
        raise ValueError(f"requested {size} test rows but only selected {len(selected)}")

    relation_counts = Counter(str(row.get("relation") or "") for row in selected)
    manifest = {
        "seed": seed,
        "requested_size": size,
        "selected_size": len(selected),
        "one_per_entity": one_per_entity,
        "pilot_size": len(pilot_rows),
        "candidate_size": len(candidate_rows),
        "pilot_unique_entities": len(pilot_entities),
        "test_unique_entities": len(selected_entities),
        "excluded": dict(excluded),
        "relation_counts": dict(sorted(relation_counts.items())),
        "pilot_sample_ids": sorted(pilot_ids),
        "test_sample_ids": [str(row["sample_id"]) for row in selected],
    }
    return selected, manifest

