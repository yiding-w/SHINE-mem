#!/usr/bin/env python
"""Build a tiny deterministic dataset for pipeline smoke tests.

This is deliberately not the final benchmark generator. Its purpose is to
exercise schema validation, I/O, and the future three-forward evaluation path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controlshine.schema import ControlSample, SourceFact  # noqa: E402


def build_samples() -> list[ControlSample]:
    rows = [
        ("Orin Vale", "access code", "1842", "7305", "9614"),
        ("Mira Fen", "badge color", "amber", "violet", "teal"),
        ("Tovan Reed", "assigned city", "Larkspur", "Norwyn", "Bellmere"),
    ]
    samples = []
    for index, (entity, relation, base, memory, context) in enumerate(rows):
        question = f"What is {entity}'s {relation}? Answer with only the value."
        samples.append(
            ControlSample(
                sample_id=f"synthetic-{index:04d}",
                question=question,
                entity=entity,
                relation=relation,
                base=SourceFact(f"{entity}'s {relation} is {base}.", base),
                memory=SourceFact(f"{entity}'s {relation} is {memory}.", memory),
                context=SourceFact(f"{entity}'s {relation} is {context}.", context),
                provenance={"dataset": "controlshine_synthetic_smoke", "version": 1},
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "synthetic_smoke.jsonl",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in build_samples():
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    print(f"wrote {len(build_samples())} samples to {args.output}")


if __name__ == "__main__":
    main()

