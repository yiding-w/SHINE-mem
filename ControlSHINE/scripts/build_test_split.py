#!/usr/bin/env python
"""Create a deterministic entity-disjoint ControlSHINE test candidate set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ControlSHINE.controlshine.splits import select_entity_disjoint_test  # noqa: E402


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    selected, manifest = select_entity_disjoint_test(
        _read_jsonl(args.pilot),
        _read_jsonl(args.candidates),
        size=args.size,
        seed=args.seed,
        one_per_entity=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "manifest": str(args.manifest),
        "selected_size": manifest["selected_size"],
        "pilot_unique_entities": manifest["pilot_unique_entities"],
        "test_unique_entities": manifest["test_unique_entities"],
        "excluded": manifest["excluded"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

