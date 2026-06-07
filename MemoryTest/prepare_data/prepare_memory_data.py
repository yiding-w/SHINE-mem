#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from MemoryTest.prepare_data.fact_schema import (
    assert_no_test_triple_leakage,
    assert_person_disjoint,
    group_by_person,
    normalize_rows,
    relation_distribution,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "MemoryTest" / "json_data" / "semantic_facts.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MemoryTest" / "json_data" / "splits"

FIRST_NAMES = [
    "Ada",
    "Mira",
    "Nolan",
    "Tessa",
    "Felix",
    "Ivy",
    "Rowan",
    "Maeve",
    "Orion",
    "Lena",
    "Caleb",
    "Sage",
]
LAST_NAMES = [
    "Arden",
    "Bellamy",
    "Corwin",
    "Davenport",
    "Everly",
    "Fletcher",
    "Hollis",
    "Ivers",
    "Keaton",
    "Mercer",
    "Winslow",
    "Vale",
]

RELATION_SPECS = {
    "favorite_fruit": {
        "values": ["guava", "plum", "nectarine", "blackberry", "apricot", "papaya"],
        "text": [
            "{person}'s favorite fruit is {value}.",
            "{person} likes {value} more than any other fruit.",
        ],
        "question": [
            "What is {person}'s favorite fruit?",
            "Which fruit does {person} like best?",
        ],
    },
    "job": {
        "values": ["cartographer", "florist", "locksmith", "tailor", "archivist", "potter"],
        "text": [
            "{person} works as a {value}.",
            "{person}'s job is {article} {value}.",
        ],
        "question": [
            "What is {person}'s job?",
            "What does {person} do for work?",
        ],
    },
    "home_city": {
        "values": ["Madison", "Boise", "Tempe", "Akron", "Eugene", "Boulder"],
        "text": [
            "{person} lives in {value}.",
            "{person}'s home city is {value}.",
        ],
        "question": [
            "Which city does {person} live in?",
            "What is {person}'s home city?",
        ],
    },
    "owned_item": {
        "values": ["green compass", "brass lantern", "striped backpack", "ceramic mug", "violet scarf"],
        "text": [
            "{person} owns a {value}.",
            "The item {person} owns is a {value}.",
        ],
        "question": [
            "What does {person} own?",
            "Which item belongs to {person}?",
        ],
    },
    "favorite_drink": {
        "values": ["ginger cocoa", "pear soda", "hazelnut milk", "rose lemonade", "vanilla cider"],
        "text": [
            "{person}'s favorite drink is {value}.",
            "{person} prefers {value} as a drink.",
        ],
        "question": [
            "What is {person}'s favorite drink?",
            "Which drink does {person} prefer?",
        ],
    },
    "favorite_color": {
        "values": ["teal", "maroon", "amber", "indigo", "coral", "silver"],
        "text": [
            "{person}'s favorite color is {value}.",
            "{person} likes the color {value} best.",
        ],
        "question": [
            "What is {person}'s favorite color?",
            "Which color does {person} like best?",
        ],
    },
    "hobby": {
        "values": ["birdwatching", "calligraphy", "kayaking", "origami", "wood carving"],
        "text": [
            "{person}'s hobby is {value}.",
            "{person} spends free time on {value}.",
        ],
        "question": [
            "What is {person}'s hobby?",
            "What does {person} do in free time?",
        ],
    },
    "pet_name": {
        "values": ["Miso", "Juniper", "Pixel", "Clover", "Basil", "Nimbus"],
        "text": [
            "{person}'s pet is named {value}.",
            "The name of {person}'s pet is {value}.",
        ],
        "question": [
            "What is {person}'s pet named?",
            "What is the name of {person}'s pet?",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare person-disjoint MemoryTest semantic fact splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--generate-synthetic-train", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_by_person(rows: list[dict], seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    grouped = group_by_person(rows)
    people = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(people)

    num_people = len(people)
    train_end = int(num_people * train_ratio)
    val_end = train_end + int(num_people * val_ratio)
    train_people = set(people[:train_end])
    val_people = set(people[train_end:val_end])
    test_people = set(people[val_end:])

    train = [row for person in people if person in train_people for row in grouped[person]]
    val = [row for person in people if person in val_people for row in grouped[person]]
    test = [row for person in people if person in test_people for row in grouped[person]]
    assert_person_disjoint(train, val, test)
    return train, val, test


def article_for(value: str) -> str:
    return "an" if value[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"


def make_unique_person(rng: random.Random, used_people: set[str]) -> str:
    for _ in range(10000):
        person = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if person not in used_people:
            used_people.add(person)
            return person
    raise RuntimeError("Unable to generate a unique synthetic person name.")


def generate_synthetic_rows(count: int, seed: int, existing_rows: list[dict], protected_rows: list[dict]) -> list[dict]:
    rng = random.Random(seed + 1009)
    protected_people = {row["person"] for row in protected_rows}
    protected_answers = {str(row["answer"]).casefold() for row in protected_rows}
    used_people = {row["person"] for row in existing_rows} | protected_people
    generated = []

    relation_cycle = list(RELATION_SPECS)
    for idx in range(count):
        attribute = relation_cycle[idx % len(relation_cycle)]
        spec = RELATION_SPECS[attribute]
        person = make_unique_person(rng, used_people)
        value_pool = [value for value in spec["values"] if value.casefold() not in protected_answers]
        if not value_pool:
            value_pool = spec["values"]
        value = rng.choice(value_pool)
        template_args = {"person": person, "value": value, "article": article_for(value)}
        text = rng.choice(spec["text"]).format(**template_args)
        question = rng.choice(spec["question"]).format(**template_args)
        generated.append(
            {
                "id": f"synthetic_semantic_{idx + 1:05d}",
                "person": person,
                "attribute": attribute,
                "text": text,
                "question": question,
                "answer": value,
                "synthetic": True,
            }
        )
    assert_no_test_triple_leakage(generated, protected_rows)
    return generated


def split_meta(seed: int, train: list[dict], val: list[dict], test: list[dict], synthetic: list[dict]) -> dict:
    all_rows = train + val + test
    return {
        "seed": seed,
        "counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "synthetic": len(synthetic),
            "train_augmented": len(train) + len(synthetic),
        },
        "persons": {
            "train": sorted({row["person"] for row in train}),
            "val": sorted({row["person"] for row in val}),
            "test": sorted({row["person"] for row in test}),
            "synthetic": sorted({row["person"] for row in synthetic}),
        },
        "relation_distribution": {
            "all": relation_distribution(all_rows),
            "train": relation_distribution(train),
            "val": relation_distribution(val),
            "test": relation_distribution(test),
            "synthetic": relation_distribution(synthetic),
        },
        "answer_distribution": dict(sorted(Counter(row["answer"] for row in all_rows).items())),
    }


def main() -> None:
    args = parse_args()
    rows = normalize_rows(read_json(args.input))
    train, val, test = split_by_person(rows, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    synthetic = []
    if args.generate_synthetic_train:
        synthetic = generate_synthetic_rows(args.generate_synthetic_train, args.seed, train, val + test)

    train_augmented = train + synthetic
    assert_person_disjoint(train_augmented, val, test)
    assert_no_test_triple_leakage(train_augmented, val + test)

    write_json(args.output_dir / "semantic_train.json", train)
    write_json(args.output_dir / "semantic_val.json", val)
    write_json(args.output_dir / "semantic_test.json", test)
    if synthetic:
        write_json(args.output_dir / "synthetic_train.json", synthetic)
        write_json(args.output_dir / "semantic_train_augmented.json", train_augmented)
    write_json(args.output_dir / "split_meta.json", split_meta(args.seed, train, val, test, synthetic))

    print(f"Wrote splits to {args.output_dir}")
    print(f"train={len(train)} val={len(val)} test={len(test)} synthetic={len(synthetic)}")


if __name__ == "__main__":
    main()
