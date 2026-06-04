#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path


FIRST_NAMES = [
    "Alden", "Briar", "Celia", "Dorian", "Elara", "Fintan", "Greta", "Harlan",
    "Iris", "Jules", "Keira", "Landon", "Mira", "Nolan", "Orla", "Pavel",
    "Quinn", "Rhea", "Silas", "Talia", "Ulric", "Vera", "Willa", "Xander",
    "Yara", "Zane",
]

LAST_NAMES = [
    "Abbott", "Bennett", "Carver", "Dunham", "Ellis", "Foster", "Granger",
    "Hayes", "Iverson", "Jensen", "Keller", "Lang", "Mercer", "Nash",
    "Osborne", "Price", "Rowan", "Sutter", "Talbot", "Vance",
]

DISTRACTOR_SUBJECTS = [
    "The museum ticket", "The meeting note", "The blue folder", "The train receipt",
    "The garden label", "The recipe card", "The hotel invoice", "The campus flyer",
    "The library badge", "The storage tag", "The weather memo", "The parcel slip",
]

DISTRACTOR_VERBS = [
    "was placed inside", "was copied onto", "was checked beside", "was moved behind",
    "was folded under", "was pinned above", "was saved near", "was written across",
]

DISTRACTOR_OBJECTS = [
    "the wooden cabinet", "the second drawer", "the north window", "the kitchen shelf",
    "the front counter", "the archive box", "the green notebook", "the hallway desk",
]

DISTRACTOR_PEOPLE = [
    "Mara", "Theo", "Nina", "Owen", "Lena", "Caleb", "Priya", "Jonas",
    "Mei", "Ronan", "Sofia", "Eli",
]

DISTRACTOR_PLACES = [
    "station", "clinic", "workshop", "bookstore", "theater", "warehouse",
    "gallery", "market", "office", "school",
]

DISTRACTOR_ITEMS = [
    "silver key", "travel pass", "coffee mug", "canvas bag", "paper map",
    "spare charger", "red umbrella", "lunch receipt", "visitor badge", "train ticket",
]

DISTRACTOR_TEMPLATES = [
    "{person} left the {item} at the {place}.",
    "The {item} was stored near {obj}.",
    "{person} wrote code {code} on the {item}.",
    "The {place} notice listed code {code}.",
    "{person} moved the {item} to the {place}.",
    "The {item} was beside {obj}.",
    "Code {code} marked the {item}.",
    "{person} returned the {item} before noon.",
]

SEMANTIC_FACT_SPECS = [
    {
        "attribute": "favorite_fruit",
        "question": "What is {person}'s favorite fruit?",
        "text": "{person}'s favorite fruit is {answer}.",
        "answers": [
            "mango", "pear", "peach", "plum", "kiwi", "papaya", "orange", "banana",
            "apricot", "pineapple", "blueberry", "raspberry",
        ],
    },
    {
        "attribute": "job",
        "question": "What is {person}'s job?",
        "text": "{person} works as a {answer}.",
        "answers": [
            "baker", "dentist", "carpenter", "teacher", "pilot", "chemist",
            "nurse", "tailor", "chef", "gardener", "librarian", "mechanic",
        ],
    },
    {
        "attribute": "home_city",
        "question": "Which city does {person} live in?",
        "text": "{person} lives in {answer}.",
        "answers": [
            "Denver", "Austin", "Seattle", "Boston", "Phoenix", "Madison",
            "Portland", "Dallas", "Chicago", "Raleigh", "Tucson", "Atlanta",
        ],
    },
    {
        "attribute": "owned_item",
        "question": "What does {person} own?",
        "text": "{person} owns a {answer}.",
        "answers": [
            "yellow bicycle", "green backpack", "silver camera", "wooden violin",
            "blue notebook", "red scooter", "white telescope", "black suitcase",
            "purple kayak", "bronze compass", "striped umbrella", "canvas tent",
        ],
    },
    {
        "attribute": "favorite_drink",
        "question": "What is {person}'s favorite drink?",
        "text": "{person}'s favorite drink is {answer}.",
        "answers": [
            "lemon tea", "apple juice", "mint coffee", "grape soda", "coconut water",
            "ginger milk", "peach tea", "berry smoothie", "iced cocoa", "melon juice",
        ],
    },
]


def make_phone(rng: random.Random) -> str:
    return f"{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def make_people(count: int, rng: random.Random):
    names = []
    for first in FIRST_NAMES:
        for last in LAST_NAMES:
            names.append(f"{first} {last}")
    rng.shuffle(names)
    if count > len(names):
        raise ValueError(f"Requested {count} people, but only {len(names)} unique names are available.")
    return names[:count]


def make_phonebook(count: int, rng: random.Random):
    people = make_people(count, rng)
    used_phones = set()
    rows = []
    for idx, person in enumerate(people, start=1):
        phone = make_phone(rng)
        while phone in used_phones:
            phone = make_phone(rng)
        used_phones.add(phone)
        rows.append(
            {
                "id": f"phone_{idx:04d}",
                "person": person,
                "phone": phone,
                "text": f"{person}'s phone number is {phone}.",
                "question": f"What is {person}'s phone number?",
                "answer": phone,
            }
        )
    return rows


def make_semantic_facts(count: int, rng: random.Random):
    people = make_people(count, rng)
    rows = []
    for idx, person in enumerate(people, start=1):
        spec = SEMANTIC_FACT_SPECS[(idx - 1) % len(SEMANTIC_FACT_SPECS)]
        answer = rng.choice(spec["answers"])
        rows.append(
            {
                "id": f"semantic_{idx:04d}",
                "person": person,
                "attribute": spec["attribute"],
                "text": spec["text"].format(person=person, answer=answer),
                "question": spec["question"].format(person=person),
                "answer": answer,
            }
        )
    return rows


def make_distractors(count: int, rng: random.Random):
    rows = []
    for idx in range(1, count + 1):
        subject = rng.choice(DISTRACTOR_SUBJECTS)
        verb = rng.choice(DISTRACTOR_VERBS)
        obj = rng.choice(DISTRACTOR_OBJECTS)
        person = rng.choice(DISTRACTOR_PEOPLE)
        place = rng.choice(DISTRACTOR_PLACES)
        item = rng.choice(DISTRACTOR_ITEMS)
        code = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(20, 99)}-{rng.randint(100, 999)}"
        template = rng.choice(DISTRACTOR_TEMPLATES)
        rows.append(
            {
                "id": f"distractor_{idx:04d}",
                "text": template.format(
                    subject=subject,
                    code=code,
                    verb=verb,
                    obj=obj,
                    person=person,
                    place=place,
                    item=item,
                ),
            }
        )
    return rows


def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate capacity-test source JSON files.")
    parser.add_argument("--count", type=int, default=100, help="Number of rows to generate for each source JSON.")
    parser.add_argument("--seed", type=int, default=20260604, help="Random seed for deterministic data.")
    parser.add_argument("--output-dir", type=str, default="MemoryTest/json_data", help="Directory for generated JSON files.")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    write_json(output_dir / "phonebook.json", make_phonebook(args.count, rng))
    write_json(output_dir / "semantic_facts.json", make_semantic_facts(args.count, rng))
    write_json(output_dir / "distractors.json", make_distractors(args.count, rng))
    print(f"Wrote {args.count} phone facts to {output_dir / 'phonebook.json'}")
    print(f"Wrote {args.count} semantic facts to {output_dir / 'semantic_facts.json'}")
    print(f"Wrote {args.count} distractors to {output_dir / 'distractors.json'}")


if __name__ == "__main__":
    main()
