#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEGMENTED long-history memory data synthesizer for SHINE_V2 (Phase 2, detach_state).

Each history = a sequence of dated "sessions" (segments). A session states a few
unique, dated facts (user turns, with assistant acks) buried in real filler, and
ends with per-segment QA about THOSE facts. A cross-segment final QA (conflict /
multi-hop / temporal / TTL) is appended to the last segment.

ANSWER DIVERSITY (this version): facts use **near-unique random values** instead of
tiny fixed pools — composed names, random titles, alphanumeric codes, numbers — and
**multiple phrasing templates** per fact type. This prevents the model from memorizing
a small closed answer set instead of actually retrieving from memory.

Output JSONL (one per line) — UNCHANGED schema (consumed by
mydatasets/pretrain_annealing/memory_stream.py):
  {"id","n_segments","approx_tokens",
   "segments":[{"date","history_turns":[{role,content}],"qa":[{question,answer,type,...}]}],
   "final_qa":[...], "teacher_prompt_template"}

Example:
  python datagen/generate_memory_seg.py --out data/mem_synth/train.jsonl --num 3000 \
      --filler-hf wiki,dialog --filler-hf-num 40000 \
      --segments 8,16,32,64 --segment-tokens 1800 --tokenizer ./models/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from typing import Any, Dict, List, Optional, Tuple

from generate_memory_data import (
    FILLER_FALLBACK, build_hf_filler, load_filler, maybe_tokenizer, MONTHS,
)

# ---------------------------------------------------------------------------
# Vocab pools (large enough that combinations are near-unique)
# ---------------------------------------------------------------------------
FIRST = ["Alex", "Bao", "Chen", "Dana", "Elena", "Farid", "Grace", "Hiro", "Ivan",
         "Jun", "Kira", "Liang", "Mei", "Nadia", "Omar", "Priya", "Qing", "Rosa",
         "Sven", "Tara", "Umar", "Vera", "Wei", "Xena", "Yara", "Zane", "Noor",
         "Inez", "Theo", "Lucia", "Marcus", "Feng", "Ravi", "Sora", "Dmitri"]
LAST = ["Vance", "Okafor", "Mendez", "Holt", "Nakamura", "Costa", "Bauer", "Singh",
        "Larsson", "Ito", "Romano", "Khan", "Petrov", "Ortega", "Lindqvist", "Abe",
        "Moreau", "Haas", "Volkov", "Reyes", "Sato", "Dubois", "Fischer", "Cruz",
        "Novak", "Park", "Weiss", "Tanaka", "Silva", "Brandt"]
SPECIES = ["corgi", "tabby cat", "axolotl", "parrot", "tortoise", "beagle", "gecko",
           "cockatiel", "ferret", "hedgehog", "ragdoll cat", "shiba inu", "canary"]
PET_NAMES = ["Mochi", "Biscuit", "Pixel", "Cobalt", "Sage", "Rumi", "Tofu", "Comet",
             "Pepper", "Waffle", "Juniper", "Mango", "Dot", "Pickle", "Echo", "Miso"]
CITY = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Reykjavik", "Chiang Mai",
        "Montreal", "Tallinn", "Valencia", "Hobart", "Kigali", "Porto", "Bergen",
        "Da Nang", "Cusco", "Ghent", "Lviv", "Pune", "Tromso", "Cebu", "Graz",
        "Almaty", "Recife", "Aarhus", "Hue", "Split", "Galway", "Aalborg"]
COLOR = ["teal", "amber", "maroon", "chartreuse", "indigo", "ochre", "vermilion",
         "cobalt", "sage green", "burgundy", "slate gray", "mustard", "lavender",
         "rust", "olive", "plum", "coral", "navy", "mint", "charcoal"]
ADJ = ["Silent", "Crimson", "Hidden", "Northern", "Broken", "Velvet", "Iron",
       "Golden", "Quiet", "Distant", "Hollow", "Amber", "Frozen", "Electric",
       "Marble", "Paper", "Glass", "Copper", "Midnight", "Wandering", "Ashen"]
NOUN = ["Meridian", "Harbor", "Lantern", "Compass", "Archive", "Foundry", "Garden",
        "Circuit", "Pendulum", "Beacon", "Cipher", "Atlas", "Cascade", "Ledger",
        "Mosaic", "Quarry", "Spire", "Almanac", "Trellis", "Verdict", "Reservoir"]
ITEM = ["espresso machine", "mechanical keyboard", "film camera", "standing desk",
        "e-ink tablet", "turntable", "drone", "telescope", "sewing machine",
        "road bike", "synthesizer", "pizza oven", "label printer", "microscope",
        "projector", "rowing machine", "ukulele", "kettlebell set"]

ACKS = ["Got it.", "Noted.", "I'll remember that.", "Sounds good.", "Understood.",
        "Okay, noted.", "Thanks, logged it.", "Will keep that in mind."]


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def r_name(rng) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def r_code(rng, prefix: str) -> str:
    return f"{prefix}-{rng.choice(string.ascii_uppercase)}{rng.randint(100, 999)}"


def r_title(rng) -> str:
    return f"The {rng.choice(ADJ)} {rng.choice(NOUN)}"


def r_number(rng) -> str:
    return str(rng.randint(10000, 99999))


def r_pet(rng) -> str:
    return f"{rng.choice(SPECIES)} named {rng.choice(PET_NAMES)}"


def r_item(rng) -> str:
    return rng.choice(ITEM)


def r_city(rng) -> str:
    return rng.choice(CITY)


def r_color(rng) -> str:
    return rng.choice(COLOR)


# fact_type -> dict(gen, utts[], q[])  ; {val} is the unique answer
FACT_TYPES: Dict[str, Dict[str, Any]] = {
    "pet": dict(
        gen=r_pet,
        utts=["I adopted {a_val}.", "We brought home {a_val}.", "My new pet is {a_val}."],
        q=["On {date}, what pet did I get?", "What pet did I adopt on {date}?"],
    ),
    "trip": dict(
        gen=r_city,
        utts=["I traveled to {val}.", "I flew to {val} for a few days.", "I took a trip to {val}."],
        q=["On {date}, where did I travel?", "Which city did I visit on {date}?"],
    ),
    "project": dict(
        gen=lambda rng: r_title(rng),
        utts=["I kicked off a project called '{val}'.", "I started a new project, '{val}'.",
              "I was put in charge of '{val}'."],
        q=["On {date}, what project did I start?", "Which project did I kick off on {date}?"],
    ),
    "book": dict(
        gen=r_title,
        utts=["I started reading '{val}'.", "I began the book '{val}'.", "I picked up '{val}'."],
        q=["On {date}, what book did I start reading?", "Which book did I start on {date}?"],
    ),
    "purchase": dict(
        gen=r_item,
        utts=["I bought {a_val}.", "I ordered {a_val} online.", "I treated myself to {a_val}."],
        q=["On {date}, what did I buy?", "What did I purchase on {date}?"],
    ),
    "room": dict(
        gen=lambda rng: "room " + r_code(rng, "R"),
        utts=["I moved into {val}.", "My new office is {val}.", "I was assigned {val}."],
        q=["On {date}, which room did I move into?", "What is my room from {date}?"],
    ),
    "badge": dict(
        gen=lambda rng: r_number(rng),
        utts=["my new badge number is {val}.", "I was given membership ID {val}.",
              "my locker code is {val}."],
        q=["On {date}, what number was I assigned?", "Which ID did I get on {date}?"],
    ),
    "color": dict(
        gen=r_color,
        utts=["I repainted my study {val}.", "I painted the hallway {val}."],
        q=["On {date}, what color did I paint?", "Which color did I use on {date}?"],
    ),
}


def _fill_utt(tpl: str, val: str, date: str) -> str:
    return tpl.format(val=val, a_val=_a(val), date=date)


def _rand_date_seq(rng, n: int) -> List[str]:
    keys = sorted(rng.sample(range(0, 1600), n))
    out = []
    for k in keys:
        y, m, d = 2021 + k // 360, (k % 360) // 30, (k % 30) + 1
        out.append(f"{MONTHS[m]} {d}, {y}")
    return out


def _filler_turns(rng, pool, n) -> List[Dict[str, str]]:
    turns = []
    for _ in range(n):
        turns.append({"role": "user", "content": rng.choice(pool)})
        turns.append({"role": "assistant", "content": rng.choice(
            ["Thanks for sharing.", "Interesting!", "Noted.", "Haha, nice."])})
    return turns


def build_history(rng, filler_pool, n_segments, segment_chars):
    person = r_name(rng)
    dates = _rand_date_seq(rng, n_segments)

    # conflict: one fact type stated twice (different unique values, in two segments)
    conflict_type = rng.choice([k for k in FACT_TYPES])
    seg_a, seg_b = sorted(rng.sample(range(n_segments), 2))
    val_a = FACT_TYPES[conflict_type]["gen"](rng)
    val_b = FACT_TYPES[conflict_type]["gen"](rng)
    while val_b == val_a:
        val_b = FACT_TYPES[conflict_type]["gen"](rng)

    # multi-hop: a colleague (unique name) in a unique city
    colleague_seg = rng.randrange(n_segments)
    colleague = r_name(rng)
    colleague_city = r_city(rng)

    rule_seg = rng.randrange(n_segments)
    rule_style = rng.choice(["all uppercase", "all lowercase", "ending with three exclamation marks"])

    segments, last_primary = [], ("", "")
    for si in range(n_segments):
        date = dates[si]
        facts: List[Tuple[str, str, str]] = []  # (utterance, question, answer)

        if si in (seg_a, seg_b):
            v = val_a if si == seg_a else val_b
            ft = FACT_TYPES[conflict_type]
            facts.append((_fill_utt(rng.choice(ft["utts"]), v, date),
                          rng.choice(ft["q"]).format(date=date), v))
        if si == colleague_seg:
            facts.append((f"I met my colleague {colleague}, who works in {colleague_city}.",
                          f"On {date}, which colleague did I meet?", colleague))
        for ftype in rng.sample([k for k in FACT_TYPES if k != conflict_type], rng.randint(1, 2)):
            ft = FACT_TYPES[ftype]
            v = ft["gen"](rng)
            facts.append((_fill_utt(rng.choice(ft["utts"]), v, date),
                          rng.choice(ft["q"]).format(date=date), v))

        last_primary = (date, facts[-1][0].rstrip("."))

        turns = [{"role": "user", "content": f"(Session on {date})"},
                 {"role": "assistant", "content": "Noted the date."}]
        fact_turns = []
        for (utt, _q, _a_) in facts:
            body = utt if (utt.startswith("I ") or utt.startswith("I'")) else utt[0].lower() + utt[1:]
            fact_turns.append({"role": "user", "content": f"On {date}, {body}"})
            fact_turns.append({"role": "assistant", "content": rng.choice(ACKS)})
        if si == rule_seg:
            fact_turns.append({"role": "user",
                               "content": f"From now on, whenever I ask you to summarize, answer {rule_style}."})
            fact_turns.append({"role": "assistant", "content": "Understood, I'll follow that."})

        chunk = max(1, (segment_chars // 120) // (len(fact_turns) // 2 + 1))
        for k in range(0, len(fact_turns), 2):
            turns.extend(_filler_turns(rng, filler_pool, chunk))
            turns.extend(fact_turns[k:k + 2])
        while sum(len(t["content"]) for t in turns) < segment_chars:
            turns.extend(_filler_turns(rng, filler_pool, 4))

        seg_qa = [{"question": q, "answer": a, "type": "segment_retrieval", "segment": si}
                  for (_u, q, a) in facts]
        segments.append({"date": date, "history_turns": turns, "qa": seg_qa})

    _recent = last_primary[1][2:].strip() if last_primary[1].startswith("I ") else last_primary[1]
    final_qa = [
        {"question": f"My {conflict_type} changed over time. What is the latest one?",
         "answer": val_b, "type": "conflict", "hops": 1},
        {"question": f"What was my {conflict_type} BEFORE the most recent change?",
         "answer": val_a, "type": "conflict", "hops": 1},
        {"question": f"Which city does my colleague {colleague} work in?",
         "answer": colleague_city, "type": "multi_hop", "hops": 2},
        {"question": "What is the most recent thing I told you I did, and on what date?",
         "answer": f"{_recent} on {last_primary[0]}", "type": "temporal", "hops": 1},
        {"question": "Summarize who I am in one line.",
         "answer": _styled(f"{person} most recently {_recent}", rule_style),
         "type": "ttl", "hops": 1, "rule": rule_style},
    ]
    return segments, final_qa, {"person": person, "n_segments": n_segments}


def _styled(s, style):
    if style == "all uppercase":
        return s.upper()
    if style == "all lowercase":
        return s.lower()
    return s.rstrip(".") + "!!!"


def approx_tokens(segments, tok) -> int:
    text = "\n".join(t["content"] for seg in segments for t in seg["history_turns"])
    return len(tok.encode(text, add_special_tokens=False)) if tok else len(text) // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--segments", default="8,16,32,64")
    ap.add_argument("--segment-tokens", type=int, default=1800)
    ap.add_argument("--filler-dir", default=None)
    ap.add_argument("--filler-hf", default=None)
    ap.add_argument("--filler-hf-num", type=int, default=20000)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    filler_pool = load_filler(args.filler_dir)
    if args.filler_hf:
        specs = [s.strip() for s in args.filler_hf.split(",") if s.strip()]
        filler_pool += build_hf_filler(specs, args.filler_hf_num, args.seed)
    if not filler_pool:
        print("[warn] no real filler; using placeholder.")
        filler_pool = list(FILLER_FALLBACK)
    print(f"[filler] pool size = {len(filler_pool)}")

    tok = maybe_tokenizer(args.tokenizer)
    seg_counts = [int(x) for x in args.segments.split(",")]
    seg_chars = args.segment_tokens * 4

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(args.num):
            n_seg = rng.choice(seg_counts)
            sub = random.Random(rng.randint(0, 2**31))
            segments, final_qa, meta = build_history(sub, filler_pool, n_seg, seg_chars)
            rec = {"id": f"mem-{i:06d}", "n_segments": n_seg,
                   "approx_tokens": approx_tokens(segments, tok),
                   "segments": segments, "final_qa": final_qa,
                   "teacher_prompt_template": (
                       "Use ONLY the conversation history below.\n=== HISTORY ===\n{history}\n"
                       "=== END ===\nQuestion: {question}\nAnswer:")}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{args.num} (n_seg={n_seg}, approx_tokens={rec['approx_tokens']})", flush=True)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
