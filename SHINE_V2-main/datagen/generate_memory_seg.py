#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEGMENTED long-history memory data synthesizer for SHINE_V2 (Phase 2, detach_state).

Why segmented: detach_state stores W with requires_grad=False (truncated-BPTT), so
the final QA loss CANNOT backprop into earlier history steps. Each streaming step
therefore needs its OWN meaningful loss to train "write this segment into W". We
attach per-segment QA (answerable from the segment just shown) to every step, plus
a cross-segment final QA (conflict / multi-hop / temporal / TTL) on the last step
to exercise reading accumulated W.

Each history = a sequence of dated "sessions" (segments). A session states a few
unique, dated facts (as user turns, with assistant acks) buried in real filler, and
ends with per-segment QA about THOSE facts (needle-in-segment). Some attributes are
re-stated in a later session (conflict); one colleague fact enables multi-hop.

Output JSONL (one per line):
  {
    "id", "target_tokens", "approx_tokens",
    "segments": [
       {"date": str, "history_turns": [{role,content}], "qa": [{question,answer,type,...}]},
       ...
    ],
    "final_qa": [{question,answer,type,hops,...}],
    "teacher_prompt_template": str
  }

Consumed by mydatasets/pretrain_annealing/memory_stream.py.

Example:
  python datagen/generate_memory_seg.py --out data/mem_synth/train.jsonl --num 3000 \
      --filler-hf arxiv,wiki,dialog --filler-hf-num 40000 \
      --segment-tokens 1800 --segments 8,16,32,64 --tokenizer ./models/Qwen3.5-4B
  # segments=8,16,32,64 -> ~14k/29k/58k/115k-token histories (at 1800 tok/segment)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# Reuse filler/tokenizer utilities from the flat synthesizer (same package).
from generate_memory_data import (
    FILLER_FALLBACK, build_hf_filler, load_filler, maybe_tokenizer,
    CITIES, PETS, PROJECTS, COLORS, JOBS, FIRST_NAMES, MONTHS,
)

GADGETS = ["a mechanical keyboard", "a film camera", "a standing desk", "an e-ink tablet",
           "a espresso machine", "a noise-cancelling headset", "a turntable"]
BOOKS = ["The Glass Bead Game", "Pale Fire", "Gormenghast", "The Dispossessed",
         "A Canticle for Leibowitz", "Piranesi", "The Peripheral"]


def _rand_date_seq(rng: random.Random, n: int) -> List[str]:
    """n strictly increasing human dates."""
    keys = sorted(rng.sample(range(0, 1600), n))  # day offsets, unique & ordered
    out = []
    base_y = 2021
    for k in keys:
        y = base_y + k // 360
        m = (k % 360) // 30
        d = (k % 30) + 1
        out.append(f"{MONTHS[m]} {d}, {y}")
    return out


# fact_type -> (user_template, question_template, value_pool)
FACT_TYPES: Dict[str, Tuple[str, str, List[str]]] = {
    "pet":     ("I adopted {val}.",                         "On {date}, what pet did I adopt?", PETS),
    "trip":    ("I traveled to {val}.",                     "On {date}, where did I travel?", CITIES),
    "project": ("I kicked off a project called {val}.",     "On {date}, what project did I start?", PROJECTS),
    "gadget":  ("I bought {val}.",                          "On {date}, what did I buy?", GADGETS),
    "book":    ("I started reading '{val}'.",               "On {date}, what book did I start reading?", BOOKS),
    "color":   ("I repainted my study {val}.",              "On {date}, what color did I repaint my study?", COLORS),
}

ACKS = ["Got it.", "Noted.", "I'll remember that.", "Sounds good.", "Understood."]


def _filler_turns(rng, pool, n) -> List[Dict[str, str]]:
    turns = []
    for _ in range(n):
        turns.append({"role": "user", "content": rng.choice(pool)})
        turns.append({"role": "assistant", "content": rng.choice(
            ["Thanks for sharing.", "Interesting!", "Noted.", "Haha, nice."])})
    return turns


def build_history(rng: random.Random, filler_pool: List[str], n_segments: int,
                  segment_chars: int) -> Tuple[List[Dict], List[Dict], Dict]:
    """Return (segments, final_qa, meta)."""
    person = rng.choice(FIRST_NAMES)
    dates = _rand_date_seq(rng, n_segments)

    # Plan cross-segment structure:
    conflict_type = rng.choice(list(FACT_TYPES.keys()))
    conflict_pool = FACT_TYPES[conflict_type][2]
    seg_a, seg_b = sorted(rng.sample(range(n_segments), 2))  # conflict stated in seg_a then seg_b
    val_a = rng.choice(conflict_pool)
    val_b = rng.choice([v for v in conflict_pool if v != val_a])

    colleague_seg = rng.randrange(n_segments)
    colleague = rng.choice([n for n in FIRST_NAMES if n != person])
    colleague_city = rng.choice(CITIES)

    rule_seg = rng.randrange(n_segments)
    rule_style = rng.choice(["all uppercase", "all lowercase", "ending with three exclamation marks"])

    segments: List[Dict] = []
    last_primary: Tuple[str, str] = ("", "")  # (date, fact_desc) for temporal

    for si in range(n_segments):
        date = dates[si]
        facts: List[Tuple[str, str, str]] = []  # (user_utterance, q, a)

        # Conflict attribute appears in its two designated segments.
        if si == seg_a or si == seg_b:
            v = val_a if si == seg_a else val_b
            ut, qt, _ = FACT_TYPES[conflict_type]
            facts.append((ut.format(val=v), qt.format(date=date), v))
        # Colleague (multi-hop) in its segment.
        if si == colleague_seg:
            facts.append((f"I met my colleague {colleague}, who lives in {colleague_city}.",
                          f"On {date}, which colleague did I meet?", colleague))
        # A couple of unique local facts.
        for ft in rng.sample([k for k in FACT_TYPES if k != conflict_type], rng.randint(1, 2)):
            ut, qt, pool = FACT_TYPES[ft]
            v = rng.choice(pool)
            facts.append((ut.format(val=v), qt.format(date=date), v))

        last_primary = (date, facts[-1][0].rstrip("."))

        # Build the segment conversation: facts (as dated user turns) + filler, to ~segment_chars.
        turns: List[Dict[str, str]] = [{"role": "user", "content": f"(Session on {date})"},
                                       {"role": "assistant", "content": "Noted the date."}]
        # interleave facts with filler
        fact_turns = []
        for (utt, _q, _a) in facts:
            fact_turns.append({"role": "user", "content": f"On {date}, {utt}"})
            fact_turns.append({"role": "assistant", "content": rng.choice(ACKS)})
        # rule statement (TTL) lives in its segment
        if si == rule_seg:
            fact_turns.append({"role": "user",
                               "content": f"From now on, whenever I ask you to summarize, answer {rule_style}."})
            fact_turns.append({"role": "assistant", "content": "Understood, I'll follow that."})

        # distribute facts among filler
        n_fill = max(2, segment_chars // 120)
        chunk = max(1, n_fill // (len(fact_turns) // 2 + 1))
        for k in range(0, len(fact_turns), 2):
            turns.extend(_filler_turns(rng, filler_pool, chunk))
            turns.extend(fact_turns[k:k + 2])
        # pad to budget
        while sum(len(t["content"]) for t in turns) < segment_chars:
            turns.extend(_filler_turns(rng, filler_pool, 4))

        # Per-segment QA (about THIS segment's facts; answerable from the segment alone).
        seg_qa = [{"question": q, "answer": a, "type": "segment_retrieval",
                   "segment": si} for (_u, q, a) in facts]

        segments.append({"date": date, "history_turns": turns, "qa": seg_qa})

    # Cross-segment final QA (exercises reading accumulated W).
    _recent = last_primary[1][2:].strip() if last_primary[1].startswith("I ") else last_primary[1]
    final_qa = [
        {"question": f"My {conflict_type} changed over time. What is it now?",
         "answer": val_b, "type": "conflict", "hops": 1},
        {"question": f"What was my {conflict_type} BEFORE the most recent change?",
         "answer": val_a, "type": "conflict", "hops": 1},
        {"question": f"Which city does my colleague {colleague} live in?",
         "answer": colleague_city, "type": "multi_hop", "hops": 2},
        {"question": "What is the most recent thing I told you I did, and on what date?",
         "answer": f"{last_primary[1]} on {last_primary[0]}", "type": "temporal", "hops": 1},
        {"question": "Summarize who I am in one line.",
         "answer": _styled(f"{person} most recently {_recent}", rule_style),
         "type": "ttl", "hops": 1, "rule": rule_style},
    ]
    meta = {"person": person, "n_segments": n_segments, "conflict_type": conflict_type}
    return segments, final_qa, meta


def _styled(s: str, style: str) -> str:
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
    ap.add_argument("--segments", default="8,16,32,64", help="comma list of segment counts to sample")
    ap.add_argument("--segment-tokens", type=int, default=1800, help="~tokens per segment (<= context_seq_length)")
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
            rec = {
                "id": f"mem-{i:06d}",
                "n_segments": n_seg,
                "approx_tokens": approx_tokens(segments, tok),
                "segments": segments,
                "final_qa": final_qa,
                "teacher_prompt_template": (
                    "Use ONLY the conversation history below.\n=== HISTORY ===\n{history}\n"
                    "=== END ===\nQuestion: {question}\nAnswer:"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{args.num} (n_seg={n_seg}, approx_tokens={rec['approx_tokens']})", flush=True)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
