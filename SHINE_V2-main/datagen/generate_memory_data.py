#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Long-history memory training-data synthesizer for SHINE_V2.

Goal: produce VERY long conversation histories (8k–128k tokens) with multi-type QA
whose answers are *guaranteed derivable* from the history (reliable gold), while the
surface text is natural — real corpus text as background/distractor + structured
facts injected as conversation turns at controlled depths.

Design (hybrid, gold-reliable):
  1. Build a structured "fact store": entities with attributes, dated events,
     relations, attribute UPDATES (→ conflict / latest-value), and a stated user
     rule (→ test-time-learning).
  2. Render it as a multi-session user/assistant conversation, interleaving real
     filler turns. Each injected fact records its char offset → needle depth.
  3. Generate QA from the store covering 5 capability types, gold computed from
     the store (not guessed).

Output: one intermediate JSON object per line (NOT yet V2 format — a separate
converter maps this to SFT JSONL or the streaming chunked dataset):

  {
    "id": str,
    "target_tokens": int,
    "approx_tokens": int,
    "history_turns": [{"role": "user"|"assistant", "content": str}, ...],
    "qa": [{"question","answer","type","hops","needle_depth"}],
    "teacher_prompt_template": "...{history}...{question}..."   # for distillation
  }

Filler source (real data, for background/distractor):
  - --filler-hf : stream real text from HF datasets (no full download). Presets:
        wiki   = wikimedia/wikipedia (20231101.en)
        arxiv  = ccdv/arxiv-summarization (article)
        dialog = daily_dialog (conversational)
        soda   = allenai/soda (conversational)
    or a custom 'repo:config:split:field[:list]' spec. Comma-separate to mix.
  - --filler-dir : a local dir of .txt files.
  - if neither is given, a small built-in placeholder filler is used (smoke only).
  NOTE: only the *background* is real; the queryable facts/QA are synthetic
  (so gold stays 100% reliable). Synthetic entity names won't collide with real text.

Example:
  python datagen/generate_memory_data.py \
      --out data/mem_synth/train.jsonl --num 2000 \
      --filler-hf arxiv,wiki,dialog --filler-hf-num 30000 \
      --tokenizer ./models/Qwen3.5-4B \
      --lengths 8000,16000,32000,64000,128000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Structured world: entities, attributes, events, updates, relations
# ---------------------------------------------------------------------------

FIRST_NAMES = ["Alex", "Bao", "Chen", "Dana", "Elena", "Farid", "Grace", "Hiro",
               "Ivan", "Jun", "Kira", "Liang", "Mei", "Nadia", "Omar", "Priya",
               "Qing", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei", "Xena",
               "Yara", "Zane"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Reykjavik",
          "Chiang Mai", "Montreal", "Tallinn", "Valencia", "Hobart", "Kigali"]
JOBS = ["radiologist", "cartographer", "luthier", "agronomist", "actuary",
        "glassblower", "epidemiologist", "hydrologist", "archivist", "sommelier"]
PETS = ["a corgi named Mochi", "a tabby cat named Biscuit", "an axolotl named Pixel",
        "a parrot named Cobalt", "a tortoise named Sage", "a beagle named Rumi"]
COLORS = ["teal", "amber", "maroon", "chartreuse", "indigo", "ochre", "vermilion"]
PROJECTS = ["Project Halcyon", "the Meridian rollout", "the Solstice audit",
            "the Beacon migration", "the Aurora pilot", "the Tempo refactor"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

FILLER_FALLBACK = [
    "I spent the morning reorganizing my bookshelf by color instead of author.",
    "The neighborhood market had an unusually good batch of stone fruit today.",
    "I've been trying to cut down on coffee but the afternoons are rough.",
    "Watched a documentary about deep-sea bioluminescence last night.",
    "The weather finally turned and I could open the windows again.",
    "Spent an hour debugging only to find a missing comma. Classic.",
    "My commute got rerouted because of construction near the bridge.",
    "Trying a new sourdough starter; day three and it's bubbling nicely.",
]


def _rand_date(rng: random.Random) -> Tuple[int, str]:
    """Return (sortable_key, human_string)."""
    y, m, d = rng.randint(2021, 2025), rng.randint(1, 12), rng.randint(1, 28)
    return y * 10000 + m * 100 + d, f"{MONTHS[m-1]} {d}, {y}"


@dataclass
class World:
    rng: random.Random
    person: str = ""
    # attribute -> list of (date_key, date_str, value); last = current
    attrs: Dict[str, List[Tuple[int, str, str]]] = field(default_factory=dict)
    events: List[Tuple[int, str, str]] = field(default_factory=list)  # (date_key, date_str, desc)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (other, rel, attr_value)
    rule: str = ""
    rule_answer_style: str = ""

    def build(self) -> None:
        rng = self.rng
        self.person = rng.choice(FIRST_NAMES)
        # Base attributes (some will receive updates -> conflict)
        self.attrs["city"] = [(*_rand_date(rng), rng.choice(CITIES))]
        self.attrs["job"] = [(*_rand_date(rng), rng.choice(JOBS))]
        self.attrs["pet"] = [(*_rand_date(rng), rng.choice(PETS))]
        self.attrs["favorite_color"] = [(*_rand_date(rng), rng.choice(COLORS))]
        self.attrs["current_project"] = [(*_rand_date(rng), rng.choice(PROJECTS))]

        # Updates: reassign 1-3 attributes at a later date -> latest-value/conflict QA
        for attr in rng.sample(list(self.attrs.keys()), rng.randint(1, 3)):
            pool = {"city": CITIES, "job": JOBS, "pet": PETS,
                    "favorite_color": COLORS, "current_project": PROJECTS}[attr]
            old = self.attrs[attr][-1][2]
            new = rng.choice([v for v in pool if v != old])
            dk, ds = _rand_date(rng)
            self.attrs[attr].append((dk, ds, new))
            self.attrs[attr].sort(key=lambda t: t[0])

        # Events with dates -> temporal QA
        verbs = ["adopted {pet}", "moved to {city}", "started {proj}",
                 "gave a talk on cartography", "ran a half-marathon",
                 "broke their wrist skating", "joined a pottery class"]
        for _ in range(rng.randint(4, 7)):
            dk, ds = _rand_date(rng)
            desc = rng.choice(verbs).format(
                pet=self.attrs["pet"][-1][2], city=self.attrs["city"][-1][2],
                proj=self.attrs["current_project"][-1][2])
            self.events.append((dk, ds, desc))
        self.events.sort(key=lambda t: t[0])

        # Relations -> multi-hop QA (person works with OTHER who lives in CITY)
        other = rng.choice([n for n in FIRST_NAMES if n != self.person])
        other_city = rng.choice(CITIES)
        self.relations.append((other, "works with", other_city))

        # Test-time-learning rule (styles chosen so gold is deterministically realizable)
        self.rule_answer_style = rng.choice(["all uppercase", "all lowercase",
                                             "ending with three exclamation marks"])
        self.rule = (f"From now on, whenever I ask you to summarize, always answer "
                     f"{self.rule_answer_style}.")

    def base_summary(self) -> str:
        return (f"{self.person} is a {self.attrs['job'][-1][2]} in "
                f"{self.attrs['city'][-1][2]} working on {self.attrs['current_project'][-1][2]}.")

    def styled_summary(self) -> str:
        s = self.base_summary()
        if self.rule_answer_style == "all uppercase":
            return s.upper()
        if self.rule_answer_style == "all lowercase":
            return s.lower()
        return s.rstrip(".") + "!!!"


# ---------------------------------------------------------------------------
# Rendering: structured world -> conversation turns with real filler
# ---------------------------------------------------------------------------

def _filler_turns(rng: random.Random, filler_pool: List[str], n: int) -> List[Dict[str, str]]:
    turns = []
    for _ in range(n):
        snippet = rng.choice(filler_pool)
        turns.append({"role": "user", "content": snippet})
        turns.append({"role": "assistant", "content": rng.choice(
            ["Got it, thanks for sharing.", "Noted!", "Interesting, tell me more sometime.",
             "Sounds good.", "Haha, I hear you."])})
    return turns


def render_history(world: World, rng: random.Random, filler_pool: List[str],
                   target_tokens: int, approx_chars_per_tok: int = 4
                   ) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """Interleave fact-bearing turns with filler to reach ~target_tokens.

    Returns (turns, needle_depths) where needle_depths maps a fact key -> depth in (0,1].
    """
    target_chars = target_tokens * approx_chars_per_tok

    # Fact-bearing user turns (each states ONE queryable fact in natural language).
    fact_turns: List[Tuple[str, str]] = []  # (fact_key, user_utterance)
    for attr, hist in world.attrs.items():
        for (_, ds, val) in hist:  # state each historical value (incl. updates) in order
            fact_turns.append((f"attr:{attr}:{val}",
                               f"By the way, as of {ds}, my {attr.replace('_',' ')} is {val}."))
    for (dk, ds, desc) in world.events:
        fact_turns.append((f"event:{ds}:{desc}", f"On {ds}, I {desc}."))
    for (other, rel, ocity) in world.relations:
        fact_turns.append((f"rel:{other}", f"I {rel} {other}, who lives in {ocity}."))
    fact_turns.append(("rule", world.rule))
    rng.shuffle(fact_turns)  # spread facts to varied depths

    turns: List[Dict[str, str]] = []
    needle_depths: Dict[str, float] = {}
    cur_chars = 0
    # Distribute facts roughly evenly across the target length.
    gap = max(1, len(fact_turns))
    filler_between = max(1, target_chars // (gap * 2 * 60))  # ~60 chars per filler turn

    for i, (key, utt) in enumerate(fact_turns):
        turns.extend(_filler_turns(rng, filler_pool, filler_between))
        turns.append({"role": "user", "content": utt})
        turns.append({"role": "assistant", "content": "Thanks, I'll remember that."})
        cur_chars = sum(len(t["content"]) for t in turns)
        needle_depths[key] = round(cur_chars / max(1, target_chars), 3)

    # Pad with filler to reach target length.
    while sum(len(t["content"]) for t in turns) < target_chars:
        turns.extend(_filler_turns(rng, filler_pool, 4))

    return turns, needle_depths


# ---------------------------------------------------------------------------
# QA generation (5 types), gold derived from the store
# ---------------------------------------------------------------------------

def make_qa(world: World, depths: Dict[str, float], rng: random.Random) -> List[Dict[str, Any]]:
    qa: List[Dict[str, Any]] = []
    p = world.person

    # 1) single-hop retrieval (current value)
    attr = rng.choice(list(world.attrs.keys()))
    cur = world.attrs[attr][-1][2]
    qa.append({"question": f"What is my current {attr.replace('_',' ')}?",
               "answer": cur, "type": "single_hop", "hops": 1,
               "needle_depth": depths.get(f"attr:{attr}:{cur}", -1)})

    # 2) conflict / latest-value (attribute that was updated)
    updated = [a for a, h in world.attrs.items() if len(h) > 1]
    if updated:
        a = rng.choice(updated)
        latest = world.attrs[a][-1][2]
        prev = world.attrs[a][-2][2]
        qa.append({"question": f"My {a.replace('_',' ')} changed over time. What is it now?",
                   "answer": latest, "type": "conflict", "hops": 1,
                   "needle_depth": depths.get(f"attr:{a}:{latest}", -1)})
        qa.append({"question": f"What was my {a.replace('_',' ')} BEFORE the most recent change?",
                   "answer": prev, "type": "conflict", "hops": 1,
                   "needle_depth": depths.get(f"attr:{a}:{prev}", -1)})

    # 3) temporal (latest event / ordering)
    if world.events:
        last_ev = world.events[-1]
        qa.append({"question": "What is the most recent thing I told you I did, and on what date?",
                   "answer": f"{last_ev[2]} on {last_ev[1]}", "type": "temporal", "hops": 1,
                   "needle_depth": depths.get(f"event:{last_ev[1]}:{last_ev[2]}", -1)})

    # 4) multi-hop (relation -> other's city)
    if world.relations:
        other, _, ocity = world.relations[0]
        qa.append({"question": f"Which city does the person I work with live in?",
                   "answer": ocity, "type": "multi_hop", "hops": 2,
                   "needle_depth": depths.get(f"rel:{other}", -1)})

    # 5) test-time-learning (apply the stated rule -> concrete styled gold)
    qa.append({"question": "Summarize who I am in one line.",
               "answer": world.styled_summary(),
               "type": "ttl", "hops": 1, "needle_depth": depths.get("rule", -1),
               "rule": world.rule_answer_style})

    return qa


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# HuggingFace filler presets: name -> (repo, config, split, field, is_list_of_str)
HF_FILLER_PRESETS: Dict[str, Tuple[str, Optional[str], str, str, bool]] = {
    "wiki":   ("wikimedia/wikipedia", "20231101.en", "train", "text", False),
    "arxiv":  ("ccdv/arxiv-summarization", None, "train", "article", False),
    "dialog": ("li2017dailydialog/daily_dialog", None, "train", "dialog", True),  # conversational (parquet ref)
    "soda":   ("allenai/soda", None, "train", "dialogue", True),   # conversational
}


def _to_snippets(text: str) -> List[str]:
    """Split text into 20–400 char conversational snippets (paragraph then sentence)."""
    out: List[str] = []
    for para in text.replace("\r", " ").split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= 400:
            if len(para) >= 20:
                out.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(buf) + len(sent) + 1 <= 400:
                buf = (buf + " " + sent).strip()
            else:
                if len(buf) >= 20:
                    out.append(buf)
                buf = sent
        if len(buf) >= 20:
            out.append(buf)
    return out


def _resolve_hf_spec(name: str) -> Tuple[str, Optional[str], str, str, bool]:
    """Accept a preset name or a custom 'repo:config:split:field[:list]' spec."""
    if name in HF_FILLER_PRESETS:
        return HF_FILLER_PRESETS[name]
    parts = name.split(":")
    repo = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    field_name = parts[3] if len(parts) > 3 and parts[3] else "text"
    is_list = len(parts) > 4 and parts[4].lower() in ("1", "list", "true")
    return repo, config, split, field_name, is_list


def build_hf_filler(specs: List[str], total: int, seed: int) -> List[str]:
    """Stream real text from HF datasets (no full download) into a snippet pool."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[warn] `datasets` not available ({e}); skipping --filler-hf.")
        return []
    pool: List[str] = []
    per_source = max(1, total // max(1, len(specs)))
    for name in specs:
        repo, config, split, field_name, is_list = _resolve_hf_spec(name)
        got = 0
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
        except Exception as e:
            # datasets>=4 removed loading scripts (e.g. daily_dialog). HF auto-converts
            # every dataset to parquet on the 'refs/convert/parquet' ref — retry there.
            try:
                ds = load_dataset(repo, config, split=split, streaming=True,
                                  revision="refs/convert/parquet")
                print(f"[filler-hf] {name}: loaded via refs/convert/parquet (script fallback)")
            except Exception as e2:
                print(f"[warn] failed to load HF dataset '{name}' ({repo},{config},{split}): "
                      f"{e} | parquet-fallback: {e2}")
                continue
        for ex in ds:
            val = ex.get(field_name)
            if val is None:
                continue
            text = " \n".join(map(str, val)) if is_list else str(val)
            for snip in _to_snippets(text):
                pool.append(snip)
                got += 1
                if got >= per_source:
                    break
            if got >= per_source:
                break
        print(f"[filler-hf] {name}: collected {got} snippets")
    return pool


def load_filler(filler_dir: Optional[str]) -> List[str]:
    if not filler_dir or not os.path.isdir(filler_dir):
        return []
    pool: List[str] = []
    for fn in os.listdir(filler_dir):
        if fn.endswith(".txt"):
            with open(os.path.join(filler_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
                pool.extend(_to_snippets(f.read()))
    return pool


def maybe_tokenizer(path: Optional[str]):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as e:
        print(f"[warn] tokenizer load failed ({e}); using char/4 length estimate.")
        return None


def approx_tokens(turns: List[Dict[str, str]], tok) -> int:
    text = "\n".join(t["content"] for t in turns)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text) // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--filler-dir", default=None, help="dir of real .txt for background/distractor")
    ap.add_argument("--filler-hf", default=None,
                    help="comma-separated HF filler sources: presets {wiki,arxiv,dialog,soda} "
                         "or custom 'repo:config:split:field[:list]'. Streamed, no full download.")
    ap.add_argument("--filler-hf-num", type=int, default=20000,
                    help="total real snippets to pull from --filler-hf sources")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer path for accurate length")
    ap.add_argument("--lengths", default="8000,16000,32000,64000,128000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    filler_pool = load_filler(args.filler_dir)
    if args.filler_hf:
        specs = [s.strip() for s in args.filler_hf.split(",") if s.strip()]
        filler_pool += build_hf_filler(specs, args.filler_hf_num, args.seed)
    if not filler_pool:
        print("[warn] no real filler (--filler-dir/--filler-hf); using built-in placeholder filler.")
        filler_pool = list(FILLER_FALLBACK)
    print(f"[filler] pool size = {len(filler_pool)} snippets")
    tok = maybe_tokenizer(args.tokenizer)
    length_buckets = [int(x) for x in args.lengths.split(",")]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_written = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in range(args.num):
            target = rng.choice(length_buckets)
            world = World(rng=random.Random(rng.randint(0, 2**31)))
            world.build()
            turns, depths = render_history(world, rng, filler_pool, target)
            qa = make_qa(world, depths, rng)
            rec = {
                "id": f"mem-{i:06d}",
                "target_tokens": target,
                "approx_tokens": approx_tokens(turns, tok),
                "history_turns": turns,
                "qa": qa,
                # For distillation: a long-context teacher gets the full history in-prompt.
                "teacher_prompt_template": (
                    "You are given the full conversation history below. Use ONLY it to answer.\n"
                    "=== HISTORY ===\n{history}\n=== END HISTORY ===\n"
                    "Question: {question}\nAnswer:"),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if (i + 1) % 200 == 0:
                print(f"  wrote {i+1}/{args.num} (last approx_tokens={rec['approx_tokens']})", flush=True)

    print(f"Done: {n_written} samples -> {args.out}")


if __name__ == "__main__":
    main()
