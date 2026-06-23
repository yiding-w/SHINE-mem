#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert the intermediate memory-synth JSONL (from generate_memory_data.py) into
SHINE_V2's SFT format consumed by mydatasets/sft/msmarco_mqa.py:

    {"query_id": int, "context": str, "conversations": [{"question","answer"}, ...]}

This is the Phase-1 (single-window, no streaming) path: the whole history is
flattened into one `context` string and truncated/padded to context_seq_length by
the SFT collator. Use it for histories up to ~context_seq_length tokens
(e.g. 8k). For 16k–128k use the streaming dataset (Phase 2).

Example:
  python datagen/to_v2_sft.py --in data/mem_synth/train.jsonl \
      --out old_data/mem_sft/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os


def flatten_history(turns) -> str:
    lines = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-history-chars", type=int, default=0,
                    help="optional hard cap on context chars (0 = no cap; collator truncates by tokens)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_in = n_out = 0
    with open(args.inp, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            ctx = flatten_history(rec["history_turns"])
            if args.max_history_chars > 0 and len(ctx) > args.max_history_chars:
                ctx = ctx[-args.max_history_chars:]  # keep most recent (or switch to streaming)
            convs = [{"question": q["question"], "answer": q["answer"]} for q in rec["qa"]]
            out = {
                "query_id": n_out,
                "context": ctx,
                "conversations": convs,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Converted {n_in} -> {n_out} SFT records: {args.out}")


if __name__ == "__main__":
    main()
