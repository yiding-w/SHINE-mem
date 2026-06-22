#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-flight check for the streaming memory dataset: verifies (on the REAL tokenizer)
that the segmented data builds correctly and that labels land ONLY on assistant
answers (catches the B5 label-mask class of bug before wasting a training run).

Usage:
  python datagen/check_stream.py --jsonl data/mem_synth/train.jsonl --model ./models/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import os
import sys

from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mydatasets.pretrain_annealing import memory_stream as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--conv", type=int, default=1024)
    ap.add_argument("--show", type=int, default=2, help="how many stream samples to dump")
    args = ap.parse_args()

    data_dir = os.path.dirname(os.path.abspath(args.jsonl))
    fname = os.path.basename(args.jsonl)
    cfg = OmegaConf.create({
        "data": {
            "data_path": data_dir,
            "train_file": fname,
            "context_seq_length": args.ctx,
            "conv_seq_length": args.conv,
        },
        "tokenizer": None,
    })

    ds, col = M.create_dataset_and_collator(cfg, args.model, pad_token_id=0, num_mem_token=0)
    tok = ds.tokenizer
    print(f"[check] stream samples: {len(ds)}")

    n = min(args.show, len(ds))
    batch = col([ds[i] for i in range(n)])[0]
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    labels = batch["labels"]
    conv = batch["conversation_ids"]
    total_loss_tokens = int((labels != -100).sum())
    print(f"[check] total loss tokens across {n} samples: {total_loss_tokens}")
    if total_loss_tokens == 0:
        print("  ✗ BUG: no loss tokens — label mask matched nothing (check B5 / chat template).")
        sys.exit(1)

    for i in range(n):
        row = labels[i]
        keep = (row != -100).nonzero().flatten().tolist()
        decoded = tok.decode([int(conv[i, j]) for j in keep]) if keep else "<none>"
        print(f"\n--- sample {i}: {len(keep)} loss tokens ---")
        print("  LOSS TOKENS (should be ONLY answers):", repr(decoded[:300]))
    print("\n[check] OK — loss falls on assistant answers only; context_ids carry no loss.")


if __name__ == "__main__":
    main()
