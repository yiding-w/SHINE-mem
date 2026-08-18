#!/usr/bin/env python
"""Measure Base(A), SHINE Memory(B), and explicit Context(C) recovery."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ControlSHINE.controlshine.metrics import answer_match, recoverability_label  # noqa: E402
from ControlSHINE.controlshine.runtime import load_pretrain_runtime  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=10, help="Use -1 for all records")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def _prompt(question: str, context: str | None = None) -> str:
    if context is None:
        return question
    return (
        "Use the following context to answer the query. Return only the answer.\n\n"
        f"Context: {context}\n\nQuery: {question}"
    )


@torch.inference_mode()
def _generate(model, tokenizer, device, prompt, loradict, max_length, max_new_tokens):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=max_length,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model.metamodel.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_cache=True,
        ignore_mem_token=True,
        loradict=loradict,
    )
    return tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


@torch.inference_mode()
def _memory_lora(model, tokenizer, metalora, cfg, device, text):
    encoded = tokenizer(
        [text],
        max_length=cfg.test.context_max_length,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )
    return model.generate_lora_dict(
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        metalora,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-8B SHINE checkpoint")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    cfg, model, metalora, tokenizer = load_pretrain_runtime(
        args.runtime_config, args.checkpoint_dir, device
    )

    rows = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if args.limit >= 0 and len(rows) >= args.limit:
                    break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "source_only.jsonl"
    counts = Counter()
    start_all = time.perf_counter()

    with output_file.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            question = row["question"]
            base = _generate(
                model, tokenizer, device, _prompt(question), None,
                cfg.test.conversation_max_length, args.max_new_tokens,
            )
            lora = _memory_lora(
                model, tokenizer, metalora, cfg, device, row["memory"]["text"]
            )
            memory = _generate(
                model, tokenizer, device, _prompt(question), lora,
                cfg.test.conversation_max_length, args.max_new_tokens,
            )
            context = _generate(
                model, tokenizer, device, _prompt(question, row["context"]["text"]), None,
                cfg.test.conversation_max_length, args.max_new_tokens,
            )
            predictions = {"base": base, "memory": memory, "context": context}
            matches = {
                source: answer_match(
                    predictions[source], row[source]["answer"], row[source].get("aliases", [])
                )
                for source in ("base", "memory", "context")
            }
            label = recoverability_label(matches["base"], matches["memory"], matches["context"])
            counts["total"] += 1
            counts[label] += 1
            for source, matched in matches.items():
                counts[f"{source}_recoverable"] += int(matched)
            result = {
                "sample_id": row["sample_id"],
                "targets": {source: row[source]["answer"] for source in ("base", "memory", "context")},
                "predictions": predictions,
                "matches": matches,
                "recoverability": label,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[ControlSHINE] {index}/{len(rows)} "
                f"A={matches['base']} B={matches['memory']} C={matches['context']} "
                f"label={label}",
                flush=True,
            )

    total = counts["total"]
    summary = {
        "checkpoint": str(Path(args.checkpoint_dir).resolve()),
        "num_examples": total,
        "counts": dict(counts),
        "rates": {
            key: counts[key] / total if total else 0.0
            for key in ("base_recoverable", "memory_recoverable", "context_recoverable", "fully_recoverable")
        },
        "wall_seconds": time.perf_counter() - start_all,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

