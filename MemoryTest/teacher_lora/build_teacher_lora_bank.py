#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from pathlib import Path

import torch

from MemoryTest.training.lora_sft_utils import (
    load_frozen_lora_model,
    read_facts,
    resolve_path,
    train_lora_dict,
)
from MemoryTest.training.run_lora_upper_bound import evaluate_facts
from MemoryTest.teacher_lora.teacher_lora_utils import (
    make_context_id,
    make_context_payload,
    resolve_context_format,
    sample_context_rows,
    save_teacher_lora_entry,
    write_bank_index,
)


LOGGER = logging.getLogger("build_teacher_lora_bank")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline teacher LoRA bank for SHINE teacher-alignment post-training.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--facts-path", "--train-file", dest="facts_path", type=str, default="MemoryTest/json_data/semantic_facts.json")
    parser.add_argument("--output-dir", type=str, default="MemoryTest/teacher_loras/qa_sft_rank8")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--fact-counts", type=int, nargs="+", default=[4, 8, 20])
    parser.add_argument("--contexts-per-count", type=int, default=20)
    parser.add_argument("--context-sampling", choices=["random", "head", "window"], default="random")
    parser.add_argument("--context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-objective", choices=["qa_sft", "ntp"], default="qa_sft")
    parser.add_argument("--variants-per-fact", type=int, default=5)
    parser.add_argument("--ntp-context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--ntp-context-variants", type=int, default=5)
    parser.add_argument("--ntp-record-mode", choices=["packed", "per_fact", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    facts = read_facts(args.facts_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    runtime_args, cfg, device, model, tokenizer, _ = load_frozen_lora_model(
        args.runtime_config,
        rank=args.rank,
        device_name=args.device,
        gpu_id=args.gpu_id,
    )
    model.eval()

    entries = []
    existing_entries = []
    index_path = output_dir / "index.json"
    if args.skip_existing and index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            existing_entries = list(payload.get("entries", []))
            entries.extend(existing_entries)
        except Exception:
            LOGGER.warning("Could not read existing index; rebuilding it: %s", index_path)

    existing_ids = {entry.get("context_id") for entry in existing_entries}
    total_contexts = len(args.fact_counts) * args.contexts_per_count
    progress = tqdm(total=total_contexts, desc="teacher LoRA bank", dynamic_ncols=True, leave=True) if tqdm is not None else None

    context_index = 0
    for fact_count in args.fact_counts:
        for local_idx in range(args.contexts_per_count):
            context_id = make_context_id(context_index)
            context_index += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix({"context": context_id, "facts": fact_count})
            if args.skip_existing and context_id in existing_ids and (output_dir / context_id / "teacher_lora.pt").exists():
                LOGGER.info("Skipping existing teacher LoRA: %s", context_id)
                continue

            context_rows = sample_context_rows(
                facts,
                fact_count,
                rng=rng,
                mode=args.context_sampling,
                start_index=local_idx,
            )
            context_format = resolve_context_format(args.context_format, context_index)
            context_payload = make_context_payload(
                context_id,
                context_rows,
                context_format,
                source_path=args.facts_path,
                extra={
                    "context_sampling": args.context_sampling,
                    "seed": args.seed,
                },
            )

            lora_dict = model.init_lora_dict(int(args.rank), scale=cfg.metanetwork.transformer_cfg.scale, device=device)

            def epoch_eval_callback(epoch: int, current_lora_dict):
                model.eval()
                result = evaluate_facts(
                    f"epoch_{epoch}_teacher_train_memorization",
                    context_rows,
                    model,
                    tokenizer,
                    current_lora_dict,
                    device,
                    args.max_new_tokens,
                    runtime_args.conversation_max_length,
                )
                model.train()
                return {
                    "epoch": epoch,
                    "correct": result["correct"],
                    "total": result["total"],
                    "accuracy": result["accuracy"],
                    "wrong_examples": result["wrong_examples"],
                }

            model.train()
            train_stats = train_lora_dict(
                model=model,
                tokenizer=tokenizer,
                lora_dict=lora_dict,
                facts=context_rows,
                device=device,
                seed=args.seed + context_index * 1009,
                variants_per_fact=args.variants_per_fact,
                max_length=args.max_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                training_objective=args.training_objective,
                ntp_context_format=args.ntp_context_format,
                ntp_context_variants=args.ntp_context_variants,
                ntp_record_mode=args.ntp_record_mode,
                progress_label=f"{context_id} facts={fact_count}",
                epoch_eval_callback=epoch_eval_callback,
            )
            best_lora_dict = train_stats.pop("best_lora_dict")
            model.eval()
            train_result = evaluate_facts(
                "teacher_train_memorization_best_lora",
                context_rows,
                model,
                tokenizer,
                best_lora_dict,
                device,
                args.max_new_tokens,
                runtime_args.conversation_max_length,
            )
            meta = {
                **context_payload,
                "teacher": {
                    "rank": args.rank,
                    "objective": args.training_objective,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "variants_per_fact": args.variants_per_fact,
                    "ntp_context_format": args.ntp_context_format,
                    "ntp_context_variants": args.ntp_context_variants,
                    "ntp_record_mode": args.ntp_record_mode,
                    "train_stats": train_stats,
                    "train_result_summary": {
                        "correct": train_result["correct"],
                        "total": train_result["total"],
                        "accuracy": train_result["accuracy"],
                    },
                    "wrong_examples": train_result["wrong_examples"],
                },
            }
            saved_meta = save_teacher_lora_entry(output_dir, context_id, best_lora_dict, meta)
            entries = [entry for entry in entries if entry.get("context_id") != context_id]
            entries.append(
                {
                    "context_id": context_id,
                    "num_facts": fact_count,
                    "fact_ids": saved_meta["fact_ids"],
                    "context_format": context_format,
                    "lora_path": saved_meta["lora_path"],
                    "meta_path": saved_meta["meta_path"],
                    "objective": args.training_objective,
                    "best_epoch": train_stats.get("best_epoch"),
                    "train_accuracy": train_result["accuracy"],
                }
            )
            write_bank_index(
                output_dir,
                {
                    "config": vars(args),
                    "num_entries": len(entries),
                    "entries": entries,
                },
            )
            del lora_dict, best_lora_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if progress is not None:
        progress.close()
    write_bank_index(
        output_dir,
        {
            "config": vars(args),
            "num_entries": len(entries),
            "entries": entries,
        },
    )
    print(f"Wrote teacher LoRA bank to {output_dir}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
