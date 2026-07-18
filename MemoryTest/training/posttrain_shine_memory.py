#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import string
from pathlib import Path

import torch

from MemoryTest.prepare_data.prompt_templates import format_answer, question_prompt, reconstruction_prompt
from MemoryTest.evaluation.metrics import make_eval_row, summarize_examples
from MemoryTest.training.lora_sft_utils import load_runtime_args
from MemoryTest.training.recurrent_data import (
    accumulated_qa,
    load_recurrent_dataset,
    sample_training_stream,
    sample_turn_qa,
)
from MemoryTest.training.shine_train_utils import (
    append_jsonl,
    cast_floating_tensors,
    clamp_lora_tensors,
    compute_combined_lora_loss,
    lora_tensor_stats,
    recurrent_memory_norm_stats,
    resolve_path,
    save_posttrain_checkpoint,
    set_posttrain_requires_grad,
    trainable_generate_context_lora,
    trainable_update_recurrent_memory,
)


LOGGER = logging.getLogger("posttrain_shine_memory")

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train SHINE on MemoryTest semantic facts.")
    parser.add_argument("--runtime-config", "--config", dest="runtime_config", type=str, default="MemoryTest/config/case_test.yaml")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--val-file", type=str, default="MemoryTest/json_data/splits/semantic_val.json")
    parser.add_argument("--test-file", type=str, default="MemoryTest/json_data/splits/semantic_test.json")
    parser.add_argument("--output-dir", type=str, default="MemoryTest/checkpoints/shine_memory_posttrain")
    parser.add_argument("--fact-counts", type=int, nargs="+", default=[1, 2, 4, 8, 12, 20])
    parser.add_argument("--qa-per-context", type=int, default=4)
    parser.add_argument("--context-format", choices=["natural", "structured", "mixed"], default="mixed")
    parser.add_argument("--recurrent-steps", type=int, default=4, help="Number of context chunks in one memory stream.")
    parser.add_argument(
        "--ordered-stream-probability",
        "--ordered-stream-prob",
        dest="ordered_stream_probability",
        type=float,
        default=0.5,
        help="When a file contains both streams and facts, probability of sampling an ordered stream.",
    )
    parser.add_argument(
        "--turn-supervision",
        choices=["final", "every"],
        default="final",
        help="Apply readout loss only after the final chunk or after every recurrent chunk.",
    )
    parser.add_argument(
        "--turn-objective",
        choices=["qa", "reconstruction", "both", "mixed"],
        default="qa",
        help="Per-turn readout objective. mixed samples QA or reconstruction independently each turn.",
    )
    parser.add_argument(
        "--qa-turn-prob",
        type=float,
        default=0.5,
        help="Probability of choosing QA when --turn-objective mixed.",
    )
    parser.add_argument(
        "--reconstruction-scope",
        choices=["current", "cumulative"],
        default="current",
        help="Whether <RECON> repeats the current chunk or all chunks observed so far.",
    )
    parser.add_argument(
        "--qa-scope",
        choices=["current", "cumulative"],
        default="cumulative",
        help="Sample QA attached only to the current turn or to all turns observed so far.",
    )
    parser.add_argument(
        "--detach-recurrent-memory-every",
        type=int,
        default=0,
        help="Detach recurrent K/V every N chunks; 0 keeps full truncated-BPTT across the stream.",
    )
    parser.add_argument(
        "--memory-position-offset",
        type=int,
        default=None,
        help="Fixed RoPE position of memory slot 0; defaults to context_max_length.",
    )
    parser.add_argument(
        "--stream-window-policy",
        choices=["contiguous", "prefix", "full"],
        default="contiguous",
        help="How to select turns when an ordered stream is longer than recurrent_steps.",
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--metalora-learning-rate", type=float, default=None)
    parser.add_argument("--mem-token-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500, help="QA evaluation interval; 0 disables evaluation.")
    parser.add_argument("--eval-trials", type=int, default=20)
    parser.add_argument("--eval-fact-counts", type=int, nargs="+", default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-max-length", type=int, default=None)
    parser.add_argument("--conversation-max-length", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bf16")
    parser.add_argument("--use-gradient-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-answer-gradient-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--empty-cache-every", type=int, default=0)
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--use-reconstruction", action="store_true")
    parser.add_argument("--recon-weight", type=float, default=0.2)
    parser.add_argument("--generated-lora-clamp", type=float, default=10.0)
    parser.add_argument("--freeze-metalora", action="store_true")
    parser.add_argument(
        "--train-mem-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also train the zero memory-token embeddings; disabled by default.",
    )
    parser.add_argument("--freeze-mem-tokens", action="store_false", dest="train_mem_tokens", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def resolve_turn_objective(args: argparse.Namespace, rng: random.Random) -> str:
    objective = args.turn_objective
    if objective == "mixed":
        objective = "qa" if rng.random() < args.qa_turn_prob else "reconstruction"
    # Backward compatibility with the old flag, which meant QA + reconstruction.
    if args.use_reconstruction and objective == "qa":
        objective = "both"
    return objective


def build_training_records(
    qa_candidates: list[dict],
    qa_rows: list[dict],
    use_contrastive: bool,
    objective: str,
    reconstruction_text: str,
) -> list[dict]:
    records = []
    include_qa = objective in {"qa", "both"}
    include_reconstruction = objective in {"reconstruction", "both"}
    if include_qa:
        for row in qa_rows:
            records.append(
                {
                    "category": "answer",
                    "prompt": question_prompt(row["question"]),
                    "answer": " " + format_answer(row["answer"]),
                }
            )
    if include_qa and use_contrastive:
        letters = list(string.ascii_uppercase)
        for qa in qa_rows:
            candidates = []
            seen = set()
            for row in [qa] + qa_candidates:
                answer = str(row["answer"])
                if answer.casefold() not in seen:
                    candidates.append(answer)
                    seen.add(answer.casefold())
            candidates = candidates[: min(len(candidates), len(letters))]
            gold_idx = candidates.index(str(qa["answer"]))
            option_lines = [f"{letters[idx]}. {answer}" for idx, answer in enumerate(candidates)]
            prompt = (
                "Choose the option that answers the question using the current memory.\n\n"
                f"Question: {qa['question']}\n"
                + "\n".join(option_lines)
                + "\nAnswer with the option letter only."
            )
            records.append(
                {
                    "category": "contrastive",
                    "prompt": prompt,
                    "answer": " " + letters[gold_idx],
                }
            )
    if include_reconstruction:
        records.append(
            {
                "category": "reconstruction",
                "prompt": reconstruction_prompt(),
                # Official SHINE pretraining reconstructs evidence text after <RECON>.
                "answer": "\n" + reconstruction_text,
            }
        )
    return records


def default_train_file() -> str:
    augmented = resolve_path("MemoryTest/json_data/splits/semantic_train_augmented.json")
    if augmented.exists():
        return str(augmented)
    return "MemoryTest/json_data/splits/semantic_train.json"


def load_shine_for_training(args: argparse.Namespace):
    from MemoryTest.case_test import build_cfg, load_runtime, resolve_device, resolve_torch_dtype

    runtime_args = load_runtime_args(args.runtime_config)
    checkpoint_dir = args.checkpoint_dir
    latest_dir = resolve_path(args.output_dir) / "latest"
    if args.resume and latest_dir.exists():
        checkpoint_dir = str(latest_dir)
    runtime_args.checkpoint_dir = checkpoint_dir
    if args.device is not None:
        runtime_args.device = args.device
    if args.gpu_id is not None:
        runtime_args.gpu_id = args.gpu_id
    if args.context_max_length is not None:
        runtime_args.context_max_length = args.context_max_length
    if args.conversation_max_length is not None:
        runtime_args.conversation_max_length = args.conversation_max_length
    device = resolve_device(runtime_args.device, runtime_args.gpu_id)
    cfg = build_cfg(runtime_args)
    cfg.model.torch_dtype = args.torch_dtype
    metanetwork, metalora, tokenizer = load_runtime(cfg, runtime_args.checkpoint_dir, device)
    dtype = resolve_torch_dtype(args.torch_dtype)
    if isinstance(dtype, torch.dtype):
        metanetwork.to(device=device, dtype=dtype)
        metalora = cast_floating_tensors(metalora, dtype)
    if hasattr(metanetwork.metamodel, "config"):
        metanetwork.metamodel.config.use_cache = False
    if hasattr(metanetwork.metamodel, "gradient_checkpointing_enable") and args.use_answer_gradient_checkpoint:
        metanetwork.metamodel.gradient_checkpointing_enable()
    metanetwork.train()
    trainable = set_posttrain_requires_grad(
        metanetwork,
        metalora,
        train_metalora=not args.freeze_metalora,
        train_mem_tokens=args.train_mem_tokens,
    )
    return runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable


def build_optimizer(trainable: dict[str, list[torch.Tensor]], args: argparse.Namespace):
    param_groups = []
    if trainable["metanetwork"]:
        param_groups.append(
            {
                "params": trainable["metanetwork"],
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "metanetwork",
            }
        )
    if trainable["mem_tokens"]:
        param_groups.append(
            {
                "params": trainable["mem_tokens"],
                "lr": args.mem_token_learning_rate if args.mem_token_learning_rate is not None else args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "mem_tokens",
            }
        )
    if trainable["metalora"]:
        param_groups.append(
            {
                "params": trainable["metalora"],
                "lr": args.metalora_learning_rate if args.metalora_learning_rate is not None else args.learning_rate,
                "weight_decay": args.weight_decay,
                "name": "metalora",
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters selected.")
    LOGGER.info(
        "Optimizer groups: %s",
        ", ".join(f"{group['name']} n={sum(param.numel() for param in group['params'])} lr={group['lr']}" for group in param_groups),
    )
    return torch.optim.AdamW(param_groups)


def encode_choice_batch(tokenizer, context_rows: list[dict], qa_rows: list[dict], max_length: int, device):
    import string

    letters = list(string.ascii_uppercase)
    input_rows = []
    label_rows = []
    for qa in qa_rows:
        candidates = []
        seen = set()
        for row in [qa] + context_rows:
            answer = str(row["answer"])
            if answer.casefold() not in seen:
                candidates.append(answer)
                seen.add(answer.casefold())
        candidates = candidates[: min(len(candidates), len(letters))]
        gold_idx = candidates.index(str(qa["answer"]))
        option_lines = [f"{letters[idx]}. {answer}" for idx, answer in enumerate(candidates)]
        prompt = (
            "Choose the option that answers the question using the current memory.\n\n"
            f"Question: {qa['question']}\n"
            + "\n".join(option_lines)
            + "\nAnswer with the option letter only."
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        answer_ids = tokenizer(" " + letters[gold_idx] + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[-max_length:]
        labels = ([-100] * len(prompt_ids) + answer_ids)[-max_length:]
        input_rows.append(input_ids)
        label_rows.append(labels)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row) for row in input_rows)
    return {
        "input_ids": torch.tensor([[pad_id] * (max_len - len(row)) + row for row in input_rows], dtype=torch.long, device=device),
        "labels": torch.tensor([[-100] * (max_len - len(row)) + labels for row, labels in zip(input_rows, label_rows)], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[0] * (max_len - len(row)) + [1] * len(row) for row in input_rows], dtype=torch.long, device=device),
    }


def compute_contrastive_loss(context_rows, qa_rows, lora_dict, metanetwork, tokenizer, device, max_length, use_gradient_checkpoint: bool = False):
    batch = encode_choice_batch(tokenizer, context_rows, qa_rows, max_length=max_length, device=device)
    outputs = metanetwork.metamodel(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        loradict=lora_dict,
        ignore_mem_token=True,
        use_gradient_checkpoint=use_gradient_checkpoint,
    )
    return outputs.loss


def generate_answer(metanetwork, tokenizer, lora_dict, question: str, device, max_new_tokens: int, max_length: int):
    from MemoryTest.case_test import extract_think_and_answer
    from MemoryTest.prepare_data.prompt_templates import question_prompt

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_prompt(question)}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    raw = tokenizer.decode(outputs[0, input_ids.shape[1] :], skip_special_tokens=True)
    _, answer = extract_think_and_answer(raw)
    return answer, raw


def evaluate_current(metanetwork, metalora, tokenizer, cfg, data, args, runtime_args, device, rng) -> dict:
    metanetwork.eval()
    eval_rows = []
    eval_fact_counts = args.eval_fact_counts if args.eval_fact_counts is not None else args.fact_counts
    progress = tqdm(
        range(args.eval_trials),
        desc="posttrain val",
        dynamic_ncols=True,
        leave=False,
    ) if tqdm is not None else range(args.eval_trials)
    memory_position_offset = args.memory_position_offset or cfg.test.context_max_length
    with torch.no_grad():
        for trial in progress:
            stream = sample_training_stream(
                data,
                recurrent_steps=args.recurrent_steps,
                fact_counts=eval_fact_counts,
                context_format=args.context_format,
                ordered_stream_probability=args.ordered_stream_probability,
                window_policy=args.stream_window_policy,
                rng=rng,
            )
            recurrent_memory = None
            for turn_index, turn in enumerate(stream.turns):
                if turn_index + 1 == len(stream.turns):
                    lora_dict, recurrent_memory = trainable_generate_context_lora(
                        turn.text,
                        metanetwork,
                        tokenizer,
                        metalora,
                        cfg,
                        device,
                        recurrent_memory=recurrent_memory,
                        return_recurrent_state=True,
                        memory_position_offset=memory_position_offset,
                    )
                else:
                    recurrent_memory = trainable_update_recurrent_memory(
                        turn.text,
                        metanetwork,
                        tokenizer,
                        metalora,
                        cfg,
                        device,
                        recurrent_memory=recurrent_memory,
                        memory_position_offset=memory_position_offset,
                    )
            lora_dict = clamp_lora_tensors(lora_dict, args.generated_lora_clamp)
            qa_source_turns = stream.turns[-1:] if args.qa_scope == "current" else stream.turns
            qa_rows = sample_turn_qa(qa_source_turns, args.qa_per_context, rng)
            for fact in qa_rows:
                answer, raw = generate_answer(
                    metanetwork,
                    tokenizer,
                    lora_dict,
                    fact["question"],
                    device,
                    args.max_new_tokens or runtime_args.max_new_tokens,
                    runtime_args.conversation_max_length,
                )
                eval_rows.append(make_eval_row(len(eval_rows) + 1, fact, answer, raw=raw))
            if tqdm is not None:
                summary = summarize_examples(eval_rows)
                progress.set_postfix({"acc": f"{summary['accuracy']:.4f}", "n": summary["total"]})
    metanetwork.train()
    return summarize_examples(eval_rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not 0.0 <= args.qa_turn_prob <= 1.0:
        raise ValueError("--qa-turn-prob must be between 0 and 1")
    if not 0.0 <= args.ordered_stream_probability <= 1.0:
        raise ValueError("--ordered-stream-probability must be between 0 and 1")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if not args.train_file:
        args.train_file = default_train_file()
    train_data = load_recurrent_dataset(resolve_path(args.train_file))
    val_data = load_recurrent_dataset(resolve_path(args.val_file)) if args.eval_every > 0 else None
    output_dir = resolve_path(args.output_dir)
    log_path = output_dir / "shine_posttrain_train_log.jsonl"
    rng = random.Random(args.seed)

    runtime_args, cfg, device, metanetwork, metalora, tokenizer, trainable = load_shine_for_training(args)
    memory_position_offset = args.memory_position_offset or cfg.test.context_max_length
    if memory_position_offset < cfg.test.context_max_length:
        raise ValueError(
            f"memory_position_offset ({memory_position_offset}) must be >= padded context length "
            f"({cfg.test.context_max_length})"
        )
    LOGGER.info(
        "Recurrent SHINE: steps=%s fixed_memory_position_offset=%s detach_every=%s",
        args.recurrent_steps,
        memory_position_offset,
        args.detach_recurrent_memory_every,
    )
    LOGGER.info("Training hypernetwork=%s Metalora=%s", bool(trainable["metanetwork"]), bool(trainable["metalora"]))
    optimizer = build_optimizer(trainable, args)
    best_val_acc = -1.0 if args.eval_every > 0 else None

    train_progress = tqdm(
        range(1, args.max_steps + 1),
        desc="SHINE posttrain",
        dynamic_ncols=True,
        leave=True,
    ) if tqdm is not None else range(1, args.max_steps + 1)

    for step in train_progress:
        stream = sample_training_stream(
            train_data,
            recurrent_steps=args.recurrent_steps,
            fact_counts=args.fact_counts,
            context_format=args.context_format,
            ordered_stream_probability=args.ordered_stream_probability,
            window_policy=args.stream_window_policy,
            rng=rng,
        )
        recurrent_memory = None
        observed_turns = []
        supervised_turns = []
        memory_norms_by_turn = []
        for turn_index, current_turn in enumerate(stream.turns):
            observed_turns.append(current_turn)
            is_final_turn = turn_index + 1 == len(stream.turns)
            supervise_turn = args.turn_supervision == "every" or is_final_turn
            if supervise_turn:
                lora_dict, recurrent_memory = trainable_generate_context_lora(
                    current_turn.text,
                    metanetwork,
                    tokenizer,
                    metalora,
                    cfg,
                    device,
                    use_gradient_checkpoint=args.use_gradient_checkpoint,
                    recurrent_memory=recurrent_memory,
                    return_recurrent_state=True,
                    memory_position_offset=memory_position_offset,
                )
                lora_dict = clamp_lora_tensors(lora_dict, args.generated_lora_clamp)
                objective = resolve_turn_objective(args, rng)
                qa_source_turns = [current_turn] if args.qa_scope == "current" else observed_turns
                qa_rows = sample_turn_qa(
                    qa_source_turns,
                    args.qa_per_context,
                    rng,
                )
                if objective in {"qa", "both"} and not qa_rows:
                    if args.turn_objective == "mixed":
                        objective = "reconstruction"
                    else:
                        raise ValueError(
                            f"Stream {stream.stream_id!r} turn {current_turn.turn_id!r} selected "
                            f"{objective} supervision but no QA targets are available."
                        )
                reconstruction_text = (
                    current_turn.text
                    if args.reconstruction_scope == "current"
                    else "\n".join(turn.text for turn in observed_turns)
                )
                training_records = build_training_records(
                    accumulated_qa(qa_source_turns),
                    qa_rows,
                    args.use_contrastive,
                    objective,
                    reconstruction_text=reconstruction_text,
                )
                category_losses = compute_combined_lora_loss(
                    training_records,
                    lora_dict,
                    metanetwork,
                    tokenizer,
                    device,
                    args.answer_max_length,
                    use_gradient_checkpoint=args.use_answer_gradient_checkpoint,
                )
                first_category_loss = next(iter(category_losses.values()))
                turn_loss = first_category_loss.new_zeros(())
                if "answer" in category_losses:
                    turn_loss = turn_loss + category_losses["answer"]
                if "contrastive" in category_losses:
                    turn_loss = turn_loss + args.contrastive_weight * category_losses["contrastive"]
                if "reconstruction" in category_losses:
                    reconstruction_weight = 1.0 if objective == "reconstruction" else args.recon_weight
                    turn_loss = turn_loss + reconstruction_weight * category_losses["reconstruction"]
                supervised_turns.append(
                    {
                        "turn": turn_index + 1,
                        "turn_id": current_turn.turn_id,
                        "objective": objective,
                        "qa_rows": qa_rows if objective in {"qa", "both"} else [],
                        "category_losses": category_losses,
                        "loss": turn_loss,
                    }
                )
            else:
                recurrent_memory = trainable_update_recurrent_memory(
                    current_turn.text,
                    metanetwork,
                    tokenizer,
                    metalora,
                    cfg,
                    device,
                    use_gradient_checkpoint=args.use_gradient_checkpoint,
                    recurrent_memory=recurrent_memory,
                    memory_position_offset=memory_position_offset,
                )
            memory_norms_by_turn.append(
                {
                    "turn": turn_index + 1,
                    "turn_id": current_turn.turn_id,
                    "norms": recurrent_memory_norm_stats(recurrent_memory),
                }
            )
            detach_every = args.detach_recurrent_memory_every
            if detach_every > 0 and (turn_index + 1) % detach_every == 0 and turn_index + 1 < len(stream.turns):
                recurrent_memory = recurrent_memory.detach()

        loss = torch.stack([turn["loss"] for turn in supervised_turns]).mean()

        def mean_category_loss(category: str):
            values = [turn["category_losses"][category] for turn in supervised_turns if category in turn["category_losses"]]
            return torch.stack(values).mean() if values else None

        answer_loss = mean_category_loss("answer")
        contrastive_loss = mean_category_loss("contrastive")
        recon_loss = mean_category_loss("reconstruction")
        lora_stats = lora_tensor_stats(lora_dict)
        memory_norms = memory_norms_by_turn[-1]["norms"]
        if lora_stats["finite"] < 1.0:
            raise FloatingPointError(f"Generated LoRA is non-finite at step {step}.")

        turn_loss_log = [
            {
                "turn": turn["turn"],
                "turn_id": turn["turn_id"],
                "objective": turn["objective"],
                "loss": float(turn["loss"].detach().cpu()),
                "answer_loss": float(turn["category_losses"]["answer"].detach().cpu()) if "answer" in turn["category_losses"] else None,
                "contrastive_loss": float(turn["category_losses"]["contrastive"].detach().cpu()) if "contrastive" in turn["category_losses"] else None,
                "reconstruction_loss": float(turn["category_losses"]["reconstruction"].detach().cpu()) if "reconstruction" in turn["category_losses"] else None,
                "qa_fact_ids": [row["id"] for row in turn["qa_rows"]],
            }
            for turn in supervised_turns
        ]

        if not torch.isfinite(loss):
            bad_record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()) if answer_loss is not None else None,
                "contrastive_loss": float(contrastive_loss.detach().cpu()) if contrastive_loss is not None else None,
                "reconstruction_loss": float(recon_loss.detach().cpu()) if recon_loss is not None else None,
                "stream_id": stream.stream_id,
                "source_kind": stream.source_kind,
                "turn_ids": [turn.turn_id for turn in stream.turns],
                "fact_ids_by_turn": [list(turn.fact_ids) for turn in stream.turns],
                "turn_losses": turn_loss_log,
                "nonfinite": True,
                "lora_stats": lora_stats,
                "memory_norms": memory_norms,
                "memory_norms_by_turn": memory_norms_by_turn,
            }
            append_jsonl(log_path, bad_record)
            raise FloatingPointError(
                f"Non-finite loss at step {step}. "
                "Try lowering --learning-rate, reducing --qa-per-context, or disabling/reweighting auxiliary losses."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            trainable_params = [param for group in trainable.values() for param in group]
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()

        log_record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "answer_loss": float(answer_loss.detach().cpu()) if answer_loss is not None else None,
            "contrastive_loss": float(contrastive_loss.detach().cpu()) if contrastive_loss is not None else None,
            "reconstruction_loss": float(recon_loss.detach().cpu()) if recon_loss is not None else None,
            "stream_id": stream.stream_id,
            "source_kind": stream.source_kind,
            "turn_ids": [turn.turn_id for turn in stream.turns],
            "fact_ids_by_turn": [list(turn.fact_ids) for turn in stream.turns],
            "turn_losses": turn_loss_log,
            "lora_stats": lora_stats,
            "memory_norms": memory_norms,
            "memory_norms_by_turn": memory_norms_by_turn,
        }
        if tqdm is not None:
            postfix = {
                "loss": f"{log_record['loss']:.4f}",
                "lora_max": f"{lora_stats['max_abs']:.2f}",
                "mem_v": f"{memory_norms.get('value_rms_mean', 0.0):.3f}",
            }
            if answer_loss is not None:
                postfix["answer"] = f"{log_record['answer_loss']:.4f}"
            if contrastive_loss is not None:
                postfix["choice"] = f"{log_record['contrastive_loss']:.4f}"
            if recon_loss is not None:
                postfix["recon"] = f"{log_record['reconstruction_loss']:.4f}"
            train_progress.set_postfix(postfix)
        if step % 10 == 0 or step == 1:
            append_jsonl(log_path, log_record)
            LOGGER.info(
                "step=%s loss=%.6f answer=%s reconstruction=%s",
                step,
                log_record["loss"],
                f"{log_record['answer_loss']:.6f}" if log_record["answer_loss"] is not None else "n/a",
                f"{log_record['reconstruction_loss']:.6f}" if log_record["reconstruction_loss"] is not None else "n/a",
            )

        eval_due = args.eval_every > 0 and (step % args.eval_every == 0 or step == args.max_steps)
        if eval_due:
            val_summary = evaluate_current(metanetwork, metalora, tokenizer, cfg, val_data, args, runtime_args, device, rng)
            log_record["val"] = val_summary
            append_jsonl(log_path, log_record)
            LOGGER.info("val step=%s accuracy=%.4f", step, val_summary["accuracy"])
            if tqdm is not None:
                train_progress.set_postfix({"loss": f"{log_record['loss']:.4f}", "val_acc": f"{val_summary['accuracy']:.4f}", "best": f"{max(best_val_acc, val_summary['accuracy']):.4f}"})
            save_posttrain_checkpoint(output_dir / "latest", metanetwork, metalora, extra_state={"step": step, "val": val_summary, "config": vars(args)})
            if val_summary["accuracy"] > best_val_acc:
                best_val_acc = val_summary["accuracy"]
                save_posttrain_checkpoint(output_dir / "best", metanetwork, metalora, extra_state={"step": step, "val": val_summary, "config": vars(args)})
        elif step == args.max_steps or (args.save_every > 0 and step % args.save_every == 0):
            save_posttrain_checkpoint(output_dir / "latest", metanetwork, metalora, extra_state={"step": step, "config": vars(args)})
        if args.empty_cache_every > 0 and step % args.empty_cache_every == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_summary = {"best_val_accuracy": best_val_acc, "max_steps": args.max_steps, "config": vars(args)}
    (output_dir / "summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del metanetwork, metalora, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
