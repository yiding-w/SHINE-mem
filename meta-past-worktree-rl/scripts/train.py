"""End-to-end launcher for batched-context RL training (HybridEngine).

Run via torchrun:

    torchrun --nproc_per_node=8 --standalone scripts/train.py \\
        --config meta_past/config/rl_musique_grpo.yaml

Each torchrun process becomes one training rank pinned to one GPU. On that
GPU the rank co-locates:
  - SHINE hypernet (trainable + frozen Qwen3-8B backbone copy)
  - vLLM ``LLM(enable_sleep_mode=True)`` for rollout sampling

vLLM weights are awake during the rollout window and asleep during
rescore + backward, freeing the GPU for training. Gradients are
all-reduced manually (SUM, since per-rank loss is normalized by the
global token count) before each ``optimizer.step()``.

Single-process fallback: if launched without torchrun, runs as a 1-rank
"DDP" world (no all-reduce, no NCCL init). Useful for smoke tests.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _bootstrap_cuda_env(yaml_path: str) -> "OmegaConf":
    """Pin this rank to its local GPU before importing torch / vllm.

    Reads ``LOCAL_RANK`` from torchrun env. With WORLD_SIZE>1, each rank's
    ``CUDA_VISIBLE_DEVICES`` is the single GPU at index ``LOCAL_RANK``;
    inside the rank, that GPU is then ``cuda:0``.

    Also forces vLLM into single-process mode and enables expandable
    PyTorch CUDA segments (defragments under tight HBM).
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(yaml_path)
    OmegaConf.resolve(cfg)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Honour torchrun's GPU pinning via LOCAL_RANK; if user pre-set
    # CUDA_VISIBLE_DEVICES (e.g. to skip GPU 0), respect it and rely on
    # set_device(local_rank) to pick within the visible set.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # vLLM's per-step "It took N seconds to wake up / fall asleep",
    # "Successfully reset prefix cache", "Sleep mode freed N GiB" land at
    # INFO level and flood the log under DDP (× 8 ranks × every step).
    # vLLM reads VLLM_LOGGING_LEVEL at import time; setting it to WARNING
    # silences all those. Boot messages (model load, KV cache size, etc.)
    # are also INFO and lost — but they only fire once during boot, which
    # is a fair trade.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # NB: do *not* set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
    # — vLLM's ``enable_sleep_mode=True`` uses ``CuMemAllocator`` which is
    # incompatible with expandable segments (asserts at init).
    return cfg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "meta_past" / "config" / "rl_musique_grpo.yaml"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _bootstrap_cuda_env(args.config)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format=f"%(asctime)s rank{rank} %(levelname)s %(name)s: %(message)s",
    )

    # vLLM's tokenizer subsystem warns "No tokenizer found in in-memory"
    # for every LoRA push (our dummy ``lora_path="in-memory"`` triggers a
    # fallback to the base tokenizer, which is exactly what we want). The
    # warning is at WARNING level and survives ``VLLM_LOGGING_LEVEL=WARNING``,
    # so attach a content filter to drop just this message.
    class _DropInMemoryTokenizerWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:
                return True
            return "No tokenizer found in in-memory" not in msg \
                and "is not a local folder and is not a valid model identifier" not in msg
    _drop_filter = _DropInMemoryTokenizerWarning()
    logging.getLogger().addFilter(_drop_filter)
    for name in ("vllm", "vllm.transformers_utils.tokenizer"):
        logging.getLogger(name).addFilter(_drop_filter)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    import torch.distributed as dist

    log = logging.getLogger("meta_past.scripts.train")

    # Pin the device handle (CUDA_VISIBLE_DEVICES already filtered to one GPU).
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Init NCCL for cross-rank gradient all-reduce. Only when running under
    # torchrun (world_size > 1).
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        log.info("torch.distributed initialized: rank=%d/%d", rank, world_size)

    from meta_past.reward.f1_reward import f1_reward
    from meta_past.rl.rollout import RLRollout, RLRolloutConfig
    from meta_past.rl.trainer import RLConfig, RLTrainer
    from meta_past.rl.vllm_engine import VLLMEngine, VLLMEngineConfig
    from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
    from meta_past.shine_adapter import ShineHypernet

    train_device = "cuda:0"

    def _build_reward(cfg_reward):
        kind = str(cfg_reward.get("type", "f1") if cfg_reward is not None else "f1").lower()
        if kind == "f1":
            return f1_reward
        if kind == "judge":
            from meta_past.reward.judge_reward import HttpJudgeReward
            return HttpJudgeReward(
                base_url=str(cfg_reward.get("judge_url", "http://127.0.0.1:8124")),
                timeout_s=float(cfg_reward.get("judge_timeout_s", 30.0)),
                max_retries=int(cfg_reward.get("judge_max_retries", 3)),
                concurrency=int(cfg_reward.get("judge_concurrency", 32)),
            )
        raise NotImplementedError(f"Reward type {kind!r} not yet wired for RL.")

    def _load_data(cfg, train_size: int, val_size: int):
        """Dispatch on ``cfg.data.name`` (default: squad)."""
        data_section = cfg.get("data", None)
        name = (data_section.get("name", "squad") if data_section else "squad").lower()
        if name == "squad":
            from meta_past.data.squad_contexts import iter_train_val
            return iter_train_val(train_size=train_size, val_size=val_size)
        if name == "musique":
            from meta_past.data.musique_contexts import iter_train_val
            cache_dir = (
                str(data_section.get("cache_dir", "")) or None
            ) if data_section else None
            return iter_train_val(
                train_size=train_size, val_size=val_size, cache_dir=cache_dir,
            )
        if name == "bbh":
            from meta_past.data.bbh_contexts import iter_train_val
            cache_dir = (
                str(data_section.get("cache_dir", "")) or None
            ) if data_section else None
            kwargs = {"train_size": train_size, "val_size": val_size,
                      "cache_dir": cache_dir}
            for k in ("K_max", "ctx_token_budget", "train_frac", "seed",
                      "tokenizer_path"):
                if data_section is not None and k in data_section:
                    kwargs[k] = data_section[k]
            return iter_train_val(**kwargs)
        raise NotImplementedError(f"Unknown data.name: {name!r}")

    # --- vLLM engine (boot first; constructs awake then immediately sleeps) ---
    engine_cfg = VLLMEngineConfig(
        model_path=str(cfg.hypernet.backbone),
        max_loras=int(cfg.vllm.max_loras),
        max_lora_rank=int(cfg.vllm.max_lora_rank),
        max_model_len=int(cfg.vllm.max_model_len),
        dtype=str(cfg.vllm.dtype),
        gpu_memory_utilization=float(cfg.vllm.get("gpu_memory_utilization", 0.55)),
        enforce_eager=bool(cfg.vllm.get("enforce_eager", False)),
        seed=int(cfg.train.seed) + rank,
    )
    log.info("booting vLLM engine on rank %d", rank)
    engine = VLLMEngine(engine_cfg)
    engine.boot()
    log.info("vLLM engine ready (asleep) on rank %d", rank)

    # --- SHINE hypernet ---
    log.info("constructing ShineHypernet on %s", train_device)
    net = ShineHypernet(
        ckpt_dir=cfg.hypernet.ckpt_dir,
        device=train_device,
        backbone=cfg.hypernet.backbone,
        lora_r=int(cfg.hypernet.lora_r),
        metalora_r=int(cfg.hypernet.metalora_r),
    )
    net.assert_only_hypernet_trainable()

    reward_fn = _build_reward(cfg.get("reward", None))

    # ``enable_thinking``: yaml may have it as null/true/false. Resolve to
    # Python None/True/False so the rollout config dispatches correctly. With
    # the stock Qwen3 template restored: True/None = no stub (model decides);
    # False = inject empty <think></think> stub forcing direct answer.
    _et_raw = cfg.rollout.get("enable_thinking", None)
    enable_thinking: bool | None = None if _et_raw is None else bool(_et_raw)
    force_think: bool = bool(cfg.rollout.get("force_think", False))

    rollout_cfg = RLRolloutConfig(
        context_max_length=int(cfg.rollout.context_max_length),
        question_max_length=int(cfg.rollout.question_max_length),
        max_new_tokens=int(cfg.rollout.max_new_tokens),
        temperature=float(cfg.rollout.temperature),
        contexts_per_step=int(cfg.rollout.contexts_per_step),
        questions_per_context=int(cfg.rollout.questions_per_context),
        rollouts_per_question=int(cfg.rollout.rollouts_per_question),
        use_gradient_checkpoint=bool(cfg.optim.use_gradient_checkpointing),
        hypernet_microbatch_contexts=int(
            cfg.rollout.get("hypernet_microbatch_contexts", 0)
        ),
        rescore_microbatch_contexts=int(
            cfg.rollout.get("rescore_microbatch_contexts", 0)
        ),
        enable_thinking=enable_thinking,
        force_think=force_think,
    )

    eval_rollout = SquadRollout(
        hypernet=net,
        reward_fn=reward_fn,
        cfg=SquadRolloutConfig(
            context_max_length=rollout_cfg.context_max_length,
            question_max_length=rollout_cfg.question_max_length,
            max_new_tokens=rollout_cfg.max_new_tokens,
            questions_per_context=rollout_cfg.questions_per_context,
            enable_thinking=enable_thinking,
            force_think=force_think,
        ),
    )

    train_ctx, heldout_ctx = _load_data(
        cfg,
        train_size=int(cfg.train.train_contexts),
        val_size=int(cfg.train.heldout_contexts),
    )
    Q = rollout_cfg.questions_per_context
    train_ctx = [c for c in train_ctx if len(c.qa) >= Q]
    if len(train_ctx) < rollout_cfg.contexts_per_step:
        raise RuntimeError(
            f"After filtering for >= {Q} questions/context, only "
            f"{len(train_ctx)} train contexts remain — fewer than B="
            f"{rollout_cfg.contexts_per_step}."
        )
    log.info("train pool: %d, heldout: %d", len(train_ctx), len(heldout_ctx))

    wandb_section = cfg.get("wandb", None)
    rl_cfg = RLConfig(
        adv_kind=str(cfg.rl.adv_kind),
        norm_adv_by_std=bool(cfg.rl.norm_adv_by_std),
        loss_agg_mode=str(cfg.rl.loss_agg_mode),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        grad_clip=float(cfg.optim.grad_clip),
        total_steps=int(cfg.train.total_steps),
        eval_every=int(cfg.train.eval_every),
        heldout_contexts=int(cfg.train.heldout_contexts),
        save_every=int(cfg.train.save_every),
        seed=int(cfg.train.seed),
        out_dir=str(cfg.train.out_dir),
        wandb_enabled=bool(wandb_section.get("enabled", False)) if wandb_section else False,
        wandb_project=str(wandb_section.get("project", "meta-past-rl")) if wandb_section else "meta-past-rl",
        wandb_name=str(wandb_section.get("name", "") or "") if wandb_section else "",
        wandb_tags=list(wandb_section.get("tags", []) or []) if wandb_section else [],
        wandb_notes=str(wandb_section.get("notes", "") or "") if wandb_section else "",
        wandb_mode=str(wandb_section.get("mode", "online")) if wandb_section else "online",
        wandb_group=str(wandb_section.get("group", "") or "") if wandb_section else "",
    )

    rl_rollout = RLRollout(
        hypernet=net,
        reward_fn=reward_fn,
        engine=engine,
        cfg=rollout_cfg,
    )
    trainer = RLTrainer(
        hypernet=net,
        rollout=rl_rollout,
        eval_rollout=eval_rollout,
        train_contexts=train_ctx,
        heldout_contexts=heldout_ctx,
        cfg=rl_cfg,
    )

    try:
        trainer.fit()
    finally:
        engine.shutdown()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
