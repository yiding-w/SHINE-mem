"""Entry point for Phase 1 ES on SQuAD single-passage.

Reads meta_past/config/es_squad_phase1.yaml via OmegaConf, builds hypernet +
rollout + trainer, runs cfg.train.total_steps. Logs to <out_dir>/train_log.jsonl.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.data.squad_contexts import iter_train_val
from meta_past.es.trainer import ESConfig, ESTrainer
from meta_past.reward.f1_reward import f1_reward
from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
from meta_past.shine_adapter import ShineHypernet


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1]
            / "meta_past" / "config" / "es_squad_phase1.yaml"
        ),
    )
    return p.parse_args()


def _noise_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _build_reward(cfg):
    kind = cfg.reward.type
    if kind == "f1":
        return f1_reward
    raise NotImplementedError(f"Reward type {kind!r} not yet implemented.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    cfg = OmegaConf.load(args.config)

    net = ShineHypernet(
        ckpt_dir=cfg.hypernet.ckpt_dir,
        device=cfg.hypernet.device,
        backbone=cfg.hypernet.backbone,
        lora_r=int(cfg.hypernet.lora_r),
        metalora_r=int(cfg.hypernet.metalora_r),
    )
    net.assert_only_hypernet_trainable()

    rollout = SquadRollout(
        hypernet=net,
        reward_fn=_build_reward(cfg),
        cfg=SquadRolloutConfig(
            context_max_length=int(cfg.rollout.context_max_length),
            question_max_length=int(cfg.rollout.question_max_length),
            max_new_tokens=int(cfg.rollout.max_new_tokens),
            questions_per_context=int(cfg.rollout.questions_per_context),
        ),
    )

    train_ctx, heldout_ctx = iter_train_val(
        train_size=int(cfg.train.train_contexts),
        val_size=int(cfg.train.heldout_contexts),
    )

    es_cfg = ESConfig(
        N=int(cfg.es.N),
        sigma=float(cfg.perturb.sigma),
        lr=float(cfg.es.lr),
        reward_norm=str(cfg.es.reward_norm),
        mode=str(cfg.es.get("mode", "one_sided")),
        noise_dtype=_noise_dtype(str(cfg.perturb.noise_dtype)),
        sigma_mode=str(cfg.perturb.get("sigma_mode", "absolute")),
        rms_floor=float(cfg.perturb.get("rms_floor", 1e-3)),
        include_metalora=bool(cfg.perturb.get("include_metalora", True)),
        include_mem_tokens=bool(cfg.perturb.get("include_mem_tokens", False)),
        min_rms=float(cfg.perturb.get("min_rms", 0.0)),
        exclude_bias=bool(cfg.perturb.get("exclude_bias", True)),
        batch_size=int(cfg.rollout.batch_size),
        anchor_coef_start=float(cfg.anchor.coef_start),
        anchor_coef_end=float(cfg.anchor.coef_end),
        anchor_decay_steps=int(cfg.anchor.decay_steps),
        total_steps=int(cfg.train.total_steps),
        eval_every=int(cfg.train.eval_every),
        heldout_contexts=int(cfg.train.heldout_contexts),
        save_every=int(cfg.train.save_every),
        seed=int(cfg.train.seed),
        out_dir=str(cfg.train.out_dir),
        heldout_regression_threshold=float(cfg.train.heldout_regression_threshold),
        heldout_rollback_window=int(cfg.train.heldout_rollback_window),
    )

    trainer = ESTrainer(
        hypernet=net,
        rollout=rollout,
        train_contexts=train_ctx,
        heldout_contexts=heldout_ctx,
        cfg=es_cfg,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
