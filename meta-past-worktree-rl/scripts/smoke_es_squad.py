"""2-step ES smoke test: verifies ESTrainer.step() + anchor + logging + save.

Not meant for real training — uses N=4 (2 antithetic pairs), B=2, Q=2 so one
step stays under a minute on a single A800. Its job is to detect breakage
between the adapter, rollout, trainer, and persistence layers before
launching a real Phase 1 run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_past.data.squad_contexts import iter_train_val
from meta_past.es.trainer import ESConfig, ESTrainer
from meta_past.reward.f1_reward import f1_reward
from meta_past.rollout.squad_rollout import SquadRollout, SquadRolloutConfig
from meta_past.shine_adapter import ShineHypernet


def main() -> None:
    home = Path.home()
    net = ShineHypernet(
        ckpt_dir=str(home / "huggingfacemodels" / "SHINE-ift_mqa_1qa"),
        device="cuda:0",
        backbone=str(home / "huggingfacemodels" / "Qwen3-8B"),
        lora_r=8,
        metalora_r=128,
    )
    net.assert_only_hypernet_trainable()

    rollout = SquadRollout(
        hypernet=net,
        reward_fn=f1_reward,
        cfg=SquadRolloutConfig(
            context_max_length=512,
            question_max_length=128,
            max_new_tokens=32,
            questions_per_context=2,
        ),
    )

    train_ctx, heldout_ctx = iter_train_val(train_size=8, val_size=4)

    es_cfg = ESConfig(
        N=4,
        sigma=0.005,
        lr=0.005,
        reward_norm="zscore",
        batch_size=2,
        anchor_coef_start=1.0,
        anchor_coef_end=0.1,
        anchor_decay_steps=10,
        total_steps=2,
        eval_every=1,
        heldout_contexts=4,
        save_every=2,
        seed=0,
        out_dir="runs/smoke_es_squad",
    )

    trainer = ESTrainer(
        hypernet=net,
        rollout=rollout,
        train_contexts=train_ctx,
        heldout_contexts=heldout_ctx,
        cfg=es_cfg,
    )
    trainer.fit()
    print("[smoke] ESTrainer.fit() completed without error.")


if __name__ == "__main__":
    main()
