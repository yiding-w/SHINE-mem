"""ES training loop, ported from ES at Scale
(`es_fine-tuning_conciseness.py`, see their main loop around L205-L319).

Per-step flow:
  1. Sample B contexts (fresh each step).
  2. Draw N seeds.
  3. For each seed s:
       phi += sigma * eps(s)
       R_s = mean reward over contexts
       phi -= sigma * eps(s)       # restore (same seed + opposite sign)
  4. Normalize rewards across the population (zscore / rank / none).
  5. update_k = (1/N) Σ r_norm_i · ε_i,k
     phi_k += lr · update_k
  6. (Optional) anchor: phi -= lr · λ(t) · (phi - phi_pretrained).
  7. Eval on held-out contexts every cfg.eval_every steps; save every
     cfg.save_every steps.

Antithetic pairing (N/2 ± pairs, `mode="antithetic"`) is supported but OFF by
default — ES at Scale's own algorithm does not use it and our smoke run #5
showed antithetic adds compute without a meaningful training-quality gain at
our scale.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..anchor.frobenius import AnchorSchedule, FrobeniusAnchor
from ..data.squad_contexts import SquadContext
from ..rollout.squad_rollout import SquadRollout
from ..shine_adapter import ShineHypernet
from .perturb import InPlacePerturber
from .update import antithetic_es_grad, one_sided_es_grad


logger = logging.getLogger("meta_past.es.trainer")


@dataclass
class ESConfig:
    # ES core
    N: int = 30                       # population size (full population, NOT antithetic pairs)
    sigma: float = 0.005
    lr: float = 0.005
    reward_norm: str = "zscore"       # "zscore" | "rank" | "none"
    mode: str = "one_sided"           # "one_sided" (ES at Scale) | "antithetic"
    noise_dtype: torch.dtype = torch.float32
    sigma_mode: str = "absolute"      # "absolute" | "rms_relative"
    rms_floor: float = 1e-3

    # Perturbation scope
    include_metalora: bool = True     # False = perturb only m2p (lower-noise subset)
    include_mem_tokens: bool = False  # mem_tokens is a singularity in SHINE-ift_mqa_1qa
    min_rms: float = 0.0              # drop tensors with RMS < this threshold
    exclude_bias: bool = True         # drop ".bias" tensors (they eat budget without effect)

    # Rollout
    batch_size: int = 16              # contexts per step
    # questions_per_context is governed by the rollout config

    # Anchor
    anchor_coef_start: float = 1.0
    anchor_coef_end: float = 0.1
    anchor_decay_steps: int = 300

    # Training bookkeeping
    total_steps: int = 500
    eval_every: int = 25
    heldout_contexts: int = 50
    save_every: int = 50
    seed: int = 42
    out_dir: str = "runs/phase1_es_squad"

    # Adaptive anchor rollback
    heldout_regression_threshold: float = 2.0   # drop in reward*100 counted as regression
    heldout_rollback_window: int = 3            # evals in a row before bumping


class ESTrainer:
    def __init__(
        self,
        hypernet: ShineHypernet,
        rollout: SquadRollout,
        train_contexts: Sequence[SquadContext],
        heldout_contexts: Sequence[SquadContext],
        cfg: ESConfig,
    ):
        if cfg.mode not in {"one_sided", "antithetic"}:
            raise ValueError(f"cfg.mode must be 'one_sided' or 'antithetic', got {cfg.mode!r}.")

        self.hypernet = hypernet
        self.rollout = rollout
        self.train_contexts = list(train_contexts)
        self.heldout_contexts = list(heldout_contexts)
        self.cfg = cfg

        self.params = hypernet.all_perturbable_params(
            include_mem_tokens=cfg.include_mem_tokens,
            include_metalora=cfg.include_metalora,
            min_rms=cfg.min_rms,
            exclude_bias=cfg.exclude_bias,
        )
        logger.info(
            "ESTrainer: %d perturbable tensors "
            "(metalora=%s, mem_tokens=%s, exclude_bias=%s, min_rms=%.2g).",
            len(self.params), cfg.include_metalora, cfg.include_mem_tokens,
            cfg.exclude_bias, cfg.min_rms,
        )
        self.perturber = InPlacePerturber(
            self.params,
            sigma=cfg.sigma,
            noise_dtype=cfg.noise_dtype,
            sigma_mode=cfg.sigma_mode,
            rms_floor=cfg.rms_floor,
        )
        self.anchor = FrobeniusAnchor(
            self.params,
            AnchorSchedule(
                coef_start=cfg.anchor_coef_start,
                coef_end=cfg.anchor_coef_end,
                decay_steps=cfg.anchor_decay_steps,
            ),
        )
        self.rng = np.random.default_rng(cfg.seed)
        self.step_idx = 0
        self._heldout_hist: list[float] = []

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.out_dir / "train_log.jsonl"

    # -- utilities -------------------------------------------------------------

    def _sample_train_batch(self) -> list[SquadContext]:
        idx = self.rng.integers(0, len(self.train_contexts), size=self.cfg.batch_size)
        return [self.train_contexts[i] for i in idx]

    def _log(self, record: dict) -> None:
        record = {"step": self.step_idx, **record}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(record)

    # -- main step -------------------------------------------------------------

    def step(self) -> dict:
        contexts = self._sample_train_batch()
        if self.cfg.mode == "antithetic":
            return self._step_antithetic(contexts)
        return self._step_one_sided(contexts)

    def _step_one_sided(self, contexts) -> dict:
        """ES at Scale kernel: N one-sided perturbations, zscore reward, update.

        See third_party/es-fine-tuning-paper/es_fine-tuning_conciseness.py
        lines 205-319 for the reference loop.
        """
        seeds = self.rng.integers(0, 2**31, size=self.cfg.N).tolist()
        t0 = time.perf_counter()
        rewards: list[float] = []
        for s in seeds:
            self.perturber.apply(s, +1)
            rewards.append(self.rollout(contexts))
            self.perturber.restore(s, +1)   # same seed, negative sigma → restore
        rollout_time = time.perf_counter() - t0

        grads = one_sided_es_grad(
            rewards, seeds, self.params,
            sigma=self.cfg.sigma,
            normalize=self.cfg.reward_norm,
            noise_dtype=self.cfg.noise_dtype,
        )
        for (_, p), g in zip(self.params, grads):
            p.data.add_(g, alpha=self.cfg.lr)

        if self.cfg.anchor_coef_start > 0 or self.cfg.anchor_coef_end > 0:
            self.anchor.apply_step(self.params, self.cfg.lr, self.step_idx)
        self.step_idx += 1

        R = np.asarray(rewards, dtype=np.float64)
        return {
            "R_mean": float(R.mean()),
            "R_min": float(R.min()),
            "R_max": float(R.max()),
            "R_std": float(R.std()),
            "rollout_time_s": rollout_time,
        }

    def _step_antithetic(self, contexts) -> dict:
        if self.cfg.N % 2 != 0:
            raise ValueError(
                f"cfg.N must be even under mode='antithetic'; got {self.cfg.N}."
            )
        n_pairs = self.cfg.N // 2
        seeds = self.rng.integers(0, 2**31, size=n_pairs).tolist()

        t0 = time.perf_counter()
        R_plus: list[float] = []
        R_minus: list[float] = []
        for s in seeds:
            self.perturber.apply(s, +1)
            R_plus.append(self.rollout(contexts))
            self.perturber.restore(s, +1)

            self.perturber.apply(s, -1)
            R_minus.append(self.rollout(contexts))
            self.perturber.restore(s, -1)
        rollout_time = time.perf_counter() - t0

        grads = antithetic_es_grad(
            R_plus, R_minus, seeds, self.params,
            sigma=self.cfg.sigma,
            normalize=self.cfg.reward_norm,
            noise_dtype=self.cfg.noise_dtype,
        )
        for (_, p), g in zip(self.params, grads):
            p.data.add_(g, alpha=self.cfg.lr)

        if self.cfg.anchor_coef_start > 0 or self.cfg.anchor_coef_end > 0:
            self.anchor.apply_step(self.params, self.cfg.lr, self.step_idx)
        self.step_idx += 1

        return {
            "R_plus_mean": float(np.mean(R_plus)),
            "R_minus_mean": float(np.mean(R_minus)),
            "R_diff_mean": float((np.mean(R_plus) - np.mean(R_minus)) / 2),
            "rollout_time_s": rollout_time,
        }

    # -- evaluation ------------------------------------------------------------

    @torch.no_grad()
    def evaluate_heldout(self, max_contexts: int | None = None) -> float:
        n = max_contexts or self.cfg.heldout_contexts
        pool = self.heldout_contexts[:n]
        if not pool:
            return float("nan")
        return self.rollout(pool)

    def _maybe_bump_anchor(self, latest_heldout: float) -> None:
        self._heldout_hist.append(latest_heldout)
        window = self.cfg.heldout_rollback_window
        if len(self._heldout_hist) < window + 1:
            return
        recent = self._heldout_hist[-(window + 1):]
        # If the last `window` evals are each below the pre-window baseline by
        # more than `heldout_regression_threshold` (in reward*100 points),
        # bump the anchor coefficient.
        baseline = recent[0]
        regressions = [
            (baseline - v) * 100.0 >= self.cfg.heldout_regression_threshold
            for v in recent[1:]
        ]
        if all(regressions):
            self.anchor.bump_coef()
            self._log({"event": "anchor_bump",
                       "rollback_mul": self.anchor._rollback_mul,
                       "heldout_recent": recent})

    # -- driver ----------------------------------------------------------------

    def fit(self) -> None:
        # Baseline eval at step 0.
        hr = self.evaluate_heldout()
        init_hr = hr
        self._log({"event": "init_heldout", "heldout_reward": hr})
        best_hr = hr
        best_step = 0

        while self.step_idx < self.cfg.total_steps:
            step_info = self.step()
            step_info["event"] = "train_step"
            self._log(step_info)

            if self.step_idx % self.cfg.eval_every == 0:
                hr = self.evaluate_heldout()
                self._log({"event": "heldout_eval", "heldout_reward": hr,
                           "delta_vs_init": hr - init_hr})
                self._maybe_bump_anchor(hr)
                # Track + checkpoint the best heldout — without this, run #8's
                # peak at step 10 (heldout 0.6625, +0.0042 above init) was lost.
                if hr > best_hr:
                    best_hr, best_step = hr, self.step_idx
                    ckpt_dir = self.out_dir / "checkpoint-best"
                    self.hypernet.save(str(ckpt_dir))
                    self._log({
                        "event": "best_checkpoint_saved",
                        "path": str(ckpt_dir),
                        "heldout_reward": hr,
                        "delta_vs_init": hr - init_hr,
                    })

            if self.step_idx % self.cfg.save_every == 0:
                ckpt_dir = self.out_dir / f"checkpoint-{self.step_idx}"
                self.hypernet.save(str(ckpt_dir))
                self._log({"event": "checkpoint_saved", "path": str(ckpt_dir)})

        # Final eval + save.
        hr = self.evaluate_heldout()
        self._log({"event": "final_heldout", "heldout_reward": hr})
        if hr > best_hr:
            best_hr, best_step = hr, self.step_idx
            self.hypernet.save(str(self.out_dir / "checkpoint-best"))
        final_dir = self.out_dir / "checkpoint-final"
        self.hypernet.save(str(final_dir))
        self._log({"event": "final_checkpoint", "path": str(final_dir),
                   "best_step": best_step, "best_heldout": best_hr})
