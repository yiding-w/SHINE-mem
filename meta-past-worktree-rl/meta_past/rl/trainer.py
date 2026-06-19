"""RL training loop for the SHINE hypernetwork (batched, no anchor).

Uses:
  - verl's ``compute_grpo_outcome_advantage`` / ``compute_rloo_outcome_advantage``
    for advantages (via :mod:`meta_past.rl.advantages`).
  - verl's ``agg_loss`` for REINFORCE policy loss aggregation
    (via :mod:`meta_past.rl.losses`).
  - Our own batched rollout orchestrator (:mod:`meta_past.rl.rollout`) and
    multi-replica vLLM pool (:mod:`meta_past.rl.vllm_pool`).

Each step samples ``B`` SquadContexts, generates ``B`` LoRAs in one hypernet
forward, pushes them in-memory to all ``N`` vLLM replicas, samples
``B*Q*K`` completions, and runs one re-score forward + backward.
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

from ..data.squad_contexts import SquadContext
from ..rollout.squad_rollout import SquadRollout
from ..shine_adapter import ShineHypernet
from . import set_phase
from .advantages import compute_advantages
from .rollout import RLRollout


logger = logging.getLogger("meta_past.rl.trainer")


# Verl-style metric keys: ``<namespace>/<name>``. Mirrors the layout used by
# ``verl/trainer/ppo`` so dashboards can be reused. We don't have a critic
# but verl uses ``critic/*`` for reward / advantage / return statistics
# regardless of whether a value head exists, so we follow that convention.
# Reference: ``Long-Digestor-Experiments/wandb/run-2026*/files/wandb-summary.json``
def _to_wandb_metrics(info: dict) -> dict:
    """Translate our flat trainer-step dict to verl-style wandb keys."""
    out: dict = {}

    # critic/rewards — environment rewards (here: F1 against gold answers)
    if "R_mean" in info:
        out["critic/rewards/mean"] = info["R_mean"]
        out["critic/rewards/min"] = info["R_min"]
        out["critic/rewards/max"] = info["R_max"]
        out["critic/rewards/std"] = info["R_std"]

    # critic/advantages — GRPO group-normalized advantage statistics
    if "adv_mean" in info:
        out["critic/advantages/mean"] = info["adv_mean"]
        out["critic/advantages/min"] = info["adv_min"]
        out["critic/advantages/max"] = info["adv_max"]
    if "adv_abs_max" in info:
        out["critic/advantages/abs_max"] = info["adv_abs_max"]

    # actor — policy stats. ``pg_loss`` is verl's name for the REINFORCE /
    # PPO policy-gradient loss; we have the on-policy REINFORCE form.
    if "loss" in info:
        out["actor/pg_loss"] = info["loss"]
    if "grad_norm" in info:
        out["actor/grad_norm"] = info["grad_norm"]
    if "lr" in info:
        out["actor/lr"] = info["lr"]

    # response_length — completion token counts
    if "rollout_sample_lens_mean" in info:
        out["response_length/mean"] = info["rollout_sample_lens_mean"]
        out["response_length/max"] = info["rollout_sample_lens_max"]
    if "rollout_sample_lens_min" in info:
        out["response_length/min"] = info["rollout_sample_lens_min"]
    if "rollout_n_oob_truncated" in info:
        # vLLM occasionally samples reserved special-token IDs above SHINE's
        # vocab; we truncate at first OOB. This mirrors verl's clip_ratio
        # semantics — fraction of completions that hit a hard cap.
        n_samples = info.get("rollout_n_samples", 0) or 1
        out["response_length/clip_ratio"] = info["rollout_n_oob_truncated"] / n_samples

    # prompt_length — input prompt token counts. We don't track per-sample
    # in this run but T_max is informative.
    if "rollout_T_max" in info:
        out["prompt_length/T_max_full_seq"] = info["rollout_T_max"]

    # timing_s — wall-clock seconds per phase, verl convention
    timing_map = {
        "t_rollout_s":      "timing_s/rollout",
        "t_backward_s":     "timing_s/update_actor",  # rescore + backward + optim
        "rollout_t_lora_s": "timing_s/hypernet",
        "rollout_t_push_s": "timing_s/lora_push",
        "rollout_t_sample_s": "timing_s/gen",         # vLLM sampling
    }
    for k_in, k_out in timing_map.items():
        if k_in in info:
            out[k_out] = info[k_in]
    if "t_rollout_s" in info and "t_backward_s" in info:
        out["timing_s/step"] = info["t_rollout_s"] + info["t_backward_s"]

    # perf — memory + throughput
    if "rollout_mem_post_hypernet" in info:
        m = info["rollout_mem_post_hypernet"]
        if isinstance(m, dict):
            out["perf/max_memory_allocated_gb"] = m.get("peak_alloc_GiB")
            out["perf/max_memory_reserved_gb"] = m.get("reserved_GiB")
    if "rollout_n_samples" in info and "t_rollout_s" in info \
            and "t_backward_s" in info:
        step_s = info["t_rollout_s"] + info["t_backward_s"]
        if step_s > 0:
            out["perf/throughput_samples_per_s"] = info["rollout_n_samples"] / step_s
            out["perf/time_per_step"] = step_s

    # training — top-level epoch / step counters
    if "step" in info:
        out["training/global_step"] = info["step"]

    # val-core — heldout eval metrics, dataset-namespaced as verl does
    if "heldout_reward" in info:
        out["val-core/squad/reward/mean@1"] = info["heldout_reward"]
    if "delta_vs_init" in info:
        out["val-core/squad/reward/delta_vs_init"] = info["delta_vs_init"]

    return out


@dataclass
class RLConfig:
    # advantage estimator — passed through to verl's registry
    adv_kind: str = "grpo"                  # "grpo" | "rloo" | "reinforce_plus_plus"
    norm_adv_by_std: bool = True            # GRPO: divide by std; False = Dr.GRPO
    loss_agg_mode: str = "token-mean"       # verl agg_loss mode

    # optim
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # loop
    total_steps: int = 500
    eval_every: int = 25
    heldout_contexts: int = 50
    save_every: int = 50
    seed: int = 42
    out_dir: str = "runs/phase1_rl_squad"

    # wandb (optional). Soft-imported; if wandb is not installed or
    # ``wandb_enabled=False``, the trainer just writes JSONL to disk.
    wandb_enabled: bool = False
    wandb_project: str = "meta-past-rl"
    wandb_name: str = ""                    # default → out_dir basename
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str = ""
    wandb_mode: str = "online"              # online | offline | disabled
    wandb_group: str = ""                   # optional group for runs


class RLTrainer:
    def __init__(
        self,
        hypernet: ShineHypernet,
        rollout: RLRollout,
        eval_rollout: SquadRollout,
        train_contexts: Sequence[SquadContext],
        heldout_contexts: Sequence[SquadContext],
        cfg: RLConfig,
    ):
        self.hypernet = hypernet
        self.rollout = rollout
        self.eval_rollout = eval_rollout
        self.train_contexts = list(train_contexts)
        self.heldout_contexts = list(heldout_contexts)
        self.cfg = cfg

        # DDP state. ``RANK`` / ``WORLD_SIZE`` come from torchrun. If unset
        # (single-process run), default to a 1-rank world so all the
        # ``rank == 0`` guards behave correctly.
        import torch.distributed as dist
        self._dist = dist
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_main = self.rank == 0
        self._dist_initialized = dist.is_available() and dist.is_initialized()

        self.params = hypernet.trainable_hypernet_params()
        tensor_params = [p for _, p in self.params]
        for _, p in self.params:
            p.requires_grad_(True)

        self.optim = torch.optim.AdamW(
            tensor_params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.rng = np.random.default_rng(cfg.seed)
        self.step_idx = 0
        # Epoch-based sampler: shuffle the pool once, draw sequentially,
        # reshuffle when exhausted. Avoids ever showing the same context
        # twice within a single pass through the pool.
        self._sampler_perm: np.ndarray | None = None
        self._sampler_pos: int = 0
        self._epoch: int = 0

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.out_dir / "train_log.jsonl"

        logger.info(
            "RLTrainer: %d trainable tensors (%d total params), adv=%s.",
            len(self.params),
            sum(p.numel() for p in tensor_params),
            cfg.adv_kind,
        )

        # Sanity-print vocab / special-token IDs. The embedding-lookup OOB
        # assertion is the most common silent failure on this path; if any of
        # these don't agree (tokenizer added tokens but model wasn't resized,
        # pad/eos id outside vocab, etc.) we want to know loudly at startup.
        tok = hypernet.tokenizer
        embed_rows = hypernet.metanetwork.metamodel.get_input_embeddings().num_embeddings
        logger.info(
            "tokenizer: len=%d, vocab_size=%d, pad_id=%s, eos_id=%s, "
            "model.embed_rows=%d",
            len(tok), tok.vocab_size, tok.pad_token_id, tok.eos_token_id,
            embed_rows,
        )
        if embed_rows < len(tok):
            logger.warning(
                "model embedding (%d) < tokenizer length (%d) — token IDs from "
                "the tokenizer's added-vocab range will OOB the embedding "
                "lookup. Did resize_token_embeddings get called?",
                embed_rows, len(tok),
            )

        if torch.cuda.is_available() and hypernet.device.type == "cuda":
            d = hypernet.device
            logger.warning(
                "[mem trainer-init] alloc=%.2f GiB  reserved=%.2f GiB",
                torch.cuda.memory_allocated(d) / 2**30,
                torch.cuda.memory_reserved(d) / 2**30,
            )

        # Soft-import wandb on rank 0 only. Other ranks never log.
        self._wandb = None
        if self.is_main and cfg.wandb_enabled and cfg.wandb_mode != "disabled":
            try:
                import wandb  # type: ignore
            except ImportError:
                logger.warning(
                    "wandb_enabled=True but wandb is not installed; falling "
                    "back to JSONL-only logging. Run `pip install wandb` to "
                    "enable.",
                )
            else:
                run_name = cfg.wandb_name or self.out_dir.name
                from dataclasses import asdict
                wandb_cfg = asdict(cfg)
                # Pull rollout config too for visibility on the dashboard.
                wandb_cfg.update({
                    f"rollout/{k}": v for k, v in
                    rollout.cfg.__dict__.items()
                })
                wandb.init(
                    project=cfg.wandb_project,
                    name=run_name,
                    tags=list(cfg.wandb_tags) or None,
                    notes=cfg.wandb_notes or None,
                    group=cfg.wandb_group or None,
                    mode=cfg.wandb_mode,
                    dir=str(self.out_dir),
                    config=wandb_cfg,
                )
                # Use our step counter as the x-axis for everything.
                wandb.define_metric("step")
                wandb.define_metric("*", step_metric="step")
                self._wandb = wandb
                logger.info(
                    "wandb run started: project=%s name=%s mode=%s",
                    cfg.wandb_project, run_name, cfg.wandb_mode,
                )

    # -- utilities -------------------------------------------------------------

    def _sample_train_batch(self, B_global: int) -> list[SquadContext]:
        """Pick the global batch this rank processes its slice of.

        Epoch-based sampling: maintain a shuffled permutation of the
        training pool, draw the next ``B_global`` indices sequentially,
        reshuffle (= start a new epoch) when exhausted. Within a single
        epoch every context appears exactly once, so consecutive draws
        cannot duplicate each other or repeat earlier draws.

        With ``world_size > 1``, the same permutation is computed on
        every rank (shared seed) and rank ``r`` gets indices
        ``[r*B_local, (r+1)*B_local)``.
        """
        if B_global % self.world_size != 0:
            raise ValueError(
                f"contexts_per_step={B_global} not divisible by world_size="
                f"{self.world_size}. Adjust contexts_per_step."
            )
        N = len(self.train_contexts)
        if B_global > N:
            raise ValueError(
                f"contexts_per_step={B_global} > train pool size {N}. "
                "Increase train_contexts or shrink contexts_per_step."
            )
        # Lazy-init / refresh the permutation when we run out.
        if self._sampler_perm is None or self._sampler_pos + B_global > N:
            self._sampler_perm = self.rng.permutation(N)
            self._sampler_pos = 0
            self._epoch += 1
            logger.info(
                "[sampler] starting epoch %d (pool size %d, B_global %d)",
                self._epoch, N, B_global,
            )
        idx = self._sampler_perm[self._sampler_pos:self._sampler_pos + B_global]
        self._sampler_pos += B_global
        B_local = B_global // self.world_size
        s = self.rank * B_local
        e = s + B_local
        return [self.train_contexts[int(i)] for i in idx[s:e]]

    def _log(self, record: dict) -> None:
        if not self.is_main:
            return
        record = {"step": self.step_idx, **record}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(record)
        if self._wandb is not None:
            wandb_metrics = _to_wandb_metrics(record)
            if wandb_metrics:
                wandb_metrics["step"] = int(self.step_idx)
                if "event" in record:
                    wandb_metrics["event"] = record["event"]
                self._wandb.log(wandb_metrics)

    # -- DDP helpers -----------------------------------------------------------

    def _all_reduce_grads(self) -> None:
        """Sum gradients across DDP ranks (SUM, not mean).

        Per-chunk loss is normalized by the GLOBAL token count
        (``total_response_tokens`` = all-reduce sum of local tokens), so each
        rank contributes ``local_sum / global_tokens``. Summing the per-rank
        gradients gives the correct global gradient. NB: this is *not* a
        mean-reduce — token-mean semantics make SUM the right op here.
        """
        if not self._dist_initialized or self.world_size == 1:
            return
        for _, p in self.params:
            if p.grad is None:
                continue
            self._dist.all_reduce(p.grad.data, op=self._dist.ReduceOp.SUM)

    def _all_reduce_scalar(self, value: float, op: str = "mean") -> float:
        """Reduce a Python scalar across ranks for logging."""
        if not self._dist_initialized or self.world_size == 1:
            return value
        t = torch.tensor([value], device=self.hypernet.device, dtype=torch.float64)
        if op == "mean":
            self._dist.all_reduce(t, op=self._dist.ReduceOp.SUM)
            t /= self.world_size
        elif op == "sum":
            self._dist.all_reduce(t, op=self._dist.ReduceOp.SUM)
        elif op == "max":
            self._dist.all_reduce(t, op=self._dist.ReduceOp.MAX)
        elif op == "min":
            self._dist.all_reduce(t, op=self._dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unknown reduce op: {op}")
        return float(t.item())

    # -- training step ---------------------------------------------------------

    def step(self) -> dict:
        B_global = self.rollout.cfg.contexts_per_step
        Q = self.rollout.cfg.questions_per_context
        K = self.rollout.cfg.rollouts_per_question
        B_local = B_global // self.world_size
        contexts = self._sample_train_batch(B_global)
        seed = int(self.rng.integers(0, 2**31))
        # Disjoint LoRA-id range per rank for log clarity (each rank's LLM
        # is independent so collisions wouldn't cause correctness issues,
        # but distinct ids make traces readable).
        lora_offset = self.rank * 4 * B_local

        t0 = time.perf_counter()
        group = self.rollout.rollout_step(
            contexts, step_idx=self.step_idx, seed=seed,
            global_lora_id_offset=lora_offset,
        )
        t_rollout = time.perf_counter() - t0

        # Advantages: GRPO is per-group within this rank's slice. Each rank
        # has its own group ids local to its slice; ranks don't share groups.
        # That means GRPO's per-group baseline is computed locally — fine
        # because each rank already has Q*K samples per group, plenty for
        # variance reduction.
        advantages = compute_advantages(
            rewards=group.rewards.detach(),
            response_mask=group.response_mask.to(group.rewards.dtype),
            group_ids=group.group_ids.cpu().numpy(),
            kind=self.cfg.adv_kind,
            norm_adv_by_std=self.cfg.norm_adv_by_std,
        )

        # Joint-microbatch: each chunk re-runs hypernet *with grad* on its
        # own slice of contexts, runs the rescore on the matching samples,
        # and immediately backwards. The chunk's hypernet + rescore graph
        # is freed before the next chunk starts. Peak memory bound is
        # ``mb`` (microbatch size in contexts), not ``B_local``.
        num_beams = Q * K  # samples per context (per rank)
        Lb = B_local
        mb = self.rollout.cfg.rescore_microbatch_contexts
        if mb <= 0 or mb >= Lb:
            chunk_starts = [0]
            chunk_ends = [Lb]
        else:
            chunk_starts = list(range(0, Lb, mb))
            chunk_ends = [min(s + mb, Lb) for s in chunk_starts]

        # token-mean normalizer must be the GLOBAL response-token count so
        # the loss across ranks averages correctly after grad all-reduce.
        local_tokens = float(group.response_mask.to(torch.float32).sum().item())
        total_response_tokens = self._all_reduce_scalar(local_tokens, op="sum")
        if total_response_tokens <= 0:
            total_response_tokens = 1.0

        # Pre-compute advantage statistics (over masked positions) once;
        # cheap, used for verl-style critic/advantages/* logging.
        with torch.no_grad():
            adv_mask = group.response_mask.to(advantages.dtype)
            adv_flat = advantages[adv_mask > 0]
            if adv_flat.numel() > 0:
                adv_mean_local = float(adv_flat.mean().item())
                adv_min_local = float(adv_flat.min().item())
                adv_max_local = float(adv_flat.max().item())
            else:
                adv_mean_local = adv_min_local = adv_max_local = 0.0

        self.optim.zero_grad(set_to_none=True)
        loss_running = 0.0
        adv_abs_max_running = 0.0

        t0 = time.perf_counter()
        n_chunks = len(chunk_starts)
        set_phase("rescore")
        for i, (s, e) in enumerate(zip(chunk_starts, chunk_ends)):
            rs, re = s * num_beams, e * num_beams
            # JOINT CHUNK: hypernet forward (with grad) + rescore + backward,
            # all on this chunk's contexts only. Each chunk's full graph
            # (hypernet + rescore) is freed at backward; peak memory is
            # bounded by ``mb`` regardless of B_local.
            chunk_loradict = self.hypernet.generate_lora_grad(
                group.evidence_ids[s:e],
                group.evidence_mask[s:e],
                use_gradient_checkpoint=self.rollout.cfg.use_gradient_checkpoint,
                microbatch_size=0,  # outer joint chunking handles it
            )
            chunk_logprobs, chunk_resp_mask = self.hypernet.score_answer_logprobs(
                loradict=chunk_loradict,
                input_ids=group.input_ids[rs:re],
                attention_mask=group.attention_mask[rs:re],
                answer_mask=group.answer_mask[rs:re],
                use_gradient_checkpoint=self.rollout.cfg.use_gradient_checkpoint,
                microbatch_contexts=0,
            )
            # Per-chunk loss contribution. token-mean: divide by GLOBAL token
            # count so contributions sum to the same loss as a single backward.
            chunk_adv = advantages[rs:re].to(chunk_logprobs.dtype)
            chunk_mask = chunk_resp_mask.to(chunk_logprobs.dtype)
            chunk_loss_contrib = (
                -(chunk_adv * chunk_logprobs * chunk_mask).sum()
                / total_response_tokens
            )
            # No retain_graph — each chunk has its own independent
            # hypernet+rescore graph, so backward() releases it cleanly.
            chunk_loss_contrib.backward()

            loss_running += float(chunk_loss_contrib.detach().item())
            with torch.no_grad():
                adv_abs_max_running = max(
                    adv_abs_max_running,
                    float((chunk_adv * chunk_mask).abs().max().item()),
                )
            del chunk_logprobs, chunk_resp_mask, chunk_loradict, chunk_loss_contrib

        # Free rescore inputs — chunk loop is done.
        group.evidence_ids = None  # type: ignore[assignment]
        group.evidence_mask = None  # type: ignore[assignment]
        group.input_ids = None  # type: ignore[assignment]
        group.attention_mask = None  # type: ignore[assignment]
        group.answer_mask = None  # type: ignore[assignment]

        # All-reduce gradients across ranks BEFORE clip+step.
        set_phase("allreduce")
        self._all_reduce_grads()

        set_phase("optim")
        tensor_params = [p for _, p in self.params]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            tensor_params, max_norm=self.cfg.grad_clip,
        )
        self.optim.step()
        t_backward = time.perf_counter() - t0

        self.step_idx += 1

        R = group.rewards.detach()
        current_lr = float(self.optim.param_groups[0]["lr"])
        # Reduce reward stats across ranks for global view; loss/grad are
        # already global because the gradient was summed.
        R_mean_global = self._all_reduce_scalar(float(R.mean()), op="mean")
        R_min_global = self._all_reduce_scalar(float(R.min()), op="min")
        R_max_global = self._all_reduce_scalar(float(R.max()), op="max")
        R_std_global = self._all_reduce_scalar(float(R.std(unbiased=False)), op="mean")
        loss_global = self._all_reduce_scalar(loss_running, op="sum")
        # Reduce advantage statistics across ranks for verl-style logging.
        adv_mean_global = self._all_reduce_scalar(adv_mean_local, op="mean")
        adv_min_global = self._all_reduce_scalar(adv_min_local, op="min")
        adv_max_global = self._all_reduce_scalar(adv_max_local, op="max")
        return {
            "R_mean": R_mean_global,
            "R_min": R_min_global,
            "R_max": R_max_global,
            "R_std": R_std_global,
            "adv_mean": adv_mean_global,
            "adv_min": adv_min_global,
            "adv_max": adv_max_global,
            "adv_abs_max": float(adv_abs_max_running),
            "loss": loss_global,
            "grad_norm": float(grad_norm),
            "lr": current_lr,
            "ctx_ids": [c.context_id for c in contexts],
            "t_rollout_s": t_rollout,
            "t_backward_s": t_backward,
            "rescore_chunks": int(n_chunks),
            "world_size": self.world_size,
            **{f"rollout_{k}": v for k, v in group.meta.items()},
        }

    # -- evaluation ------------------------------------------------------------

    @torch.no_grad()
    def evaluate_heldout(self, max_contexts: int | None = None) -> float:
        """Heldout eval is sharded across ranks: each rank evaluates its
        share of contexts, results are averaged. With 50 contexts × 8 ranks
        each rank handles ~6-7 contexts."""
        set_phase("heldout")
        n = max_contexts or self.cfg.heldout_contexts
        pool = self.heldout_contexts[:n]
        if not pool:
            return float("nan")
        # Shard contexts across ranks — round-robin keeps difficulty balanced.
        my_pool = [pool[i] for i in range(self.rank, len(pool), self.world_size)]
        if my_pool:
            local = self.eval_rollout(my_pool)
        else:
            local = float("nan")
        # Average via all-reduce. Skip NaN ranks by carrying a count.
        if self._dist_initialized and self.world_size > 1:
            valid = 0.0 if (isinstance(local, float) and (local != local)) else 1.0
            value = 0.0 if valid == 0.0 else float(local) * len(my_pool)
            t = torch.tensor([value, float(len(my_pool)) * valid],
                             device=self.hypernet.device, dtype=torch.float64)
            self._dist.all_reduce(t, op=self._dist.ReduceOp.SUM)
            total_value, total_count = float(t[0].item()), float(t[1].item())
            result = (total_value / total_count) if total_count > 0 else float("nan")
        else:
            result = float(local)
        # Free transient eval-side allocations before training resumes.
        if torch.cuda.is_available() and self.hypernet.device.type == "cuda":
            d = self.hypernet.device
            torch.cuda.empty_cache()
            if self.is_main:
                logger.warning(
                    "[mem post-heldout] alloc=%.2f GiB  reserved=%.2f GiB",
                    torch.cuda.memory_allocated(d) / 2**30,
                    torch.cuda.memory_reserved(d) / 2**30,
                )
        return result

    # -- driver ----------------------------------------------------------------

    def fit(self) -> None:
        try:
            hr = self.evaluate_heldout()
            init_hr = hr
            self._log({"event": "init_heldout", "heldout_reward": hr})
            best_hr = hr
            best_step = 0

            while self.step_idx < self.cfg.total_steps:
                info = self.step()
                info["event"] = "train_step"
                self._log(info)

                if self.step_idx % self.cfg.eval_every == 0:
                    hr = self.evaluate_heldout()
                    self._log({
                        "event": "heldout_eval",
                        "heldout_reward": hr,
                        "delta_vs_init": hr - init_hr,
                    })
                    if hr > best_hr:
                        best_hr, best_step = hr, self.step_idx
                        if self.is_main:
                            ckpt_dir = self.out_dir / "checkpoint-best"
                            self.hypernet.save(str(ckpt_dir))
                            self._log({
                                "event": "best_checkpoint_saved",
                                "path": str(ckpt_dir),
                                "heldout_reward": hr,
                                "delta_vs_init": hr - init_hr,
                            })

                if self.step_idx % self.cfg.save_every == 0:
                    if self.is_main:
                        ckpt_dir = self.out_dir / f"checkpoint-{self.step_idx}"
                        self.hypernet.save(str(ckpt_dir))
                        self._log({"event": "checkpoint_saved", "path": str(ckpt_dir)})

            hr = self.evaluate_heldout()
            self._log({"event": "final_heldout", "heldout_reward": hr})
            if self.is_main:
                if hr > best_hr:
                    best_hr, best_step = hr, self.step_idx
                    self.hypernet.save(str(self.out_dir / "checkpoint-best"))
                final_dir = self.out_dir / "checkpoint-final"
                self.hypernet.save(str(final_dir))
                self._log({
                    "event": "final_checkpoint",
                    "path": str(final_dir),
                    "best_step": best_step,
                    "best_heldout": best_hr,
                })
        finally:
            if self._wandb is not None:
                try:
                    self._wandb.finish()
                except Exception:
                    logger.exception("wandb.finish() failed; ignoring.")
