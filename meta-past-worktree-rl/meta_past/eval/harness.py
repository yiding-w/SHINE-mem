"""End-to-end eval orchestration.

For one (dataset, modes) configuration:
  1. Load items via the dataset's adapter (parameterized by ``shots`` etc.).
  2. For each requested mode, run the appropriate generator.
  3. Score each prediction, aggregate, write per-item JSONL + summary.

The same items / scorer are shared across modes so ``shine`` vs
``icl`` vs ``zero`` numbers are directly comparable.

**Data-parallel sharding.** When launched under ``torchrun``, each rank
processes ``items[rank::world_size]``, generates predictions through
its own co-located vLLM + SHINE on its own GPU, then all-reduces the
per-item scores so rank 0 writes a single unified JSONL + summary.
Single-process launch (world_size=1) falls back to processing the
whole list locally with no all-reduce.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .datasets import REGISTRY, DatasetSpec
from .items import EvalItem
from .runner import EvalRunner
from .scoring import SCORERS, aggregate


def _dist_info() -> tuple[int, int, bool]:
    """Return ``(rank, world_size, is_dist_initialized)``."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        import torch.distributed as dist
        initialized = dist.is_initialized()
    except Exception:
        initialized = False
    return rank, world_size, initialized


def _all_gather_objects(local: list) -> list:
    """All-gather a list per rank → one flat list on every rank.

    Returns ``local`` unchanged when world_size == 1 or torch.distributed
    isn't initialized.
    """
    rank, world_size, ready = _dist_info()
    if world_size <= 1 or not ready:
        return list(local)
    import torch.distributed as dist
    gathered: list[list] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, list(local))
    flat: list = []
    for chunk in gathered:
        flat.extend(chunk or [])
    return flat


logger = logging.getLogger("meta_past.eval.harness")


@dataclass
class EvalResult:
    dataset: str
    bucket: str
    mode: str
    n_items: int
    score_mean: float
    elapsed_s: float
    shots: int | None = None
    extra: dict[str, Any] = None


def _score_items(spec: DatasetSpec, items: Sequence[EvalItem],
                 preds: Sequence[str]) -> list[float]:
    scorer = SCORERS[spec.scorer]
    kwargs_base = spec.scorer_kwargs or {}
    scores: list[float] = []
    for it, pred in zip(items, preds):
        kw = dict(kwargs_base)
        # Per-item overrides: MCQ adapters pass n_options in metadata,
        # HumanEval passes test_code / entry_point / prompt.
        for k in ("n_options", "test_code", "entry_point", "prompt"):
            if k in it.metadata:
                kw[k] = it.metadata[k]
        # HumanEval uses keys `test` (not test_code) and `prompt` from metadata.
        if spec.scorer == "humaneval_pass1":
            if "test" in it.metadata:
                kw["test_code"] = it.metadata["test"]
        scores.append(float(scorer(pred, it.references, **kw)))
    return scores


def evaluate(
    *,
    spec: DatasetSpec,
    items: Sequence[EvalItem],
    runner: EvalRunner,
    modes: Iterable[str],
    out_dir: Path,
    shots: int | None = None,
) -> list[EvalResult]:
    """Run one dataset across the requested modes; write per-mode JSONL
    + return aggregated results.

    Multi-rank: every rank receives the FULL ``items`` list, slices it
    locally via ``items[rank::world_size]``, runs its slice, then all-
    gathers per-item records so rank 0 writes the unified output.
    Single-rank: behavior is unchanged.
    """
    rank, world_size, _ = _dist_info()
    is_main = (rank == 0)
    results: list[EvalResult] = []
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Local slice for DP sharding. Stride-by-world preserves ordering so
    # gathered chunks can be re-zipped into a global list.
    local_items = list(items[rank::world_size]) if world_size > 1 else list(items)

    for mode in modes:
        if mode not in ("shine", "icl", "zero"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode not in spec.default_modes:
            logger.warning("dataset %s does not declare mode %s; running anyway",
                           spec.name, mode)

        t0 = time.perf_counter()
        if mode == "shine":
            preds = runner.run_shine(local_items)
        elif mode == "icl":
            preds = runner.run_icl(local_items)
        else:
            preds = runner.run_zero(local_items)
        local_elapsed = time.perf_counter() - t0

        local_scores = _score_items(spec, local_items, preds)

        # Build per-item records BEFORE the gather so we can serialize once.
        local_records = [
            {
                "rank": rank,
                "item_id": it.item_id,
                "mode": mode,
                "shots": shots,
                "pred": pred,
                "references": list(it.references),
                "score": float(s),
                "metadata": it.metadata,
            }
            for it, pred, s in zip(local_items, preds, local_scores)
        ]
        all_records = _all_gather_objects(local_records)

        # Rank-0 writes; everyone keeps the aggregate for their own return value.
        mean = aggregate([r["score"] for r in all_records])
        results.append(EvalResult(
            dataset=spec.name, bucket=spec.bucket, mode=mode,
            n_items=len(all_records),
            score_mean=mean,
            elapsed_s=local_elapsed,
            shots=shots,
            extra={"world_size": world_size},
        ))

        if is_main:
            log_name = f"{spec.name}.{mode}"
            if shots is not None:
                log_name += f".k{shots}"
            log_path = out_dir / f"{log_name}.jsonl"
            with open(log_path, "w") as f:
                for rec in all_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(
                "[%s/%s%s] mean=%.4f n=%d local_elapsed=%.1fs world=%d  -> %s",
                spec.name, mode,
                f"/k{shots}" if shots is not None else "",
                mean, len(all_records), local_elapsed, world_size,
                log_path.name,
            )

    return results


def run_one_dataset(
    *,
    name: str,
    runner: EvalRunner,
    out_dir: Path,
    modes: Iterable[str] | None = None,
    shots_list: Iterable[int] | None = None,
    limit: int | None = None,
) -> list[EvalResult]:
    """Driver for one dataset across mode × shots grid.

    ``shots_list`` only matters for bucket-B datasets; bucket-A/C
    adapters ignore it. If ``None`` and the dataset is bucket B, defaults
    to a single value of 8.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset: {name!r}. "
                       f"Known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    use_modes = tuple(modes) if modes else spec.default_modes

    all_results: list[EvalResult] = []
    if spec.bucket == "B":
        shots_iter = list(shots_list) if shots_list else [8]
        for k in shots_iter:
            items = spec.load(limit=limit, shots=k)
            all_results.extend(evaluate(
                spec=spec, items=items, runner=runner,
                modes=use_modes, out_dir=out_dir, shots=k,
            ))
    else:
        items = spec.load(limit=limit)
        all_results.extend(evaluate(
            spec=spec, items=items, runner=runner,
            modes=use_modes, out_dir=out_dir, shots=None,
        ))
    return all_results


def write_summary(results: Sequence[EvalResult], out_path: Path) -> None:
    """Append-friendly JSON summary (one ``EvalResult`` per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info("summary written to %s (%d rows)", out_path, len(results))
