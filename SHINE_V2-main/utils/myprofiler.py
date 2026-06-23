"""
Profiler utilities for training performance analysis.

Provides a zero-overhead wrapper around torch.profiler that:
  - Does nothing when profiling is disabled (no import cost, no runtime cost)
  - Automatically stops training after profiling completes
  - Outputs a lightweight key_averages summary table (few KB)
  - Optionally writes full trace to local /tmp (fast) instead of network FS

Usage in training loop:
    from utils.myprofiler import TrainingProfiler
    profiler = TrainingProfiler(cfg.debug.get("profiler", {}), log_dir)
    ...
    with profiler.context():
        while global_step < max_steps:
            ...  # training step
            if profiler.step():
                break  # profiling complete, exit early
"""

import os
import logging
from contextlib import contextmanager, nullcontext
from typing import Optional

logger = logging.getLogger(__name__)


def _make_trace_handler(output_dir: str, rank: int, write_full_trace: bool):
    """Create a custom on_trace_ready callback.

    Instead of writing a multi-GB JSON trace to network FS, this:
      1. Always writes a lightweight key_averages summary table (~10KB)
      2. Optionally writes full trace to local /tmp (much faster than CephFS)
    """
    import torch

    def handler(prof):
        # 1. Write key_averages summary (the useful part for H800/H200 decision)
        summary_path = os.path.join(output_dir, f"rank_{rank}_summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 120 + "\n")
            f.write(f"PROFILER SUMMARY (rank {rank})\n")
            f.write("=" * 120 + "\n\n")

            # Table sorted by CUDA time (most useful for GPU-bound workloads)
            f.write("--- Sorted by CUDA time total ---\n")
            f.write(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=50
            ))
            f.write("\n\n")

            # Table sorted by CPU time (useful for detecting CPU bottlenecks)
            f.write("--- Sorted by CPU time total ---\n")
            f.write(prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=30
            ))
            f.write("\n\n")

            # Group by input shape (useful for understanding kernel efficiency)
            f.write("--- Grouped by input shape (top 30 by CUDA time) ---\n")
            f.write(prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=30
            ))
            f.write("\n\n")

            # NVTX ranges (our custom annotations: PhaseA, PhaseB, etc.)
            # These show up as regular events in key_averages
            f.write("--- NVTX Ranges (custom annotations) ---\n")
            nvtx_events = [
                e for e in prof.key_averages()
                if any(tag in e.key for tag in [
                    "Phase", "Step1_", "Step2_", "Step4_",
                    "Backward", "GradSync", "OptimizerStep", "DataLoading"
                ])
            ]
            if nvtx_events:
                for e in sorted(nvtx_events, key=lambda x: x.cuda_time, reverse=True):
                    f.write(
                        f"  {e.key:<40s}  "
                        f"CUDA(avg): {e.cuda_time / 1000:.1f}ms  "
                        f"CPU(avg): {e.cpu_time / 1000:.1f}ms  "
                        f"calls: {e.count}\n"
                    )
            else:
                f.write("  (no NVTX ranges found)\n")
            f.write("\n")

            # Memory summary
            f.write("--- Memory Summary ---\n")
            try:
                f.write(f"  Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB\n")
                f.write(f"  Peak CUDA memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB\n")
                f.write(f"  Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
            except Exception as ex:
                f.write(f"  (memory info unavailable: {ex})\n")

        logger.info(f"[Profiler] Summary written to: {summary_path}")

        # 2. Optionally write full trace to local /tmp (fast local disk)
        if write_full_trace:
            local_trace_dir = f"/tmp/shine_profiler_traces/rank_{rank}"
            os.makedirs(local_trace_dir, exist_ok=True)
            trace_path = os.path.join(local_trace_dir, "trace.json.gz")
            prof.export_chrome_trace(trace_path)
            logger.info(f"[Profiler] Full trace (gzipped) written to: {trace_path}")

    return handler


class TrainingProfiler:
    """Zero-overhead profiler wrapper.

    When disabled (default), all methods are no-ops with zero runtime cost.
    When enabled, wraps the training loop with torch.profiler.profile and
    automatically signals completion after the configured number of steps.

    Only rank 0 actually profiles (other ranks are no-ops to avoid
    redundant I/O and trace data).
    """

    def __init__(self, profiler_cfg: dict, log_dir: str = "."):
        """
        Args:
            profiler_cfg: Dict from cfg.debug.profiler (or empty dict).
            log_dir: Base directory for profiler output.
        """
        self._globally_enabled = bool(profiler_cfg.get("enabled", False))
        self.enabled = self._globally_enabled
        self._prof = None
        self._total_steps = 0
        self._current_step = 0
        self._rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

        if not self.enabled:
            return

        # All ranks track total_steps for synchronized exit
        wait = int(profiler_cfg.get("wait", 2))
        warmup = int(profiler_cfg.get("warmup", 2))
        active = int(profiler_cfg.get("active", 3))
        repeat = int(profiler_cfg.get("repeat", 1))
        self._total_steps = (wait + warmup + active) * repeat

        # Only profile on rank 0 (other ranks produce symmetric data in TP)
        if self._rank != 0:
            # Keep enabled=True so step() still counts steps for sync exit
            return

        import torch.profiler

        output_dir = profiler_cfg.get("output_dir", "./profiling_logs")
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(log_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        write_full_trace = bool(profiler_cfg.get("write_full_trace", False))

        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
            ),
            on_trace_ready=_make_trace_handler(output_dir, self._rank, write_full_trace),
            record_shapes=bool(profiler_cfg.get("record_shapes", False)),
            profile_memory=bool(profiler_cfg.get("profile_memory", True)),
            with_stack=False,  # Stack traces are the #1 cause of huge trace files
            with_flops=bool(profiler_cfg.get("with_flops", True)),
        )

        logger.info(
            f"[Profiler] Enabled: wait={wait}, warmup={warmup}, active={active}, "
            f"repeat={repeat}, total_steps={self._total_steps}, output={output_dir}, "
            f"write_full_trace={write_full_trace}"
        )

    @contextmanager
    def context(self):
        """Context manager that wraps the training loop.

        When disabled, yields immediately with zero overhead.
        When enabled, enters the torch.profiler context.
        """
        if not self.enabled or self._prof is None:
            yield
            return

        with self._prof:
            yield

    def step(self) -> bool:
        """Signal one training step completed.

        Returns:
            True if profiling is complete and training should exit early.
            False otherwise (including when profiling is disabled).

        All ranks return True simultaneously to ensure synchronized exit
        in PP mode (avoids NCCL timeout from asymmetric exit).
        """
        if not self.enabled:
            return False

        # Only rank 0 has the actual profiler object
        if self._prof is not None:
            self._prof.step()

        self._current_step += 1

        if self._current_step >= self._total_steps:
            if self._rank == 0:
                logger.info(
                    f"[Profiler] Profiling complete after {self._current_step} steps. "
                    f"Signaling early exit."
                )
            return True
        return False

    @property
    def should_exit(self) -> bool:
        """Check if profiling has completed (for use outside the step call)."""
        if not self._globally_enabled:
            return False
        return self._current_step >= self._total_steps
