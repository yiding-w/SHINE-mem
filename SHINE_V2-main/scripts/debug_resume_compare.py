#!/usr/bin/env python3
"""
Debug Resume Verification Script.

Compares two JSONL debug dump files (from Run A continuous and Run B resumed)
to verify that resume produces bit-identical training state.

Usage:
    python scripts/debug_resume_compare.py \
        logs/debug_resume_runA/node_0_debug_steps.jsonl \
        logs/debug_resume_runB/node_0_debug_steps.jsonl \
        --start_step 11

This compares steps 11-20 from both runs and reports any differences.
"""

import argparse
import json
import sys
from pathlib import Path


def load_steps(path: str, start_step: int):
    """Load steps from JSONL file, filtering to steps >= start_step."""
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip incomplete lines (can happen if training was forcefully stopped)
                continue
            if record["step"] >= start_step:
                steps.append(record)
    return steps


def compare_field(a_val, b_val, field_name: str, step: int, tolerance: float = 0.0):
    """Compare a single field. Returns (passed, message)."""
    if tolerance > 0 and isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
        diff = abs(a_val - b_val)
        if diff > tolerance:
            return False, f"Step {step} | {field_name}: A={a_val}, B={b_val}, diff={diff:.2e} > tol={tolerance:.2e}"
        return True, None
    else:
        if a_val != b_val:
            return False, f"Step {step} | {field_name}: A={a_val}, B={b_val}"
        return True, None


def main():
    parser = argparse.ArgumentParser(description="Compare debug resume dumps")
    parser.add_argument("file_a", help="Path to Run A (continuous) debug JSONL")
    parser.add_argument("file_b", help="Path to Run B (resumed) debug JSONL")
    parser.add_argument("--start_step", type=int, default=11,
                        help="First step to compare (default: 11)")
    parser.add_argument("--loss_tol", type=float, default=1e-4,
                        help="Tolerance for loss comparison (default: 1e-4)")
    parser.add_argument("--regu_tol", type=float, default=1e-2,
                        help="Tolerance for regu_sq_norm comparison (default: 1e-2)")
    args = parser.parse_args()

    steps_a = load_steps(args.file_a, args.start_step)
    steps_b = load_steps(args.file_b, args.start_step)

    if len(steps_a) != len(steps_b):
        print(f"ERROR: Different number of steps: A={len(steps_a)}, B={len(steps_b)}")
        sys.exit(1)

    if not steps_a:
        print(f"ERROR: No steps found >= {args.start_step}")
        sys.exit(1)

    print(f"Comparing {len(steps_a)} steps (step {args.start_step} to {steps_a[-1]['step']})")
    print(f"  File A: {args.file_a}")
    print(f"  File B: {args.file_b}")
    print(f"  Loss tolerance: {args.loss_tol}")
    print()

    # Fields that must be exactly equal
    exact_fields = [
        "step",
        "epoch",
        "lr",
        "prev_repo",
        "cur_repo",
        "repo_reset_triggered",
        "update_steps",
        "batch_counter",
        "context_ids_hash",
        "conv_ids_hash",
        "labels_hash",
        "context_lengths",
        "skipped",
    ]

    # Fields with floating-point tolerance
    # NOTE: epoch_loss_sum, epoch_steps, and running_loss are NOT compared because:
    # - epoch_loss_sum/epoch_steps: epoch-level accumulators not saved in checkpoints
    # - running_loss: saved before logging reset, so resume starts with stale value
    # None of these affect training correctness (only logging display).
    float_fields = {
        "loss": args.loss_tol,
        "regu_sq_norm": args.regu_tol,
        "reset_ratio": 1e-6,
        "mean_update_step": 1e-6,
        "grad_norm": 1e-4,
    }

    total_checks = 0
    failures = []

    for a, b in zip(steps_a, steps_b):
        step = a["step"]

        # Exact comparisons
        for field in exact_fields:
            if field in a and field in b:
                total_checks += 1
                passed, msg = compare_field(a[field], b[field], field, step)
                if not passed:
                    failures.append(msg)
            elif field in a or field in b:
                total_checks += 1
                failures.append(f"Step {step} | {field}: missing in {'B' if field in a else 'A'}")

        # Float comparisons
        for field, tol in float_fields.items():
            if field in a and field in b:
                total_checks += 1
                passed, msg = compare_field(a[field], b[field], field, step, tolerance=tol)
                if not passed:
                    failures.append(msg)

    # Summary
    print("=" * 60)
    if not failures:
        print(f"ALL PASSED ✓  ({total_checks} checks across {len(steps_a)} steps)")
    else:
        print(f"FAILURES: {len(failures)} / {total_checks} checks failed")
        print()
        for msg in failures[:50]:  # Show first 50 failures
            print(f"  ✗ {msg}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more failures")
    print("=" * 60)

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
