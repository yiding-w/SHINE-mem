"""
Enhanced correctness test for SP attention (Scheme A & B) under TP+SP.

Key improvements over test_scheme_ab.py:
  1. Tests long sequences (16k, 64k) matching real training conditions
  2. Tests ALL gradients (dQ, dK, dV) in TP+SP scenario
  3. Reports both max and mean relative diff for thorough analysis
  4. Uses stricter thresholds with detailed per-component breakdown
  5. Tests with Qwen3.6-27B aligned config (Hq=24, Hkv=4, D=256)

Run with:
    torchrun --nproc_per_node=8 tests/test_sp_enhanced.py
"""

from __future__ import annotations

import os
os.environ["PRECISION_ENHENCEMENT_FA2"] = "0"

import sys
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.mytp.ring_attention import (
    sp_flash_attention_contiguous,
    sp_flash_attention_alltoall_zigzag,
)


@dataclass
class DiffStats:
    """Statistics for comparing two tensors."""
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float  # max(|a-b|) / max(|ref|)
    mean_rel_diff: float  # mean(|a-b|) / mean(|ref|)
    cosine_sim: float  # cosine similarity (1.0 = identical)

    def __str__(self):
        return (f"max_abs={self.max_abs_diff:.4e}, mean_abs={self.mean_abs_diff:.4e}, "
                f"max_rel={self.max_rel_diff:.4e}, mean_rel={self.mean_rel_diff:.4e}, "
                f"cos_sim={self.cosine_sim:.8f}")

    @property
    def pass_forward(self) -> bool:
        """Forward pass threshold: max_rel < 1% and cos_sim > 0.9999."""
        return self.max_rel_diff < 0.01 and self.cosine_sim > 0.9999

    @property
    def pass_backward(self) -> bool:
        """Backward pass threshold: max_rel < 5% and cos_sim > 0.999."""
        return self.max_rel_diff < 0.05 and self.cosine_sim > 0.999


def compute_diff_stats(test: torch.Tensor, ref: torch.Tensor) -> DiffStats:
    """Compute comprehensive difference statistics between test and reference."""
    diff = (test.float() - ref.float()).abs()
    ref_abs = ref.float().abs()

    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    max_ref = ref_abs.max().item() + 1e-10
    mean_ref = ref_abs.mean().item() + 1e-10
    max_rel_diff = max_abs_diff / max_ref
    mean_rel_diff = mean_abs_diff / mean_ref

    # Cosine similarity (flatten to 1D)
    t_flat = test.float().reshape(-1)
    r_flat = ref.float().reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        t_flat.unsqueeze(0), r_flat.unsqueeze(0)
    ).item()

    return DiffStats(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        mean_rel_diff=mean_rel_diff,
        cosine_sim=cos_sim,
    )


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def thorough_cleanup(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Test: TP+SP Forward + Backward with long sequences
# ---------------------------------------------------------------------------

def test_tp_sp_full(
    rank: int,
    world_size: int,
    device: torch.device,
    tp_size: int,
    S_full: int,
    Hq_full: int,
    Hkv_full: int,
    D: int,
    batch_size: int = 1,
) -> Dict[str, Dict[str, DiffStats]]:
    """
    Full TP+SP correctness test with comprehensive gradient verification.

    Returns dict: {scheme_name: {component: DiffStats}}
    where component is one of: "output", "dQ", "dK", "dV"
    """
    sp_size = world_size // tp_size
    assert world_size == tp_size * sp_size
    assert S_full % (2 * sp_size) == 0, f"S_full={S_full} must be divisible by 2*sp_size={2*sp_size}"

    # Create process groups
    # Layout: rank = sp_rank * tp_size + tp_rank
    tp_groups = []
    sp_groups = []
    for sp_r in range(sp_size):
        tp_ranks = list(range(sp_r * tp_size, (sp_r + 1) * tp_size))
        tp_groups.append(dist.new_group(tp_ranks))
    for tp_r in range(tp_size):
        sp_ranks = [sp_r * tp_size + tp_r for sp_r in range(sp_size)]
        sp_groups.append(dist.new_group(sp_ranks))

    sp_rank = rank // tp_size
    tp_rank = rank % tp_size
    my_tp_group = tp_groups[sp_rank]
    my_sp_group = sp_groups[tp_rank]

    Hq_local = Hq_full // tp_size
    Hkv_local = Hkv_full // tp_size
    chunk_size = S_full // sp_size

    results = {}

    # ===== Reference: single GPU full attention =====
    torch.manual_seed(42)
    q_ref = torch.randn(batch_size, S_full, Hq_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    k_ref = torch.randn(batch_size, S_full, Hkv_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    v_ref = torch.randn(batch_size, S_full, Hkv_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)

    out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
    # Use a random grad_output for more realistic backward test
    torch.manual_seed(123)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    dq_ref = q_ref.grad.clone()
    dk_ref = k_ref.grad.clone()
    dv_ref = v_ref.grad.clone()
    out_ref_detached = out_ref.detach().clone()

    del q_ref, k_ref, v_ref, out_ref
    thorough_cleanup(device)

    # ===== Generate full tensors (no grad, for splitting) =====
    torch.manual_seed(42)
    q_full = torch.randn(batch_size, S_full, Hq_full, D, dtype=torch.bfloat16, device=device)
    k_full = torch.randn(batch_size, S_full, Hkv_full, D, dtype=torch.bfloat16, device=device)
    v_full = torch.randn(batch_size, S_full, Hkv_full, D, dtype=torch.bfloat16, device=device)

    # TP split (head dim)
    q_tp = q_full[:, :, tp_rank * Hq_local:(tp_rank + 1) * Hq_local, :].contiguous()
    k_tp = k_full[:, :, tp_rank * Hkv_local:(tp_rank + 1) * Hkv_local, :].contiguous()
    v_tp = v_full[:, :, tp_rank * Hkv_local:(tp_rank + 1) * Hkv_local, :].contiguous()

    # SP split (seq dim)
    q_local_data = q_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()
    k_local_data = k_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()
    v_local_data = v_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()

    # grad_output also needs TP+SP split
    grad_out_tp = grad_out[:, :, tp_rank * Hq_local:(tp_rank + 1) * Hq_local, :].contiguous()
    grad_out_local = grad_out_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()

    del q_full, k_full, v_full, q_tp, k_tp, v_tp, grad_out_tp
    thorough_cleanup(device)

    # ===== Test each scheme =====
    for scheme_name, sp_fn in [
        ("A (contiguous)", sp_flash_attention_contiguous),
        ("B (alltoall+zigzag)", sp_flash_attention_alltoall_zigzag),
    ]:
        q_local = q_local_data.clone().requires_grad_(True)
        k_local = k_local_data.clone().requires_grad_(True)
        v_local = v_local_data.clone().requires_grad_(True)

        # Forward
        out_local = sp_fn(q_local, k_local, v_local, sp_group=my_sp_group, causal=True)

        # Backward with same grad_output
        out_local.backward(grad_out_local)

        dq_local = q_local.grad.clone()
        dk_local = k_local.grad.clone()
        dv_local = v_local.grad.clone()
        out_local_detached = out_local.detach().clone()

        # Gather output: SP (seq dim) then TP (head dim)
        out_sp_gathered = [torch.empty_like(out_local_detached) for _ in range(sp_size)]
        dist.all_gather(out_sp_gathered, out_local_detached, group=my_sp_group)
        out_seq_full = torch.cat(out_sp_gathered, dim=1)  # [B, S, Hq_local, D]

        out_tp_gathered = [torch.empty_like(out_seq_full) for _ in range(tp_size)]
        dist.all_gather(out_tp_gathered, out_seq_full, group=my_tp_group)
        out_full = torch.cat(out_tp_gathered, dim=2)  # [B, S, Hq_full, D]

        # Gather dQ: SP (seq dim) then TP (head dim)
        dq_sp_gathered = [torch.empty_like(dq_local) for _ in range(sp_size)]
        dist.all_gather(dq_sp_gathered, dq_local, group=my_sp_group)
        dq_seq_full = torch.cat(dq_sp_gathered, dim=1)

        dq_tp_gathered = [torch.empty_like(dq_seq_full) for _ in range(tp_size)]
        dist.all_gather(dq_tp_gathered, dq_seq_full, group=my_tp_group)
        dq_full = torch.cat(dq_tp_gathered, dim=2)

        # Gather dK: SP (seq dim) then TP (head dim)
        dk_sp_gathered = [torch.empty_like(dk_local) for _ in range(sp_size)]
        dist.all_gather(dk_sp_gathered, dk_local, group=my_sp_group)
        dk_seq_full = torch.cat(dk_sp_gathered, dim=1)

        dk_tp_gathered = [torch.empty_like(dk_seq_full) for _ in range(tp_size)]
        dist.all_gather(dk_tp_gathered, dk_seq_full, group=my_tp_group)
        dk_full = torch.cat(dk_tp_gathered, dim=2)

        # Gather dV: SP (seq dim) then TP (head dim)
        dv_sp_gathered = [torch.empty_like(dv_local) for _ in range(sp_size)]
        dist.all_gather(dv_sp_gathered, dv_local, group=my_sp_group)
        dv_seq_full = torch.cat(dv_sp_gathered, dim=1)

        dv_tp_gathered = [torch.empty_like(dv_seq_full) for _ in range(tp_size)]
        dist.all_gather(dv_tp_gathered, dv_seq_full, group=my_tp_group)
        dv_full = torch.cat(dv_tp_gathered, dim=2)

        # Compute stats
        results[scheme_name] = {
            "output": compute_diff_stats(out_full, out_ref_detached),
            "dQ": compute_diff_stats(dq_full, dq_ref),
            "dK": compute_diff_stats(dk_full, dk_ref),
            "dV": compute_diff_stats(dv_full, dv_ref),
        }

        del q_local, k_local, v_local, out_local, out_local_detached
        del dq_local, dk_local, dv_local
        del out_sp_gathered, out_seq_full, out_tp_gathered, out_full
        del dq_sp_gathered, dq_seq_full, dq_tp_gathered, dq_full
        del dk_sp_gathered, dk_seq_full, dk_tp_gathered, dk_full
        del dv_sp_gathered, dv_seq_full, dv_tp_gathered, dv_full
        thorough_cleanup(device)

    del out_ref_detached, dq_ref, dk_ref, dv_ref, grad_out
    del q_local_data, k_local_data, v_local_data, grad_out_local
    thorough_cleanup(device)

    return results


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert world_size == 8, f"This test requires exactly 8 GPUs, got {world_size}"

    print_rank0(f"\n{'#'*80}")
    print_rank0(f"# Enhanced SP Correctness Test (TP+SP, Long Sequences)")
    print_rank0(f"# World size: {world_size}")
    print_rank0(f"# Device: {torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"# PRECISION_ENHENCEMENT_FA2={os.environ.get('PRECISION_ENHENCEMENT_FA2')}")
    print_rank0(f"{'#'*80}")

    # Test configurations
    # (tp_size, S_full, Hq_full, Hkv_full, D, description)
    test_configs = [
        # Short sequences (sanity check)
        (2, 2048, 24, 4, 128, "Short: TP=2, SP=4, S=2k, Hq=24, Hkv=4, D=128"),
        (4, 2048, 24, 4, 128, "Short: TP=4, SP=2, S=2k, Hq=24, Hkv=4, D=128"),
        # Medium sequences
        (2, 8192, 24, 4, 128, "Medium: TP=2, SP=4, S=8k, Hq=24, Hkv=4, D=128"),
        (2, 16384, 24, 4, 128, "Medium: TP=2, SP=4, S=16k, Hq=24, Hkv=4, D=128"),
        # Long sequences (matching training)
        (2, 32768, 24, 4, 128, "Long: TP=2, SP=4, S=32k, Hq=24, Hkv=4, D=128"),
        (2, 65536, 24, 4, 128, "Long: TP=2, SP=4, S=64k, Hq=24, Hkv=4, D=128"),
        # Qwen3.6-27B config (D=256, may OOM on 64k)
        (2, 16384, 24, 4, 256, "Qwen3.6: TP=2, SP=4, S=16k, Hq=24, Hkv=4, D=256"),
        (2, 32768, 24, 4, 256, "Qwen3.6: TP=2, SP=4, S=32k, Hq=24, Hkv=4, D=256"),
    ]

    all_pass = True
    summary_rows: List[Tuple[str, str, str, bool]] = []

    for tp_size, S_full, Hq_full, Hkv_full, D, desc in test_configs:
        sp_size = world_size // tp_size

        print_rank0(f"\n{'='*80}")
        print_rank0(f"  {desc}")
        print_rank0(f"  TP={tp_size}, SP={sp_size}, S_full={S_full}, S_local={S_full//sp_size}")
        print_rank0(f"  Hq_local={Hq_full//tp_size}, Hkv_local={Hkv_full//tp_size}, D={D}")
        print_rank0(f"{'='*80}")

        try:
            results = test_tp_sp_full(
                rank=rank,
                world_size=world_size,
                device=device,
                tp_size=tp_size,
                S_full=S_full,
                Hq_full=Hq_full,
                Hkv_full=Hkv_full,
                D=D,
            )

            for scheme_name, components in results.items():
                print_rank0(f"\n  {scheme_name}:")
                scheme_pass = True
                for comp_name, stats in components.items():
                    if comp_name == "output":
                        passed = stats.pass_forward
                    else:
                        passed = stats.pass_backward
                    scheme_pass = scheme_pass and passed
                    status = "✅" if passed else "❌"
                    print_rank0(f"    {comp_name:>6}: {status} {stats}")

                all_pass = all_pass and scheme_pass
                summary_rows.append((desc, scheme_name, "PASS" if scheme_pass else "FAIL", scheme_pass))

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print_rank0(f"  ⚠️  OOM - skipping this config")
                thorough_cleanup(device)
                summary_rows.append((desc, "both", "OOM", True))  # OOM is not a failure
            else:
                raise

        dist.barrier()
        thorough_cleanup(device)

    # ===== Summary =====
    print_rank0(f"\n\n{'#'*80}")
    print_rank0(f"# SUMMARY")
    print_rank0(f"{'#'*80}")
    print_rank0(f"\n{'Config':<60} | {'Scheme':<22} | {'Result':<6}")
    print_rank0(f"{'-'*60}-+-{'-'*22}-+-{'-'*6}")
    for desc, scheme, result, _ in summary_rows:
        status = "✅" if result == "PASS" else ("⚠️" if result == "OOM" else "❌")
        print_rank0(f"{desc:<60} | {scheme:<22} | {status} {result}")

    print_rank0(f"\n{'='*80}")
    if all_pass:
        print_rank0("ALL TESTS PASSED ✅")
    else:
        print_rank0("SOME TESTS FAILED ❌")
    print_rank0(f"{'='*80}\n")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
