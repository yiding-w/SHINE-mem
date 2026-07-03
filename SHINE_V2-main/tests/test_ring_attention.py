"""
Unit tests for Ring Flash Attention with Zigzag partition.

Run with:
    torchrun --nproc_per_node=4 tests/test_ring_attention.py
    torchrun --nproc_per_node=8 tests/test_ring_attention.py

Tests verify that:
  1. zigzag_split + zigzag_unsplit is a perfect round-trip (identity).
  2. sp_flash_attention with SP>1 produces the same output as standard
     flash_attn with SP=1 (full sequence on one GPU), within bf16 tolerance.
  3. Backward gradients (dQ, dK, dV) match between SP and non-SP.
  4. GQA (num_q_heads != num_kv_heads) works correctly.
  5. Self-consistency: repeated calls give identical results.
  6. Performance benchmark.

Note: Requires PRECISION_ENHENCEMENT_FA2=0 for deterministic flash_attn behavior
on H20 GPUs with head_dim=128.
"""

from __future__ import annotations

import os
# Must be set before importing flash_attn
os.environ.setdefault("PRECISION_ENHENCEMENT_FA2", "0")

import sys
import time

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.mytp.ring_attention import (
    zigzag_split,
    zigzag_unsplit,
    sp_flash_attention,
    _zigzag_indices,
)


def setup_distributed():
    """Initialize distributed environment."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def print_rank0(msg, rank=None):
    """Print only on rank 0."""
    if rank is None:
        rank = dist.get_rank()
    if rank == 0:
        print(msg, flush=True)


def run_ring_attn(q_full, k_full, v_full, SP, sp_group, rank):
    """Run ring attention and return full output (gathered from all ranks)."""
    S = q_full.shape[1]
    indices = _zigzag_indices(S, SP)
    my_indices = indices[rank].to(q_full.device)
    q_local = q_full[:, my_indices].detach()
    k_local = k_full[:, my_indices].detach()
    v_local = v_full[:, my_indices].detach()

    out_local = sp_flash_attention(q_local, k_local, v_local, sp_group=sp_group, causal=True)

    all_out = [torch.empty_like(out_local) for _ in range(SP)]
    dist.all_gather(all_out, out_local, group=sp_group)
    out_gathered = torch.cat(all_out, dim=1)
    all_indices = indices.reshape(-1).to(q_full.device)
    inv_indices = torch.empty_like(all_indices)
    inv_indices[all_indices] = torch.arange(S, device=q_full.device)
    return out_gathered[:, inv_indices]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zigzag_roundtrip(rank, sp_size, device):
    """Test that zigzag_split followed by zigzag_unsplit recovers the original."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: Zigzag split/unsplit round-trip (SP={sp_size})")
    print_rank0(f"{'='*60}")

    B, S, H, D = 1, 2048, 8, 128
    torch.manual_seed(42)  # Same seed on all ranks for identical x_full
    x_full = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)

    x_local = zigzag_split(x_full, sp_rank=rank, sp_size=sp_size, seq_dim=1)
    x_recovered = zigzag_unsplit(x_local, sp_rank=rank, sp_size=sp_size,
                                  seq_dim=1, sp_group=dist.group.WORLD)

    max_diff = (x_recovered - x_full).abs().max().item()
    passed = max_diff == 0.0
    print_rank0(f"  B=1: max_diff={max_diff:.6e} {'✅' if passed else '❌'}")

    # Also test B=2
    B2 = 2
    torch.manual_seed(123)
    x_full2 = torch.randn(B2, S, H, D, dtype=torch.bfloat16, device=device)
    x_local2 = zigzag_split(x_full2, sp_rank=rank, sp_size=sp_size, seq_dim=1)
    x_recovered2 = zigzag_unsplit(x_local2, sp_rank=rank, sp_size=sp_size,
                                   seq_dim=1, sp_group=dist.group.WORLD)
    max_diff2 = (x_recovered2 - x_full2).abs().max().item()
    passed2 = max_diff2 == 0.0
    print_rank0(f"  B=2: max_diff={max_diff2:.6e} {'✅' if passed2 else '❌'}")

    return passed and passed2


def test_forward_correctness(rank, sp_size, device):
    """Test forward correctness for various configs."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: Forward correctness (SP={sp_size})")
    print_rank0(f"{'='*60}")

    sp_group = dist.group.WORLD
    all_pass = True

    configs = [
        # (S, H, D, description)
        (512, 8, 64, "D=64"),
        (1024, 8, 64, "D=64"),
        (4096, 8, 64, "D=64"),
        (256, 8, 128, "D=128"),
        (512, 8, 128, "D=128"),
        (1024, 8, 128, "D=128"),
        (2048, 8, 128, "D=128"),
        (4096, 8, 128, "D=128"),
    ]

    for S, H, D, desc in configs:
        if S % (2 * sp_size) != 0:
            continue
        torch.manual_seed(42)
        q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)

        out_ref = flash_attn_func(q, k, v, causal=True)
        out_sp = run_ring_attn(q, k, v, sp_size, sp_group, rank)

        max_diff = (out_sp - out_ref).abs().max().item()
        passed = max_diff < 0.01
        all_pass = all_pass and passed
        print_rank0(f"  S={S:>5}, {desc}: max_diff={max_diff:.6e} {'✅' if passed else '❌'}")
        dist.barrier()

    return all_pass


def test_gqa(rank, sp_size, device):
    """Test GQA (grouped query attention) correctness."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: GQA correctness (SP={sp_size})")
    print_rank0(f"{'='*60}")

    sp_group = dist.group.WORLD
    all_pass = True

    configs = [
        # (S, Hq, Hkv, D)
        (1024, 14, 2, 128),   # Qwen3.5 TP=2
        (2048, 28, 4, 128),   # Qwen3.5 TP=1
        (4096, 14, 2, 128),
        (1024, 14, 2, 64),
        (2048, 7, 1, 128),    # Qwen3.5 TP=4
    ]

    for S, Hq, Hkv, D in configs:
        if S % (2 * sp_size) != 0:
            continue
        torch.manual_seed(42)
        q = torch.randn(1, S, Hq, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device)

        out_ref = flash_attn_func(q, k, v, causal=True)
        out_sp = run_ring_attn(q, k, v, sp_size, sp_group, rank)

        max_diff = (out_sp - out_ref).abs().max().item()
        passed = max_diff < 0.01
        all_pass = all_pass and passed
        print_rank0(f"  S={S}, Hq={Hq}, Hkv={Hkv}, D={D}: max_diff={max_diff:.6e} {'✅' if passed else '❌'}")
        dist.barrier()

    return all_pass


def test_self_consistency(rank, sp_size, device):
    """Test that repeated calls give identical results."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: Self-consistency (SP={sp_size})")
    print_rank0(f"{'='*60}")

    sp_group = dist.group.WORLD
    all_pass = True

    for S in [512, 1024, 2048]:
        torch.manual_seed(42)
        q = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)

        out1 = run_ring_attn(q, k, v, sp_size, sp_group, rank)
        out2 = run_ring_attn(q, k, v, sp_size, sp_group, rank)

        max_diff = (out1 - out2).abs().max().item()
        passed = max_diff == 0.0
        all_pass = all_pass and passed
        print_rank0(f"  S={S}: diff={max_diff:.6e} {'✅' if passed else '❌'}")
        dist.barrier()

    return all_pass


def test_backward(rank, sp_size, device):
    """Test backward gradient correctness."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: Backward gradients (SP={sp_size})")
    print_rank0(f"{'='*60}")

    sp_group = dist.group.WORLD
    all_pass = True

    for S, H, D in [(1024, 8, 64), (2048, 8, 128), (1024, 14, 128)]:
        if S % (2 * sp_size) != 0:
            continue

        torch.manual_seed(42)
        # Reference
        q_ref = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        k_ref = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        v_ref = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)

        out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
        out_ref.sum().backward()
        dq_ref = q_ref.grad.clone()
        dk_ref = k_ref.grad.clone()
        dv_ref = v_ref.grad.clone()

        # SP version
        torch.manual_seed(42)
        q_full = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
        k_full = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
        v_full = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)

        indices = _zigzag_indices(S, sp_size)
        my_indices = indices[rank].to(device)
        q_local = q_full[:, my_indices].requires_grad_(True)
        k_local = k_full[:, my_indices].requires_grad_(True)
        v_local = v_full[:, my_indices].requires_grad_(True)

        out_local = sp_flash_attention(q_local, k_local, v_local, sp_group=sp_group, causal=True)
        out_local.sum().backward()

        dq_local = q_local.grad
        dk_local = k_local.grad
        dv_local = v_local.grad

        # Gather and unsplit gradients
        dq_sp = zigzag_unsplit(dq_local, sp_rank=rank, sp_size=sp_size, seq_dim=1, sp_group=sp_group)
        dk_sp = zigzag_unsplit(dk_local, sp_rank=rank, sp_size=sp_size, seq_dim=1, sp_group=sp_group)
        dv_sp = zigzag_unsplit(dv_local, sp_rank=rank, sp_size=sp_size, seq_dim=1, sp_group=sp_group)

        dq_diff = (dq_sp - dq_ref).abs().max().item()
        dk_diff = (dk_sp - dk_ref).abs().max().item()
        dv_diff = (dv_sp - dv_ref).abs().max().item()

        passed = dq_diff < 0.1 and dk_diff < 0.1 and dv_diff < 0.1
        all_pass = all_pass and passed
        print_rank0(f"  S={S}, H={H}, D={D}: dQ={dq_diff:.4e}, dK={dk_diff:.4e}, dV={dv_diff:.4e} {'✅' if passed else '❌'}")
        dist.barrier()

    return all_pass


def test_performance(rank, sp_size, device):
    """Benchmark ring attention throughput."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: Performance (SP={sp_size})")
    print_rank0(f"{'='*60}")

    S, H, D = 8192, 14, 128
    torch.manual_seed(42)
    q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)

    indices = _zigzag_indices(S, sp_size)
    q_local = q[:, indices[rank].to(device)]
    k_local = k[:, indices[rank].to(device)]
    v_local = v[:, indices[rank].to(device)]

    sp_group = dist.group.WORLD

    # Warmup
    for _ in range(5):
        sp_flash_attention(q_local, k_local, v_local, sp_group=sp_group, causal=True)
    torch.cuda.synchronize()

    N = 20
    start = time.perf_counter()
    for _ in range(N):
        sp_flash_attention(q_local, k_local, v_local, sp_group=sp_group, causal=True)
    torch.cuda.synchronize()
    ring_time = (time.perf_counter() - start) / N * 1000

    if rank == 0:
        for _ in range(5):
            flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(N):
            flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        single_time = (time.perf_counter() - start) / N * 1000
        print(f"  S={S}, H={H}, D={D}")
        print(f"  Ring attention (SP={sp_size}): {ring_time:.2f} ms")
        print(f"  Single GPU flash_attn:         {single_time:.2f} ms")
        print(f"  Speedup: {single_time/ring_time:.2f}x")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    print_rank0(f"\n{'#'*60}")
    print_rank0(f"# Ring Flash Attention Unit Tests")
    print_rank0(f"# World size (SP size): {world_size}")
    print_rank0(f"# Device: {torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"# PRECISION_ENHENCEMENT_FA2={os.environ.get('PRECISION_ENHENCEMENT_FA2', 'not set')}")
    print_rank0(f"{'#'*60}")

    results = {}

    results["zigzag_roundtrip"] = test_zigzag_roundtrip(rank, world_size, device)
    results["forward_correctness"] = test_forward_correctness(rank, world_size, device)
    results["gqa"] = test_gqa(rank, world_size, device)
    results["self_consistency"] = test_self_consistency(rank, world_size, device)
    results["backward"] = test_backward(rank, world_size, device)
    results["performance"] = test_performance(rank, world_size, device)

    # Summary
    print_rank0(f"\n{'='*60}")
    print_rank0(f"SUMMARY (SP={world_size})")
    print_rank0(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print_rank0(f"  {name}: {status}")
        all_passed = all_passed and passed

    print_rank0(f"\n{'='*60}")
    if all_passed:
        print_rank0("ALL TESTS PASSED ✅")
    else:
        print_rank0("SOME TESTS FAILED ❌")
    print_rank0(f"{'='*60}\n")

    dist.destroy_process_group()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
