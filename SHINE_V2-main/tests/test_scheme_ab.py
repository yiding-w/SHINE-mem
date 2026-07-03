"""
Test and benchmark Scheme A (contiguous ring) vs Scheme B (all-to-all + zigzag ring).

Verifies:
  1. Pure SP (no TP): Both schemes produce same forward output as non-SP flash_attn.
  2. Pure SP (no TP): Both schemes produce same backward gradients as non-SP.
  3. TP+SP: Both schemes produce same forward/backward as TP-only (no SP).
  4. Performance: Time and memory comparison between Scheme A and B.

Run with:
    torchrun --nproc_per_node=4 tests/test_scheme_ab.py
    torchrun --nproc_per_node=8 tests/test_scheme_ab.py
"""

from __future__ import annotations

import os
os.environ.setdefault("PRECISION_ENHENCEMENT_FA2", "0")

import sys
import time
import gc

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.mytp.ring_attention import (
    sp_flash_attention_contiguous,
    sp_flash_attention_alltoall_zigzag,
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


# ---------------------------------------------------------------------------
# Test 1: Pure SP forward correctness
# ---------------------------------------------------------------------------

def test_forward_pure_sp(rank, sp_size, device):
    """Test that both schemes produce same output as non-SP flash_attn."""
    print_rank0(f"\n{'='*70}")
    print_rank0(f"Test 1: Forward correctness - Pure SP (SP={sp_size})")
    print_rank0(f"{'='*70}")

    sp_group = dist.group.WORLD
    all_pass = True

    configs = [
        # (S, Hq, Hkv, D)
        (512, 8, 8, 128),
        (1024, 8, 8, 128),
        (2048, 14, 2, 128),
        (4096, 14, 2, 128),
        (2048, 8, 8, 64),
    ]

    for S, Hq, Hkv, D in configs:
        if S % (2 * sp_size) != 0:
            continue

        torch.manual_seed(42)
        q_full = torch.randn(1, S, Hq, D, dtype=torch.bfloat16, device=device)
        k_full = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device)
        v_full = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device)

        # Reference: single GPU flash_attn
        out_ref = flash_attn_func(q_full, k_full, v_full, causal=True)

        # Split into contiguous chunks
        chunk_size = S // sp_size
        q_local = q_full[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
        k_local = k_full[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
        v_local = v_full[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()

        # Scheme A: contiguous ring
        out_a_local = sp_flash_attention_contiguous(
            q_local.clone(), k_local.clone(), v_local.clone(),
            sp_group=sp_group, causal=True
        )
        # Gather output from all ranks
        out_a_gathered = [torch.empty_like(out_a_local) for _ in range(sp_size)]
        dist.all_gather(out_a_gathered, out_a_local, group=sp_group)
        out_a_full = torch.cat(out_a_gathered, dim=1)

        # Scheme B: all-to-all + zigzag ring
        out_b_local = sp_flash_attention_alltoall_zigzag(
            q_local.clone(), k_local.clone(), v_local.clone(),
            sp_group=sp_group, causal=True
        )
        out_b_gathered = [torch.empty_like(out_b_local) for _ in range(sp_size)]
        dist.all_gather(out_b_gathered, out_b_local, group=sp_group)
        out_b_full = torch.cat(out_b_gathered, dim=1)

        # Compare
        diff_a = (out_a_full - out_ref).abs().max().item()
        diff_b = (out_b_full - out_ref).abs().max().item()
        rel_a = diff_a / (out_ref.abs().mean().item() + 1e-8)
        rel_b = diff_b / (out_ref.abs().mean().item() + 1e-8)

        pass_a = diff_a < 0.02
        pass_b = diff_b < 0.02
        all_pass = all_pass and pass_a and pass_b

        print_rank0(
            f"  S={S:>5}, Hq={Hq:>2}, Hkv={Hkv}, D={D}: "
            f"A max_diff={diff_a:.4e} rel={rel_a:.4e} {'✅' if pass_a else '❌'} | "
            f"B max_diff={diff_b:.4e} rel={rel_b:.4e} {'✅' if pass_b else '❌'}"
        )
        dist.barrier()

    return all_pass


# ---------------------------------------------------------------------------
# Test 2: Pure SP backward correctness
# ---------------------------------------------------------------------------

def test_backward_pure_sp(rank, sp_size, device):
    """Test backward gradient correctness for both schemes."""
    print_rank0(f"\n{'='*70}")
    print_rank0(f"Test 2: Backward correctness - Pure SP (SP={sp_size})")
    print_rank0(f"{'='*70}")

    sp_group = dist.group.WORLD
    all_pass = True

    configs = [
        (1024, 8, 8, 128),
        (2048, 14, 2, 128),
        (4096, 8, 2, 128),
    ]

    for S, Hq, Hkv, D in configs:
        if S % (2 * sp_size) != 0:
            continue

        # --- Reference ---
        torch.manual_seed(42)
        q_ref = torch.randn(1, S, Hq, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        k_ref = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        v_ref = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device, requires_grad=True)

        out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
        out_ref.sum().backward()
        dq_ref = q_ref.grad.clone()
        dk_ref = k_ref.grad.clone()
        dv_ref = v_ref.grad.clone()

        # --- Scheme A ---
        torch.manual_seed(42)
        q_full = torch.randn(1, S, Hq, D, dtype=torch.bfloat16, device=device)
        k_full = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device)
        v_full = torch.randn(1, S, Hkv, D, dtype=torch.bfloat16, device=device)

        chunk_size = S // sp_size
        q_a = q_full[:, rank * chunk_size:(rank + 1) * chunk_size].clone().requires_grad_(True)
        k_a = k_full[:, rank * chunk_size:(rank + 1) * chunk_size].clone().requires_grad_(True)
        v_a = v_full[:, rank * chunk_size:(rank + 1) * chunk_size].clone().requires_grad_(True)

        out_a = sp_flash_attention_contiguous(q_a, k_a, v_a, sp_group=sp_group, causal=True)
        out_a.sum().backward()

        # Gather gradients
        dq_a_list = [torch.empty_like(q_a.grad) for _ in range(sp_size)]
        dk_a_list = [torch.empty_like(k_a.grad) for _ in range(sp_size)]
        dv_a_list = [torch.empty_like(v_a.grad) for _ in range(sp_size)]
        dist.all_gather(dq_a_list, q_a.grad, group=sp_group)
        dist.all_gather(dk_a_list, k_a.grad, group=sp_group)
        dist.all_gather(dv_a_list, v_a.grad, group=sp_group)
        dq_a_full = torch.cat(dq_a_list, dim=1)
        dk_a_full = torch.cat(dk_a_list, dim=1)
        dv_a_full = torch.cat(dv_a_list, dim=1)

        # --- Scheme B ---
        q_b = q_full[:, rank * chunk_size:(rank + 1) * chunk_size].clone().requires_grad_(True)
        k_b = k_full[:, rank * chunk_size:(rank + 1) * chunk_size].clone().requires_grad_(True)
        v_b = v_full[:, rank * chunk_size:(rank + 1) * chunk_size].clone().requires_grad_(True)

        out_b = sp_flash_attention_alltoall_zigzag(q_b, k_b, v_b, sp_group=sp_group, causal=True)
        out_b.sum().backward()

        dq_b_list = [torch.empty_like(q_b.grad) for _ in range(sp_size)]
        dk_b_list = [torch.empty_like(k_b.grad) for _ in range(sp_size)]
        dv_b_list = [torch.empty_like(v_b.grad) for _ in range(sp_size)]
        dist.all_gather(dq_b_list, q_b.grad, group=sp_group)
        dist.all_gather(dk_b_list, k_b.grad, group=sp_group)
        dist.all_gather(dv_b_list, v_b.grad, group=sp_group)
        dq_b_full = torch.cat(dq_b_list, dim=1)
        dk_b_full = torch.cat(dk_b_list, dim=1)
        dv_b_full = torch.cat(dv_b_list, dim=1)

        # Compare
        dq_diff_a = (dq_a_full - dq_ref).abs().max().item()
        dk_diff_a = (dk_a_full - dk_ref).abs().max().item()
        dv_diff_a = (dv_a_full - dv_ref).abs().max().item()

        dq_diff_b = (dq_b_full - dq_ref).abs().max().item()
        dk_diff_b = (dk_b_full - dk_ref).abs().max().item()
        dv_diff_b = (dv_b_full - dv_ref).abs().max().item()

        # Use relative diff based on max value (not mean) for better threshold
        dq_scale = dq_ref.abs().max().item() + 1e-8
        dk_scale = dk_ref.abs().max().item() + 1e-8
        dv_scale = dv_ref.abs().max().item() + 1e-8

        pass_a = (dq_diff_a / dq_scale < 0.1 and dk_diff_a / dk_scale < 0.1 and dv_diff_a / dv_scale < 0.1)
        pass_b = (dq_diff_b / dq_scale < 0.1 and dk_diff_b / dk_scale < 0.1 and dv_diff_b / dv_scale < 0.1)
        all_pass = all_pass and pass_a and pass_b

        print_rank0(
            f"  S={S:>5}, Hq={Hq:>2}, Hkv={Hkv}, D={D}:\n"
            f"    Scheme A: dQ={dq_diff_a:.4e} dK={dk_diff_a:.4e} dV={dv_diff_a:.4e} "
            f"(rel: {dq_diff_a/dq_scale:.4e}, {dk_diff_a/dk_scale:.4e}, {dv_diff_a/dv_scale:.4e}) {'✅' if pass_a else '❌'}\n"
            f"    Scheme B: dQ={dq_diff_b:.4e} dK={dk_diff_b:.4e} dV={dv_diff_b:.4e} "
            f"(rel: {dq_diff_b/dq_scale:.4e}, {dk_diff_b/dk_scale:.4e}, {dv_diff_b/dv_scale:.4e}) {'✅' if pass_b else '❌'}"
        )
        dist.barrier()

    return all_pass


# ---------------------------------------------------------------------------
# Test 3: TP+SP correctness (requires 8 GPUs)
# ---------------------------------------------------------------------------

def test_tp_sp(rank, world_size, device):
    """Test TP+SP correctness for both schemes.
    
    Uses 8 GPUs: TP=4, SP=2 or TP=2, SP=4.
    """
    if world_size < 8:
        print_rank0(f"\n{'='*70}")
        print_rank0(f"Test 3: TP+SP - SKIPPED (need 8 GPUs, have {world_size})")
        print_rank0(f"{'='*70}")
        return True

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Test 3: TP+SP correctness (8 GPUs)")
    print_rank0(f"{'='*70}")

    all_pass = True

    # Test with TP=2, SP=4
    tp_size = 2
    sp_size = world_size // tp_size

    # Create TP and SP groups
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

    # Test config: Qwen3.5-like
    S = 2048
    Hq_full = 28
    Hkv_full = 4
    D = 128
    Hq_local = Hq_full // tp_size
    Hkv_local = Hkv_full // tp_size

    torch.manual_seed(42)
    q_full = torch.randn(1, S, Hq_full, D, dtype=torch.bfloat16, device=device)
    k_full = torch.randn(1, S, Hkv_full, D, dtype=torch.bfloat16, device=device)
    v_full = torch.randn(1, S, Hkv_full, D, dtype=torch.bfloat16, device=device)

    # Reference: full attention (no SP, no TP)
    out_ref = flash_attn_func(q_full, k_full, v_full, causal=True)

    # TP split (along head dim) + SP split (along seq dim)
    chunk_size = S // sp_size
    q_tp = q_full[:, :, tp_rank * Hq_local:(tp_rank + 1) * Hq_local, :]
    k_tp = k_full[:, :, tp_rank * Hkv_local:(tp_rank + 1) * Hkv_local, :]
    v_tp = v_full[:, :, tp_rank * Hkv_local:(tp_rank + 1) * Hkv_local, :]

    q_local = q_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()
    k_local = k_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()
    v_local = v_tp[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].contiguous()

    # Scheme A
    out_a_local = sp_flash_attention_contiguous(
        q_local.clone(), k_local.clone(), v_local.clone(),
        sp_group=my_sp_group, causal=True
    )

    # Scheme B
    out_b_local = sp_flash_attention_alltoall_zigzag(
        q_local.clone(), k_local.clone(), v_local.clone(),
        sp_group=my_sp_group, causal=True
    )

    # Gather across SP (seq dim)
    out_a_sp_gathered = [torch.empty_like(out_a_local) for _ in range(sp_size)]
    dist.all_gather(out_a_sp_gathered, out_a_local, group=my_sp_group)
    out_a_seq_full = torch.cat(out_a_sp_gathered, dim=1)  # [B, S, Hq_local, D]

    out_b_sp_gathered = [torch.empty_like(out_b_local) for _ in range(sp_size)]
    dist.all_gather(out_b_sp_gathered, out_b_local, group=my_sp_group)
    out_b_seq_full = torch.cat(out_b_sp_gathered, dim=1)

    # Gather across TP (head dim)
    out_a_tp_gathered = [torch.empty_like(out_a_seq_full) for _ in range(tp_size)]
    dist.all_gather(out_a_tp_gathered, out_a_seq_full, group=my_tp_group)
    out_a_full = torch.cat(out_a_tp_gathered, dim=2)  # [B, S, Hq_full, D]

    out_b_tp_gathered = [torch.empty_like(out_b_seq_full) for _ in range(tp_size)]
    dist.all_gather(out_b_tp_gathered, out_b_seq_full, group=my_tp_group)
    out_b_full = torch.cat(out_b_tp_gathered, dim=2)

    diff_a = (out_a_full - out_ref).abs().max().item()
    diff_b = (out_b_full - out_ref).abs().max().item()
    rel_a = diff_a / (out_ref.abs().mean().item() + 1e-8)
    rel_b = diff_b / (out_ref.abs().mean().item() + 1e-8)

    pass_a = diff_a < 0.02
    pass_b = diff_b < 0.02
    all_pass = all_pass and pass_a and pass_b

    print_rank0(
        f"  TP={tp_size}, SP={sp_size}, S={S}, Hq={Hq_full}, Hkv={Hkv_full}:\n"
        f"    Scheme A: max_diff={diff_a:.4e}, rel={rel_a:.4e} {'✅' if pass_a else '❌'}\n"
        f"    Scheme B: max_diff={diff_b:.4e}, rel={rel_b:.4e} {'✅' if pass_b else '❌'}"
    )

    # Also test backward with TP+SP
    torch.manual_seed(42)
    q_full2 = torch.randn(1, S, Hq_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    k_full2 = torch.randn(1, S, Hkv_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    v_full2 = torch.randn(1, S, Hkv_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    out_ref2 = flash_attn_func(q_full2, k_full2, v_full2, causal=True)
    out_ref2.sum().backward()
    dq_ref2 = q_full2.grad.clone()

    # Scheme A backward
    torch.manual_seed(42)
    q_full3 = torch.randn(1, S, Hq_full, D, dtype=torch.bfloat16, device=device)
    k_full3 = torch.randn(1, S, Hkv_full, D, dtype=torch.bfloat16, device=device)
    v_full3 = torch.randn(1, S, Hkv_full, D, dtype=torch.bfloat16, device=device)

    q_tp3 = q_full3[:, :, tp_rank * Hq_local:(tp_rank + 1) * Hq_local, :]
    k_tp3 = k_full3[:, :, tp_rank * Hkv_local:(tp_rank + 1) * Hkv_local, :]
    v_tp3 = v_full3[:, :, tp_rank * Hkv_local:(tp_rank + 1) * Hkv_local, :]

    q_a3 = q_tp3[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].clone().requires_grad_(True)
    k_a3 = k_tp3[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].clone().requires_grad_(True)
    v_a3 = v_tp3[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].clone().requires_grad_(True)

    out_a3 = sp_flash_attention_contiguous(q_a3, k_a3, v_a3, sp_group=my_sp_group, causal=True)
    out_a3.sum().backward()

    # Gather dQ across SP and TP
    dq_a3_sp = [torch.empty_like(q_a3.grad) for _ in range(sp_size)]
    dist.all_gather(dq_a3_sp, q_a3.grad, group=my_sp_group)
    dq_a3_seq = torch.cat(dq_a3_sp, dim=1)
    dq_a3_tp = [torch.empty_like(dq_a3_seq) for _ in range(tp_size)]
    dist.all_gather(dq_a3_tp, dq_a3_seq, group=my_tp_group)
    dq_a3_full = torch.cat(dq_a3_tp, dim=2)

    dq_diff_a = (dq_a3_full - dq_ref2).abs().max().item()
    dq_scale = dq_ref2.abs().max().item() + 1e-8
    pass_bwd_a = dq_diff_a / dq_scale < 0.1
    all_pass = all_pass and pass_bwd_a

    # Scheme B backward
    q_b3 = q_tp3[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].clone().requires_grad_(True)
    k_b3 = k_tp3[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].clone().requires_grad_(True)
    v_b3 = v_tp3[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size].clone().requires_grad_(True)

    out_b3 = sp_flash_attention_alltoall_zigzag(q_b3, k_b3, v_b3, sp_group=my_sp_group, causal=True)
    out_b3.sum().backward()

    dq_b3_sp = [torch.empty_like(q_b3.grad) for _ in range(sp_size)]
    dist.all_gather(dq_b3_sp, q_b3.grad, group=my_sp_group)
    dq_b3_seq = torch.cat(dq_b3_sp, dim=1)
    dq_b3_tp = [torch.empty_like(dq_b3_seq) for _ in range(tp_size)]
    dist.all_gather(dq_b3_tp, dq_b3_seq, group=my_tp_group)
    dq_b3_full = torch.cat(dq_b3_tp, dim=2)

    dq_diff_b = (dq_b3_full - dq_ref2).abs().max().item()
    pass_bwd_b = dq_diff_b / dq_scale < 0.1
    all_pass = all_pass and pass_bwd_b

    print_rank0(
        f"  TP+SP Backward (dQ):\n"
        f"    Scheme A: dQ_diff={dq_diff_a:.4e}, rel={dq_diff_a/dq_scale:.4e} {'✅' if pass_bwd_a else '❌'}\n"
        f"    Scheme B: dQ_diff={dq_diff_b:.4e}, rel={dq_diff_b/dq_scale:.4e} {'✅' if pass_bwd_b else '❌'}"
    )

    dist.barrier()
    return all_pass


# ---------------------------------------------------------------------------
# Test 4: Performance benchmark
# ---------------------------------------------------------------------------

def test_performance(rank, sp_size, device):
    """Benchmark time and memory for both schemes."""
    print_rank0(f"\n{'='*70}")
    print_rank0(f"Test 4: Performance benchmark (SP={sp_size})")
    print_rank0(f"{'='*70}")

    sp_group = dist.group.WORLD

    configs = [
        # (S_full, Hq, Hkv, D, description)
        (4096, 14, 2, 128, "S=4k, GQA 14/2"),
        (8192, 14, 2, 128, "S=8k, GQA 14/2"),
        (16384, 14, 2, 128, "S=16k, GQA 14/2"),
        (32768, 14, 2, 128, "S=32k, GQA 14/2"),
    ]

    for S_full, Hq, Hkv, D, desc in configs:
        if S_full % (2 * sp_size) != 0:
            continue

        chunk_size = S_full // sp_size

        torch.manual_seed(42 + rank)
        q_local = torch.randn(1, chunk_size, Hq, D, dtype=torch.bfloat16, device=device)
        k_local = torch.randn(1, chunk_size, Hkv, D, dtype=torch.bfloat16, device=device)
        v_local = torch.randn(1, chunk_size, Hkv, D, dtype=torch.bfloat16, device=device)

        N_warmup = 3
        N_iter = 10

        # --- Scheme A: Forward ---
        for _ in range(N_warmup):
            sp_flash_attention_contiguous(q_local, k_local, v_local, sp_group=sp_group, causal=True)
        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats()
        mem_before_a = torch.cuda.memory_allocated()
        start = time.perf_counter()
        for _ in range(N_iter):
            out_a = sp_flash_attention_contiguous(q_local, k_local, v_local, sp_group=sp_group, causal=True)
        torch.cuda.synchronize()
        time_a_fwd = (time.perf_counter() - start) / N_iter * 1000
        mem_peak_a_fwd = torch.cuda.max_memory_allocated() - mem_before_a
        del out_a
        torch.cuda.empty_cache()
        gc.collect()

        # --- Scheme B: Forward ---
        for _ in range(N_warmup):
            sp_flash_attention_alltoall_zigzag(q_local, k_local, v_local, sp_group=sp_group, causal=True)
        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats()
        mem_before_b = torch.cuda.memory_allocated()
        start = time.perf_counter()
        for _ in range(N_iter):
            out_b = sp_flash_attention_alltoall_zigzag(q_local, k_local, v_local, sp_group=sp_group, causal=True)
        torch.cuda.synchronize()
        time_b_fwd = (time.perf_counter() - start) / N_iter * 1000
        mem_peak_b_fwd = torch.cuda.max_memory_allocated() - mem_before_b
        del out_b
        torch.cuda.empty_cache()
        gc.collect()

        # --- Scheme A: Forward + Backward ---
        q_a = q_local.clone().requires_grad_(True)
        k_a = k_local.clone().requires_grad_(True)
        v_a = v_local.clone().requires_grad_(True)

        for _ in range(N_warmup):
            out = sp_flash_attention_contiguous(q_a, k_a, v_a, sp_group=sp_group, causal=True)
            out.sum().backward()
            q_a.grad = None
            k_a.grad = None
            v_a.grad = None
        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        for _ in range(N_iter):
            out = sp_flash_attention_contiguous(q_a, k_a, v_a, sp_group=sp_group, causal=True)
            out.sum().backward()
            q_a.grad = None
            k_a.grad = None
            v_a.grad = None
        torch.cuda.synchronize()
        time_a_fwdbwd = (time.perf_counter() - start) / N_iter * 1000
        mem_peak_a_fwdbwd = torch.cuda.max_memory_allocated() - mem_before

        # --- Scheme B: Forward + Backward ---
        q_b = q_local.clone().requires_grad_(True)
        k_b = k_local.clone().requires_grad_(True)
        v_b = v_local.clone().requires_grad_(True)

        for _ in range(N_warmup):
            out = sp_flash_attention_alltoall_zigzag(q_b, k_b, v_b, sp_group=sp_group, causal=True)
            out.sum().backward()
            q_b.grad = None
            k_b.grad = None
            v_b.grad = None
        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        for _ in range(N_iter):
            out = sp_flash_attention_alltoall_zigzag(q_b, k_b, v_b, sp_group=sp_group, causal=True)
            out.sum().backward()
            q_b.grad = None
            k_b.grad = None
            v_b.grad = None
        torch.cuda.synchronize()
        time_b_fwdbwd = (time.perf_counter() - start) / N_iter * 1000
        mem_peak_b_fwdbwd = torch.cuda.max_memory_allocated() - mem_before

        # --- Single GPU reference (only on rank 0) ---
        time_ref_fwd = 0
        time_ref_fwdbwd = 0
        if rank == 0:
            torch.manual_seed(42)
            q_ref = torch.randn(1, S_full, Hq, D, dtype=torch.bfloat16, device=device, requires_grad=True)
            k_ref = torch.randn(1, S_full, Hkv, D, dtype=torch.bfloat16, device=device, requires_grad=True)
            v_ref = torch.randn(1, S_full, Hkv, D, dtype=torch.bfloat16, device=device, requires_grad=True)

            for _ in range(N_warmup):
                flash_attn_func(q_ref, k_ref, v_ref, causal=True)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(N_iter):
                flash_attn_func(q_ref, k_ref, v_ref, causal=True)
            torch.cuda.synchronize()
            time_ref_fwd = (time.perf_counter() - start) / N_iter * 1000

            for _ in range(N_warmup):
                out = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
                out.sum().backward()
                q_ref.grad = None
                k_ref.grad = None
                v_ref.grad = None
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(N_iter):
                out = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
                out.sum().backward()
                q_ref.grad = None
                k_ref.grad = None
                v_ref.grad = None
            torch.cuda.synchronize()
            time_ref_fwdbwd = (time.perf_counter() - start) / N_iter * 1000

            del q_ref, k_ref, v_ref
            torch.cuda.empty_cache()

        # Collect per-rank times to show load imbalance
        time_a_tensor = torch.tensor([time_a_fwd], device=device)
        time_b_tensor = torch.tensor([time_b_fwd], device=device)
        time_a_fwdbwd_tensor = torch.tensor([time_a_fwdbwd], device=device)
        time_b_fwdbwd_tensor = torch.tensor([time_b_fwdbwd], device=device)

        all_time_a = [torch.empty(1, device=device) for _ in range(sp_size)]
        all_time_b = [torch.empty(1, device=device) for _ in range(sp_size)]
        all_time_a_fb = [torch.empty(1, device=device) for _ in range(sp_size)]
        all_time_b_fb = [torch.empty(1, device=device) for _ in range(sp_size)]
        dist.all_gather(all_time_a, time_a_tensor, group=sp_group)
        dist.all_gather(all_time_b, time_b_tensor, group=sp_group)
        dist.all_gather(all_time_a_fb, time_a_fwdbwd_tensor, group=sp_group)
        dist.all_gather(all_time_b_fb, time_b_fwdbwd_tensor, group=sp_group)

        if rank == 0:
            times_a = [t.item() for t in all_time_a]
            times_b = [t.item() for t in all_time_b]
            times_a_fb = [t.item() for t in all_time_a_fb]
            times_b_fb = [t.item() for t in all_time_b_fb]

            print(f"\n  {desc} (S_local={chunk_size}):")
            print(f"  {'─'*60}")
            print(f"  Forward only:")
            print(f"    Single GPU:  {time_ref_fwd:.2f} ms")
            print(f"    Scheme A:    max={max(times_a):.2f} ms, min={min(times_a):.2f} ms, "
                  f"imbalance={max(times_a)/min(times_a):.2f}x")
            print(f"    Scheme B:    max={max(times_b):.2f} ms, min={min(times_b):.2f} ms, "
                  f"imbalance={max(times_b)/min(times_b):.2f}x")
            print(f"    Speedup vs single: A={time_ref_fwd/max(times_a):.2f}x, B={time_ref_fwd/max(times_b):.2f}x")
            print(f"  Forward + Backward:")
            print(f"    Single GPU:  {time_ref_fwdbwd:.2f} ms")
            print(f"    Scheme A:    max={max(times_a_fb):.2f} ms, min={min(times_a_fb):.2f} ms, "
                  f"imbalance={max(times_a_fb)/min(times_a_fb):.2f}x")
            print(f"    Scheme B:    max={max(times_b_fb):.2f} ms, min={min(times_b_fb):.2f} ms, "
                  f"imbalance={max(times_b_fb)/min(times_b_fb):.2f}x")
            print(f"    Speedup vs single: A={time_ref_fwdbwd/max(times_a_fb):.2f}x, B={time_ref_fwdbwd/max(times_b_fb):.2f}x")
            print(f"  Memory (peak, per rank):")
            print(f"    Scheme A fwd:     {mem_peak_a_fwd / 1024**2:.1f} MB")
            print(f"    Scheme B fwd:     {mem_peak_b_fwd / 1024**2:.1f} MB")
            print(f"    Scheme A fwd+bwd: {mem_peak_a_fwdbwd / 1024**2:.1f} MB")
            print(f"    Scheme B fwd+bwd: {mem_peak_b_fwdbwd / 1024**2:.1f} MB")
            print(f"    Per-rank fwd times A: {[f'{t:.2f}' for t in times_a]}")
            print(f"    Per-rank fwd times B: {[f'{t:.2f}' for t in times_b]}")

        dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()

    return True


# ---------------------------------------------------------------------------
# Test 5: Self-consistency
# ---------------------------------------------------------------------------

def test_self_consistency(rank, sp_size, device):
    """Test that repeated calls give identical results."""
    print_rank0(f"\n{'='*70}")
    print_rank0(f"Test 5: Self-consistency (SP={sp_size})")
    print_rank0(f"{'='*70}")

    sp_group = dist.group.WORLD
    all_pass = True
    S = 2048
    chunk_size = S // sp_size

    torch.manual_seed(42)
    q_full = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
    k_full = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
    v_full = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)

    q_local = q_full[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
    k_local = k_full[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()
    v_local = v_full[:, rank * chunk_size:(rank + 1) * chunk_size].contiguous()

    # Scheme A
    out_a1 = sp_flash_attention_contiguous(q_local, k_local, v_local, sp_group=sp_group, causal=True)
    out_a2 = sp_flash_attention_contiguous(q_local, k_local, v_local, sp_group=sp_group, causal=True)
    diff_a = (out_a1 - out_a2).abs().max().item()
    pass_a = diff_a == 0.0
    all_pass = all_pass and pass_a

    # Scheme B
    out_b1 = sp_flash_attention_alltoall_zigzag(q_local, k_local, v_local, sp_group=sp_group, causal=True)
    out_b2 = sp_flash_attention_alltoall_zigzag(q_local, k_local, v_local, sp_group=sp_group, causal=True)
    diff_b = (out_b1 - out_b2).abs().max().item()
    pass_b = diff_b == 0.0
    all_pass = all_pass and pass_b

    print_rank0(
        f"  Scheme A: diff={diff_a:.6e} {'✅' if pass_a else '❌'}\n"
        f"  Scheme B: diff={diff_b:.6e} {'✅' if pass_b else '❌'}"
    )

    dist.barrier()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    print_rank0(f"\n{'#'*70}")
    print_rank0(f"# Scheme A vs Scheme B: Correctness & Performance Test")
    print_rank0(f"# World size (SP size): {world_size}")
    print_rank0(f"# Device: {torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"# PRECISION_ENHENCEMENT_FA2={os.environ.get('PRECISION_ENHENCEMENT_FA2', 'not set')}")
    print_rank0(f"{'#'*70}")

    results = {}

    results["forward_pure_sp"] = test_forward_pure_sp(rank, world_size, device)
    results["backward_pure_sp"] = test_backward_pure_sp(rank, world_size, device)
    results["self_consistency"] = test_self_consistency(rank, world_size, device)
    results["tp_sp"] = test_tp_sp(rank, world_size, device)
    results["performance"] = test_performance(rank, world_size, device)

    # Summary
    print_rank0(f"\n{'='*70}")
    print_rank0(f"SUMMARY (SP={world_size})")
    print_rank0(f"{'='*70}")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print_rank0(f"  {name}: {status}")
        all_passed = all_passed and passed

    print_rank0(f"\n{'='*70}")
    if all_passed:
        print_rank0("ALL TESTS PASSED ✅")
    else:
        print_rank0("SOME TESTS FAILED ❌")
    print_rank0(f"{'='*70}\n")

    dist.destroy_process_group()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
