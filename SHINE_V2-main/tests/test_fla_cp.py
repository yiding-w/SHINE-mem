"""
Unit tests for FLA CP (Context Parallel) integration with GatedDeltaNet.

Tests verify that the SP-aware forward path produces identical results
to single-GPU full-sequence computation, for both forward output and
backward gradients.

Test hierarchy:
  Test 1: Conv1d halo correctness (FLA causal_conv1d with cp_context)
  Test 2: chunk_gated_delta_rule CP correctness (pure SP, no TP)
  Test 3: Full GatedDeltaNet forward_sp correctness (pure SP)
  Test 4: TP+SP correctness (TP=2, SP=4)
  Test 5: Long sequence stability (S=16k/32k)

Run with:
    torchrun --nproc_per_node=8 tests/test_fla_cp.py

Requires 8 GPUs.
"""

from __future__ import annotations

import os
import sys
import gc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from utils.mytp.fla_cp import build_sp_cp_context


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@dataclass
class DiffStats:
    """Statistics for comparing two tensors."""
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    cosine_sim: float

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
    diff = (test.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    max_ref = ref_abs.max().item() + 1e-10
    mean_ref = ref_abs.mean().item() + 1e-10
    max_rel_diff = max_abs_diff / max_ref
    mean_rel_diff = mean_abs_diff / mean_ref
    t_flat = test.float().reshape(-1)
    r_flat = ref.float().reshape(-1)
    cos_sim = F.cosine_similarity(t_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()
    return DiffStats(max_abs_diff, mean_abs_diff, max_rel_diff, mean_rel_diff, cos_sim)


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
# Test 1: Conv1d halo correctness
# ---------------------------------------------------------------------------

def test_conv1d_halo(
    sp_group: ProcessGroup,
    device: torch.device,
    S_full: int = 4096,
    D: int = 512,
    kernel_size: int = 4,
) -> Dict[str, DiffStats]:
    """Test that FLA's causal_conv1d with cp_context matches single-GPU."""
    sp_world = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    S_local = S_full // sp_world

    # Generate full input and weight
    torch.manual_seed(42)
    x_full = torch.randn(1, S_full, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    weight = torch.randn(D, kernel_size, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Reference: single-GPU full sequence
    ref_out, _ = fla_causal_conv1d(x=x_full, weight=weight, activation="silu")
    torch.manual_seed(123)
    grad_out_full = torch.randn_like(ref_out)
    ref_out.backward(grad_out_full)
    dx_ref = x_full.grad.clone()
    ref_out_detached = ref_out.detach().clone()

    # Reset grads
    x_full.grad = None

    # SP: each rank gets its local chunk
    x_local_data = x_full.detach()[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    x_local = x_local_data.clone().requires_grad_(True)
    weight_sp = weight.detach().clone().requires_grad_(True)

    cp_context = build_sp_cp_context(
        seq_len_local=S_local,
        sp_group=sp_group,
        conv1d_kernel_size=kernel_size,
        device=device,
    )

    out_local, _ = fla_causal_conv1d(x=x_local, weight=weight_sp, activation="silu", cp_context=cp_context)

    # Backward
    grad_out_local = grad_out_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    out_local.backward(grad_out_local)
    dx_local = x_local.grad.clone()

    # Gather output
    out_gathered = [torch.empty_like(out_local) for _ in range(sp_world)]
    dist.all_gather(out_gathered, out_local.detach(), group=sp_group)
    out_full_sp = torch.cat(out_gathered, dim=1)

    # Gather dx
    dx_gathered = [torch.empty_like(dx_local) for _ in range(sp_world)]
    dist.all_gather(dx_gathered, dx_local, group=sp_group)
    dx_full_sp = torch.cat(dx_gathered, dim=1)

    results = {
        "output": compute_diff_stats(out_full_sp, ref_out_detached),
        "dx": compute_diff_stats(dx_full_sp, dx_ref),
    }
    return results


# ---------------------------------------------------------------------------
# Test 2: chunk_gated_delta_rule CP correctness
# ---------------------------------------------------------------------------

def test_chunk_gated_delta_rule_cp(
    sp_group: ProcessGroup,
    device: torch.device,
    S_full: int = 4096,
    num_heads: int = 8,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
) -> Dict[str, DiffStats]:
    """Test chunk_gated_delta_rule with cp_context matches single-GPU."""
    sp_world = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    S_local = S_full // sp_world

    # Generate full tensors
    torch.manual_seed(42)
    q_full = torch.randn(1, S_full, num_heads, head_k_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    k_full = torch.randn(1, S_full, num_heads, head_k_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    v_full = torch.randn(1, S_full, num_heads, head_v_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    # g must be negative (decay factor) to avoid state explosion
    g_full = -torch.rand(1, S_full, num_heads, dtype=torch.float32, device=device).requires_grad_(True)
    beta_full = torch.sigmoid(torch.randn(1, S_full, num_heads, dtype=torch.bfloat16, device=device)).requires_grad_(True)

    # Reference: single-GPU
    ref_out, _ = chunk_gated_delta_rule(
        q=q_full, k=k_full, v=v_full, g=g_full, beta=beta_full,
        use_qk_l2norm_in_kernel=True,
    )
    torch.manual_seed(123)
    grad_out_full = torch.randn_like(ref_out)
    ref_out.backward(grad_out_full)
    dq_ref = q_full.grad.clone()
    dk_ref = k_full.grad.clone()
    dv_ref = v_full.grad.clone()
    ref_out_detached = ref_out.detach().clone()

    # SP: local chunks
    def local_chunk(t, dim=1):
        return t.detach()[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()

    q_local = local_chunk(q_full).clone().requires_grad_(True)
    k_local = local_chunk(k_full).clone().requires_grad_(True)
    v_local = local_chunk(v_full).clone().requires_grad_(True)
    g_local = local_chunk(g_full).clone().requires_grad_(True)
    beta_local = local_chunk(beta_full).clone().requires_grad_(True)

    cp_context = build_sp_cp_context(
        seq_len_local=S_local,
        sp_group=sp_group,
        conv1d_kernel_size=4,  # Still needed for cp_context construction
        device=device,
    )

    out_local, _ = chunk_gated_delta_rule(
        q=q_local, k=k_local, v=v_local, g=g_local, beta=beta_local,
        use_qk_l2norm_in_kernel=True,
        cp_context=cp_context,
    )

    grad_out_local = grad_out_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    out_local.backward(grad_out_local)

    # Gather results
    def gather_sp(t):
        gathered = [torch.empty_like(t) for _ in range(sp_world)]
        dist.all_gather(gathered, t, group=sp_group)
        return torch.cat(gathered, dim=1)

    out_full_sp = gather_sp(out_local.detach())
    dq_full_sp = gather_sp(q_local.grad)
    dk_full_sp = gather_sp(k_local.grad)
    dv_full_sp = gather_sp(v_local.grad)

    results = {
        "output": compute_diff_stats(out_full_sp, ref_out_detached),
        "dQ": compute_diff_stats(dq_full_sp, dq_ref),
        "dK": compute_diff_stats(dk_full_sp, dk_ref),
        "dV": compute_diff_stats(dv_full_sp, dv_ref),
    }
    return results


# ---------------------------------------------------------------------------
# Test 3: Full GatedDeltaNet forward_sp (pure SP, no TP)
# ---------------------------------------------------------------------------

def test_full_forward_sp(
    sp_group: ProcessGroup,
    device: torch.device,
    S_full: int = 4096,
) -> Dict[str, DiffStats]:
    """Test TPQwen3_5GatedDeltaNet.forward_sp matches single-GPU forward."""
    from transformers import AutoConfig
    from utils.mytp.tp_gated_deltanet import TPQwen3_5GatedDeltaNet, load_gated_deltanet_weights_from_full
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    sp_world = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    S_local = S_full // sp_world

    config_path = "/apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp/models/Qwen3.6-27B"
    config = AutoConfig.from_pretrained(config_path).text_config
    layer_idx = 0

    # Create a single-GPU reference model
    torch.manual_seed(42)
    ref_model = Qwen3_5GatedDeltaNet(config, layer_idx).to(device=device, dtype=torch.bfloat16)
    ref_model.eval()

    # Create TP=1 SP model (tp_world=1 means no TP sharding, just SP)
    # We need a dummy process group for TP with world_size=1
    tp_group = dist.new_group([dist.get_rank()])
    sp_model = TPQwen3_5GatedDeltaNet(
        config, layer_idx,
        tp_rank=0, tp_world=1, tp_process_group=tp_group,
        sp_group=sp_group, sp_world=sp_world,
    ).to(device=device, dtype=torch.bfloat16)

    # Load weights from reference into SP model
    load_gated_deltanet_weights_from_full(sp_model, ref_model)
    sp_model.eval()

    # Generate input
    torch.manual_seed(100)
    hidden_full = torch.randn(1, S_full, config.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Reference forward + backward
    ref_out = ref_model(hidden_full)
    torch.manual_seed(200)
    grad_out_full = torch.randn_like(ref_out)
    ref_out.backward(grad_out_full)
    d_hidden_ref = hidden_full.grad.clone()
    ref_out_detached = ref_out.detach().clone()

    # SP forward + backward
    hidden_local_data = hidden_full.detach()[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    hidden_local = hidden_local_data.clone().requires_grad_(True)

    sp_out = sp_model(hidden_local)
    grad_out_local = grad_out_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    sp_out.backward(grad_out_local)
    d_hidden_local = hidden_local.grad.clone()

    # Gather
    def gather_sp(t):
        gathered = [torch.empty_like(t) for _ in range(sp_world)]
        dist.all_gather(gathered, t, group=sp_group)
        return torch.cat(gathered, dim=1)

    out_full_sp = gather_sp(sp_out.detach())
    d_hidden_full_sp = gather_sp(d_hidden_local)

    results = {
        "output": compute_diff_stats(out_full_sp, ref_out_detached),
        "d_hidden": compute_diff_stats(d_hidden_full_sp, d_hidden_ref),
    }

    del ref_model, sp_model
    return results


# ---------------------------------------------------------------------------
# Test 4: TP+SP correctness
# ---------------------------------------------------------------------------

def test_tp_sp(
    rank: int,
    world_size: int,
    device: torch.device,
    tp_size: int = 2,
    S_full: int = 4096,
) -> Dict[str, DiffStats]:
    """Test TP+SP correctness: TP=2, SP=4 vs single-GPU."""
    from transformers import AutoConfig
    from utils.mytp.tp_gated_deltanet import TPQwen3_5GatedDeltaNet, load_gated_deltanet_weights_from_full
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    sp_size = world_size // tp_size
    S_local = S_full // sp_size

    # Create process groups
    tp_groups = []
    sp_groups = []
    for sp_r in range(sp_size):
        tp_ranks = list(range(sp_r * tp_size, (sp_r + 1) * tp_size))
        tp_groups.append(dist.new_group(tp_ranks))
    for tp_r in range(tp_size):
        sp_ranks = [sp_r * tp_size + tp_r for sp_r in range(sp_size)]
        sp_groups.append(dist.new_group(sp_ranks))

    sp_rank_idx = rank // tp_size
    tp_rank_idx = rank % tp_size
    my_tp_group = tp_groups[sp_rank_idx]
    my_sp_group = sp_groups[tp_rank_idx]

    config_path = "/apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp/models/Qwen3.6-27B"
    config = AutoConfig.from_pretrained(config_path).text_config
    layer_idx = 0

    # Reference: single-GPU full model
    torch.manual_seed(42)
    ref_model = Qwen3_5GatedDeltaNet(config, layer_idx).to(device=device, dtype=torch.bfloat16)
    ref_model.eval()

    # TP+SP model
    tp_sp_model = TPQwen3_5GatedDeltaNet(
        config, layer_idx,
        tp_rank=tp_rank_idx, tp_world=tp_size, tp_process_group=my_tp_group,
        sp_group=my_sp_group, sp_world=sp_size,
    ).to(device=device, dtype=torch.bfloat16)
    load_gated_deltanet_weights_from_full(tp_sp_model, ref_model)
    tp_sp_model.eval()

    # Generate input
    torch.manual_seed(100)
    hidden_full = torch.randn(1, S_full, config.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Reference forward + backward
    ref_out = ref_model(hidden_full)
    torch.manual_seed(200)
    grad_out_full = torch.randn_like(ref_out)
    ref_out.backward(grad_out_full)
    d_hidden_ref = hidden_full.grad.clone()
    ref_out_detached = ref_out.detach().clone()

    # TP+SP forward + backward
    # Each SP rank gets its chunk of the sequence
    hidden_local_data = hidden_full.detach()[:, sp_rank_idx * S_local:(sp_rank_idx + 1) * S_local].contiguous()
    hidden_local = hidden_local_data.clone().requires_grad_(True)

    tp_sp_out = tp_sp_model(hidden_local)
    grad_out_local = grad_out_full[:, sp_rank_idx * S_local:(sp_rank_idx + 1) * S_local].contiguous()
    tp_sp_out.backward(grad_out_local)
    d_hidden_local = hidden_local.grad.clone()

    # Gather output across SP ranks
    def gather_sp(t, group):
        sp_w = dist.get_world_size(group)
        gathered = [torch.empty_like(t) for _ in range(sp_w)]
        dist.all_gather(gathered, t, group=group)
        return torch.cat(gathered, dim=1)

    out_sp_gathered = gather_sp(tp_sp_out.detach(), my_sp_group)
    d_hidden_sp_gathered = gather_sp(d_hidden_local, my_sp_group)

    # For TP+SP, the output from out_proj already has TP all-reduce built in
    # (RowwiseLoraLinear does all-reduce in forward), so out_sp_gathered is
    # the full output on each TP rank. We just compare with reference.
    results = {
        "output": compute_diff_stats(out_sp_gathered, ref_out_detached),
        "d_hidden": compute_diff_stats(d_hidden_sp_gathered, d_hidden_ref),
    }

    del ref_model, tp_sp_model
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert world_size == 8, f"This test requires exactly 8 GPUs, got {world_size}"

    print_rank0(f"\n{'#'*80}")
    print_rank0(f"# FLA CP Unit Tests (GatedDeltaNet SP Integration)")
    print_rank0(f"# World size: {world_size}")
    print_rank0(f"# Device: {torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"{'#'*80}")

    all_pass = True

    # ===== Test 1: Conv1d halo correctness =====
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 1: Conv1d halo correctness (SP=4)")
    print_rank0(f"{'='*80}")

    # Create SP group of size 4 (use first 4 ranks, replicate for second 4)
    sp_group_4 = None
    for start in range(0, world_size, 4):
        g = dist.new_group(list(range(start, start + 4)))
        if start <= rank < start + 4:
            sp_group_4 = g

    for S_full, D in [(2048, 512), (8192, 512), (16384, 1024)]:
        try:
            results = test_conv1d_halo(sp_group_4, device, S_full=S_full, D=D)
            for comp, stats in results.items():
                passed = stats.pass_forward if comp == "output" else stats.pass_backward
                status = "✅" if passed else "❌"
                all_pass = all_pass and passed
                print_rank0(f"    S={S_full:>6}, D={D:>4} | {comp:>6}: {status} {stats}")
        except Exception as e:
            print_rank0(f"    S={S_full}, D={D} | ERROR: {e}")
            all_pass = False

    thorough_cleanup(device)
    dist.barrier()

    # ===== Test 2: chunk_gated_delta_rule CP =====
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 2: chunk_gated_delta_rule CP (SP=4)")
    print_rank0(f"{'='*80}")

    for S_full, num_heads in [(2048, 8), (4096, 16), (8192, 8)]:
        try:
            results = test_chunk_gated_delta_rule_cp(
                sp_group_4, device, S_full=S_full, num_heads=num_heads
            )
            for comp, stats in results.items():
                passed = stats.pass_forward if comp == "output" else stats.pass_backward
                status = "✅" if passed else "❌"
                all_pass = all_pass and passed
                print_rank0(f"    S={S_full:>6}, H={num_heads:>2} | {comp:>6}: {status} {stats}")
        except Exception as e:
            print_rank0(f"    S={S_full}, H={num_heads} | ERROR: {e}")
            all_pass = False

    thorough_cleanup(device)
    dist.barrier()

    # ===== Test 3: Full GatedDeltaNet forward_sp (pure SP) =====
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 3: Full GatedDeltaNet forward_sp (SP=4, no TP)")
    print_rank0(f"{'='*80}")

    for S_full in [2048, 4096, 8192]:
        try:
            results = test_full_forward_sp(sp_group_4, device, S_full=S_full)
            for comp, stats in results.items():
                passed = stats.pass_forward if comp == "output" else stats.pass_backward
                status = "✅" if passed else "❌"
                all_pass = all_pass and passed
                print_rank0(f"    S={S_full:>6} | {comp:>10}: {status} {stats}")
        except Exception as e:
            print_rank0(f"    S={S_full} | ERROR: {e}")
            import traceback
            if rank == 0:
                traceback.print_exc()
            all_pass = False

    thorough_cleanup(device)
    dist.barrier()

    # ===== Test 4: TP+SP correctness =====
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 4: TP+SP correctness (TP=2, SP=4)")
    print_rank0(f"{'='*80}")

    for S_full in [2048, 4096, 8192]:
        try:
            results = test_tp_sp(rank, world_size, device, tp_size=2, S_full=S_full)
            for comp, stats in results.items():
                passed = stats.pass_forward if comp == "output" else stats.pass_backward
                status = "✅" if passed else "❌"
                all_pass = all_pass and passed
                print_rank0(f"    S={S_full:>6} | {comp:>10}: {status} {stats}")
        except Exception as e:
            print_rank0(f"    S={S_full} | ERROR: {e}")
            import traceback
            if rank == 0:
                traceback.print_exc()
            all_pass = False

    thorough_cleanup(device)
    dist.barrier()

    # ===== Test 5: Long sequence stability =====
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 5: Long sequence stability (TP=2, SP=4)")
    print_rank0(f"{'='*80}")

    for S_full in [16384, 32768]:
        try:
            results = test_tp_sp(rank, world_size, device, tp_size=2, S_full=S_full)
            for comp, stats in results.items():
                passed = stats.pass_forward if comp == "output" else stats.pass_backward
                status = "✅" if passed else "❌"
                all_pass = all_pass and passed
                print_rank0(f"    S={S_full:>6} | {comp:>10}: {status} {stats}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print_rank0(f"    S={S_full} | ⚠️  OOM - skipping")
                thorough_cleanup(device)
            else:
                raise
        except Exception as e:
            print_rank0(f"    S={S_full} | ERROR: {e}")
            import traceback
            if rank == 0:
                traceback.print_exc()
            all_pass = False

    thorough_cleanup(device)
    dist.barrier()

    # ===== Summary =====
    print_rank0(f"\n{'#'*80}")
    if all_pass:
        print_rank0("ALL FLA CP TESTS PASSED ✅")
    else:
        print_rank0("SOME FLA CP TESTS FAILED ❌")
    print_rank0(f"{'#'*80}\n")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
