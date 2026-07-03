"""
Unit tests for TP + SP joint correctness on Full Attention.

Run with:
    torchrun --nproc_per_node=8 tests/test_tp_sp_attention.py

Tests verify that:
  1. TP=4 (no SP) on a full sequence produces the same output as
     TP=4 + SP=2 (zigzag ring attention over 2 SP ranks), where each
     SP rank holds half the sequence.
  2. Backward gradients (dQ, dK, dV w.r.t. hidden_states input) match
     between the two configurations.

Topology (8 GPUs, TP=4, SP=2):
  TP groups: [0,1,2,3], [4,5,6,7]
  SP groups: [0,4], [1,5], [2,6], [3,7]
  
  Each SP group contains ranks with the SAME tp_rank but different
  sequence chunks. This ensures ring attention exchanges KV for the
  same head shard.

Note: Requires PRECISION_ENHENCEMENT_FA2=0 for deterministic flash_attn.
"""

from __future__ import annotations

import os
os.environ.setdefault("PRECISION_ENHENCEMENT_FA2", "0")

import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.mytp.ring_attention import (
    zigzag_split,
    zigzag_unsplit,
    sp_flash_attention,
    _zigzag_indices,
)
from utils.mytp.tp_linear import (
    ColwiseLoraLinear,
    RowwiseLoraLinear,
    copy_to_tp_region,
    reduce_from_tp_region,
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


def create_tp_sp_groups(world_size, tp_size, sp_size):
    """Create TP and SP process groups.
    
    Layout: ranks are arranged as [SP0_TP0, SP0_TP1, ..., SP0_TP(tp-1), SP1_TP0, SP1_TP1, ...]
    i.e., rank = sp_rank * tp_size + tp_rank
    
    TP groups: consecutive tp_size ranks (same sp_rank)
      e.g., TP=4, SP=2: [0,1,2,3], [4,5,6,7]
    
    SP groups: same tp_rank across different sp_ranks
      e.g., TP=4, SP=2: [0,4], [1,5], [2,6], [3,7]
    """
    assert world_size == tp_size * sp_size, (
        f"world_size={world_size} != tp_size={tp_size} * sp_size={sp_size}"
    )
    
    rank = dist.get_rank()
    tp_rank = rank % tp_size
    sp_rank = rank // tp_size
    
    # Create TP groups
    tp_group = None
    for s in range(sp_size):
        ranks = list(range(s * tp_size, (s + 1) * tp_size))
        group = dist.new_group(ranks)
        if sp_rank == s:
            tp_group = group
    
    # Create SP groups
    sp_group = None
    for t in range(tp_size):
        ranks = [s * tp_size + t for s in range(sp_size)]
        group = dist.new_group(ranks)
        if tp_rank == t:
            sp_group = group
    
    return tp_rank, sp_rank, tp_group, sp_group


class SimpleTPAttention(torch.nn.Module):
    """A simplified TP attention module for testing.
    
    Implements: Q, K, V projection (colwise) -> attention -> O projection (rowwise)
    with proper TP communication.
    """
    
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, head_dim,
                 tp_rank, tp_size, tp_group, device, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        
        # Local head counts
        assert num_q_heads % tp_size == 0
        assert num_kv_heads % tp_size == 0
        self.q_heads_local = num_q_heads // tp_size
        self.kv_heads_local = num_kv_heads // tp_size
        
        # Colwise projections (shard output dim)
        self.q_weight = torch.nn.Parameter(
            torch.empty(self.q_heads_local * head_dim, hidden_size, device=device, dtype=dtype)
        )
        self.k_weight = torch.nn.Parameter(
            torch.empty(self.kv_heads_local * head_dim, hidden_size, device=device, dtype=dtype)
        )
        self.v_weight = torch.nn.Parameter(
            torch.empty(self.kv_heads_local * head_dim, hidden_size, device=device, dtype=dtype)
        )
        # Rowwise projection (shard input dim)
        self.o_weight = torch.nn.Parameter(
            torch.empty(hidden_size, self.q_heads_local * head_dim, device=device, dtype=dtype)
        )
    
    def load_full_weights(self, q_w, k_w, v_w, o_w):
        """Load from full (unsharded) weight matrices."""
        q_per = self.q_heads_local * self.head_dim
        kv_per = self.kv_heads_local * self.head_dim
        
        with torch.no_grad():
            self.q_weight.copy_(q_w[self.tp_rank * q_per:(self.tp_rank + 1) * q_per])
            self.k_weight.copy_(k_w[self.tp_rank * kv_per:(self.tp_rank + 1) * kv_per])
            self.v_weight.copy_(v_w[self.tp_rank * kv_per:(self.tp_rank + 1) * kv_per])
            self.o_weight.copy_(o_w[:, self.tp_rank * q_per:(self.tp_rank + 1) * q_per])
    
    def forward(self, hidden_states, sp_group=None):
        """
        Args:
            hidden_states: [B, S_local, hidden_size]
            sp_group: if not None, use ring attention over this SP group
        
        Returns:
            output: [B, S_local, hidden_size]
        """
        B, S_local, H = hidden_states.shape
        
        # ColwiseLinear: identity forward, all-reduce backward
        x = copy_to_tp_region(hidden_states, self.tp_group, self.tp_size)
        
        # Q, K, V projections (local)
        q = torch.nn.functional.linear(x, self.q_weight)  # [B, S_local, q_heads_local * head_dim]
        k = torch.nn.functional.linear(x, self.k_weight)  # [B, S_local, kv_heads_local * head_dim]
        v = torch.nn.functional.linear(x, self.v_weight)  # [B, S_local, kv_heads_local * head_dim]
        
        # Reshape to [B, S_local, num_heads, head_dim]
        q = q.view(B, S_local, self.q_heads_local, self.head_dim)
        k = k.view(B, S_local, self.kv_heads_local, self.head_dim)
        v = v.view(B, S_local, self.kv_heads_local, self.head_dim)
        
        # flash_attn and sp_flash_attention natively support GQA
        # (q_heads != kv_heads), no need to manually repeat KV
        
        # Attention
        if sp_group is not None:
            # Ring attention over SP group
            attn_out = sp_flash_attention(q, k, v, sp_group=sp_group, causal=True)
        else:
            # Standard flash attention
            from flash_attn import flash_attn_func
            attn_out = flash_attn_func(q, k, v, causal=True)
        
        # Reshape back: [B, S_local, q_heads_local * head_dim]
        attn_out = attn_out.reshape(B, S_local, self.q_heads_local * self.head_dim)
        
        # O projection (rowwise): all-reduce forward, identity backward
        out_partial = torch.nn.functional.linear(attn_out, self.o_weight)
        output = reduce_from_tp_region(out_partial, self.tp_group, self.tp_size)
        
        return output


def test_tp_sp_forward(rank, world_size, tp_size, sp_size, device):
    """Test that TP+SP forward output matches TP-only forward output."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: TP+SP Forward Correctness (TP={tp_size}, SP={sp_size})")
    print_rank0(f"{'='*60}")
    
    tp_rank, sp_rank, tp_group, sp_group = create_tp_sp_groups(world_size, tp_size, sp_size)
    
    # Model config (must be divisible by tp_size=4)
    hidden_size = 1024
    num_q_heads = 16
    num_kv_heads = 4
    head_dim = 64
    
    all_pass = True
    
    for S in [512, 1024, 2048]:
        if S % (2 * sp_size) != 0:
            continue
        
        B = 1
        
        # Generate full weights (same on all ranks)
        torch.manual_seed(42)
        q_w_full = torch.randn(num_q_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        k_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        v_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        o_w_full = torch.randn(hidden_size, num_q_heads * head_dim, device=device, dtype=torch.bfloat16) * 0.02
        
        # Generate full input (same on all ranks)
        torch.manual_seed(123 + S)
        hidden_full = torch.randn(B, S, hidden_size, device=device, dtype=torch.bfloat16)
        
        # --- TP+SP path ---
        # Create model with TP sharding
        model_tp_sp = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_tp_sp.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        
        # Zigzag split the input for this SP rank
        hidden_local = zigzag_split(hidden_full, sp_rank=sp_rank, sp_size=sp_size, seq_dim=1)
        
        # Forward with ring attention
        out_local = model_tp_sp(hidden_local, sp_group=sp_group)
        
        # Gather and unsplit to get full output
        out_tp_sp = zigzag_unsplit(out_local, sp_rank=sp_rank, sp_size=sp_size,
                                    seq_dim=1, sp_group=sp_group)
        
        # --- TP-only reference (no SP, full sequence) ---
        model_tp_only = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_tp_only.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        
        # Forward with full sequence (no SP)
        out_tp_only = model_tp_only(hidden_full, sp_group=None)
        
        # Compare
        max_diff = (out_tp_sp - out_tp_only).abs().max().item()
        rel_diff = max_diff / (out_tp_only.abs().max().item() + 1e-8)
        passed = max_diff < 0.05  # bf16 tolerance with TP+SP
        all_pass = all_pass and passed
        
        print_rank0(
            f"  S={S:>5}, tp_rank={tp_rank}, sp_rank={sp_rank}: "
            f"max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e} "
            f"{'✅' if passed else '❌'}"
        )
        dist.barrier()
    
    return all_pass


def test_tp_sp_backward(rank, world_size, tp_size, sp_size, device):
    """Test that TP+SP backward gradients match TP-only backward gradients."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: TP+SP Backward Correctness (TP={tp_size}, SP={sp_size})")
    print_rank0(f"{'='*60}")
    
    tp_rank, sp_rank, tp_group, sp_group = create_tp_sp_groups(world_size, tp_size, sp_size)
    
    # Model config (must be divisible by tp_size=4)
    hidden_size = 1024
    num_q_heads = 16
    num_kv_heads = 4
    head_dim = 64
    
    all_pass = True
    
    for S in [512, 1024, 2048]:
        if S % (2 * sp_size) != 0:
            continue
        
        B = 1
        
        # Generate full weights (same on all ranks)
        torch.manual_seed(42)
        q_w_full = torch.randn(num_q_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        k_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        v_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        o_w_full = torch.randn(hidden_size, num_q_heads * head_dim, device=device, dtype=torch.bfloat16) * 0.02
        
        # Generate full input
        torch.manual_seed(123 + S)
        hidden_full = torch.randn(B, S, hidden_size, device=device, dtype=torch.bfloat16)
        
        # --- TP+SP path ---
        model_tp_sp = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_tp_sp.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        
        # Zigzag split input
        hidden_local_sp = zigzag_split(hidden_full, sp_rank=sp_rank, sp_size=sp_size, seq_dim=1)
        hidden_local_sp = hidden_local_sp.clone().requires_grad_(True)
        
        # Forward + backward
        out_local_sp = model_tp_sp(hidden_local_sp, sp_group=sp_group)
        out_local_sp.sum().backward()
        
        # Gather gradient and unsplit
        grad_local_sp = hidden_local_sp.grad.clone()
        grad_full_sp = zigzag_unsplit(grad_local_sp, sp_rank=sp_rank, sp_size=sp_size,
                                       seq_dim=1, sp_group=sp_group)
        
        # Also gather weight gradients (they should be partial - need SP reduce)
        # For weight gradients: each SP rank has partial grad, need to sum across SP
        q_grad_sp = model_tp_sp.q_weight.grad.clone()
        k_grad_sp = model_tp_sp.k_weight.grad.clone()
        v_grad_sp = model_tp_sp.v_weight.grad.clone()
        o_grad_sp = model_tp_sp.o_weight.grad.clone()
        
        # All-reduce weight grads across SP group (since each SP rank only sees part of sequence)
        dist.all_reduce(q_grad_sp, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(k_grad_sp, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(v_grad_sp, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(o_grad_sp, op=dist.ReduceOp.SUM, group=sp_group)
        
        # --- TP-only reference ---
        model_tp_only = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_tp_only.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        
        hidden_full_ref = hidden_full.clone().requires_grad_(True)
        out_ref = model_tp_only(hidden_full_ref, sp_group=None)
        out_ref.sum().backward()
        
        grad_full_ref = hidden_full_ref.grad.clone()
        q_grad_ref = model_tp_only.q_weight.grad.clone()
        k_grad_ref = model_tp_only.k_weight.grad.clone()
        v_grad_ref = model_tp_only.v_weight.grad.clone()
        o_grad_ref = model_tp_only.o_weight.grad.clone()
        
        # Compare input gradients
        input_grad_diff = (grad_full_sp - grad_full_ref).abs().max().item()
        input_grad_rel = input_grad_diff / (grad_full_ref.abs().max().item() + 1e-8)
        
        # Compare weight gradients
        q_grad_diff = (q_grad_sp - q_grad_ref).abs().max().item()
        k_grad_diff = (k_grad_sp - k_grad_ref).abs().max().item()
        v_grad_diff = (v_grad_sp - v_grad_ref).abs().max().item()
        o_grad_diff = (o_grad_sp - o_grad_ref).abs().max().item()
        
        # Use relative tolerance for weight gradients (they can be large)
        q_grad_rel = q_grad_diff / (q_grad_ref.abs().max().item() + 1e-8)
        k_grad_rel = k_grad_diff / (k_grad_ref.abs().max().item() + 1e-8)
        v_grad_rel = v_grad_diff / (v_grad_ref.abs().max().item() + 1e-8)
        o_grad_rel = o_grad_diff / (o_grad_ref.abs().max().item() + 1e-8)
        
        max_weight_grad_rel = max(q_grad_rel, k_grad_rel, v_grad_rel, o_grad_rel)
        
        # Tolerance: bf16 with TP+SP - use relative tolerance
        # input_grad: relative diff < 1% is acceptable for bf16
        # weight_grad: relative diff < 1% is acceptable for bf16
        input_passed = input_grad_rel < 0.01
        weight_passed = max_weight_grad_rel < 0.01
        passed = input_passed and weight_passed
        all_pass = all_pass and passed
        
        print_rank0(
            f"  S={S:>5}: input_grad_rel={input_grad_rel:.4e}, "
            f"weight_grad_max_rel={max_weight_grad_rel:.4e} "
            f"{'✅' if passed else '❌'}"
        )
        if not passed:
            print_rank0(
                f"    Detail: dQ_w_rel={q_grad_rel:.4e}, dK_w_rel={k_grad_rel:.4e}, "
                f"dV_w_rel={v_grad_rel:.4e}, dO_w_rel={o_grad_rel:.4e}"
            )
        dist.barrier()
    
    return all_pass


def test_tp_sp_gqa_configs(rank, world_size, tp_size, sp_size, device):
    """Test various GQA configurations with TP+SP."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: TP+SP GQA Configs (TP={tp_size}, SP={sp_size})")
    print_rank0(f"{'='*60}")
    
    tp_rank, sp_rank, tp_group, sp_group = create_tp_sp_groups(world_size, tp_size, sp_size)
    
    all_pass = True
    
    # Various GQA configs that divide evenly by tp_size=4
    configs = [
        # (hidden_size, num_q_heads, num_kv_heads, head_dim, S)
        (512, 8, 4, 64, 1024),     # GQA ratio 2
        (896, 12, 4, 64, 1024),    # GQA ratio 3
        (1024, 16, 4, 64, 512),    # GQA ratio 4
        (512, 8, 8, 64, 1024),     # MHA (no GQA)
    ]
    
    for hidden_size, num_q_heads, num_kv_heads, head_dim, S in configs:
        if S % (2 * sp_size) != 0:
            continue
        if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            continue
        
        B = 1
        
        torch.manual_seed(42)
        q_w_full = torch.randn(num_q_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        k_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        v_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        o_w_full = torch.randn(hidden_size, num_q_heads * head_dim, device=device, dtype=torch.bfloat16) * 0.02
        
        torch.manual_seed(123)
        hidden_full = torch.randn(B, S, hidden_size, device=device, dtype=torch.bfloat16)
        
        # TP+SP
        model_sp = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_sp.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        
        hidden_local = zigzag_split(hidden_full, sp_rank=sp_rank, sp_size=sp_size, seq_dim=1)
        out_local = model_sp(hidden_local, sp_group=sp_group)
        out_sp = zigzag_unsplit(out_local, sp_rank=sp_rank, sp_size=sp_size,
                                seq_dim=1, sp_group=sp_group)
        
        # TP only
        model_ref = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_ref.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        out_ref = model_ref(hidden_full, sp_group=None)
        
        max_diff = (out_sp - out_ref).abs().max().item()
        passed = max_diff < 0.05
        all_pass = all_pass and passed
        
        print_rank0(
            f"  H={hidden_size}, Hq={num_q_heads}, Hkv={num_kv_heads}, D={head_dim}, S={S}: "
            f"max_diff={max_diff:.6e} {'✅' if passed else '❌'}"
        )
        dist.barrier()
    
    return all_pass


def test_tp_sp_self_consistency(rank, world_size, tp_size, sp_size, device):
    """Test that TP+SP gives identical results on repeated calls."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: TP+SP Self-Consistency (TP={tp_size}, SP={sp_size})")
    print_rank0(f"{'='*60}")
    
    tp_rank, sp_rank, tp_group, sp_group = create_tp_sp_groups(world_size, tp_size, sp_size)
    
    hidden_size = 1024
    num_q_heads = 16
    num_kv_heads = 4
    head_dim = 64
    S = 1024
    B = 1
    
    torch.manual_seed(42)
    q_w_full = torch.randn(num_q_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
    k_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
    v_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
    o_w_full = torch.randn(hidden_size, num_q_heads * head_dim, device=device, dtype=torch.bfloat16) * 0.02
    
    torch.manual_seed(123)
    hidden_full = torch.randn(B, S, hidden_size, device=device, dtype=torch.bfloat16)
    hidden_local = zigzag_split(hidden_full, sp_rank=sp_rank, sp_size=sp_size, seq_dim=1)
    
    model = SimpleTPAttention(
        hidden_size, num_q_heads, num_kv_heads, head_dim,
        tp_rank, tp_size, tp_group, device, torch.bfloat16
    )
    model.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
    
    out1 = model(hidden_local, sp_group=sp_group)
    out2 = model(hidden_local, sp_group=sp_group)
    
    max_diff = (out1 - out2).abs().max().item()
    passed = max_diff == 0.0
    
    print_rank0(f"  S={S}: diff={max_diff:.6e} {'✅' if passed else '❌'}")
    dist.barrier()
    
    return passed


def test_tp_sp_different_sp_sizes(rank, world_size, device):
    """Test TP+SP with different SP sizes (SP=2 and SP=4 if world_size allows)."""
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Test: TP+SP with different configurations")
    print_rank0(f"{'='*60}")
    
    all_pass = True
    
    # Test configs: (tp_size, sp_size) that multiply to world_size
    configs_to_test = []
    for tp in [2, 4]:
        for sp in [2, 4]:
            if tp * sp == world_size:
                configs_to_test.append((tp, sp))
    
    if not configs_to_test:
        print_rank0(f"  No valid TP*SP={world_size} configs found, skipping")
        return True
    
    for tp_size, sp_size in configs_to_test:
        print_rank0(f"\n  --- TP={tp_size}, SP={sp_size} ---")
        
        tp_rank, sp_rank, tp_group, sp_group = create_tp_sp_groups(world_size, tp_size, sp_size)
        
        # Use a config that divides by both tp sizes
        hidden_size = 512
        num_q_heads = 8
        num_kv_heads = 4  # Must divide by max tp_size
        head_dim = 64
        S = 1024
        B = 1
        
        if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            print_rank0(f"    Skipping: heads not divisible by tp_size={tp_size}")
            continue
        
        torch.manual_seed(42)
        q_w_full = torch.randn(num_q_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        k_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        v_w_full = torch.randn(num_kv_heads * head_dim, hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        o_w_full = torch.randn(hidden_size, num_q_heads * head_dim, device=device, dtype=torch.bfloat16) * 0.02
        
        torch.manual_seed(123)
        hidden_full = torch.randn(B, S, hidden_size, device=device, dtype=torch.bfloat16)
        
        # TP+SP
        model_sp = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_sp.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        
        hidden_local = zigzag_split(hidden_full, sp_rank=sp_rank, sp_size=sp_size, seq_dim=1)
        out_local = model_sp(hidden_local, sp_group=sp_group)
        out_sp = zigzag_unsplit(out_local, sp_rank=sp_rank, sp_size=sp_size,
                                seq_dim=1, sp_group=sp_group)
        
        # TP only (full sequence)
        model_ref = SimpleTPAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim,
            tp_rank, tp_size, tp_group, device, torch.bfloat16
        )
        model_ref.load_full_weights(q_w_full, k_w_full, v_w_full, o_w_full)
        out_ref = model_ref(hidden_full, sp_group=None)
        
        max_diff = (out_sp - out_ref).abs().max().item()
        passed = max_diff < 0.05
        all_pass = all_pass and passed
        
        print_rank0(
            f"    TP={tp_size}, SP={sp_size}: max_diff={max_diff:.6e} {'✅' if passed else '❌'}"
        )
        dist.barrier()
        
        # Need to destroy and recreate groups for next config
        # (dist.new_group is cumulative, but we track via local variables)
    
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    if world_size != 8:
        if rank == 0:
            print(f"ERROR: This test requires exactly 8 GPUs, got {world_size}")
        dist.destroy_process_group()
        return 1
    
    tp_size = 4
    sp_size = 2
    
    print_rank0(f"\n{'#'*60}")
    print_rank0(f"# TP + SP Joint Attention Tests")
    print_rank0(f"# World size: {world_size}, TP={tp_size}, SP={sp_size}")
    print_rank0(f"# Device: {torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"# PRECISION_ENHENCEMENT_FA2={os.environ.get('PRECISION_ENHENCEMENT_FA2', 'not set')}")
    print_rank0(f"# Topology:")
    print_rank0(f"#   TP groups: [0,1,2,3], [4,5,6,7]")
    print_rank0(f"#   SP groups: [0,4], [1,5], [2,6], [3,7]")
    print_rank0(f"{'#'*60}")
    
    results = {}
    
    results["forward"] = test_tp_sp_forward(rank, world_size, tp_size, sp_size, device)
    results["backward"] = test_tp_sp_backward(rank, world_size, tp_size, sp_size, device)
    results["gqa_configs"] = test_tp_sp_gqa_configs(rank, world_size, tp_size, sp_size, device)
    results["self_consistency"] = test_tp_sp_self_consistency(rank, world_size, tp_size, sp_size, device)
    results["different_configs"] = test_tp_sp_different_sp_sizes(rank, world_size, device)
    
    # Summary
    print_rank0(f"\n{'='*60}")
    print_rank0(f"SUMMARY (TP={tp_size}, SP={sp_size}, World={world_size})")
    print_rank0(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print_rank0(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print_rank0(f"\n{'='*60}")
    if all_passed:
        print_rank0("ALL TP+SP TESTS PASSED ✅")
    else:
        print_rank0("SOME TP+SP TESTS FAILED ❌")
    print_rank0(f"{'='*60}\n")
    
    dist.destroy_process_group()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
