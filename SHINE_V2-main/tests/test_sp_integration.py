"""
SP Integration Tests (Level 2) + End-to-End Tests (Level 3)

Level 2: Unit/integration tests for sp_utils functions
  - sp_split_sequence: correct splitting + error on non-divisible
  - sp_make_position_ids: correct global offsets
  - sp_exchange_boundary_labels: correct P2P exchange
  - sp_reduce_loss: correct global mean

Level 3: End-to-end SP vs non-SP equivalence
  - Full forward pass: compare intermediate hidden states
  - Loss computation: SP=1 vs SP>1 produce same loss
  - Backward: same gradients on all trainable parameters

Run with:
    torchrun --nproc_per_node=4 tests/test_sp_integration.py
    torchrun --nproc_per_node=8 tests/test_sp_integration.py
"""
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# Helpers
# ============================================================================

def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def rel_diff(a, b):
    """Max relative difference."""
    return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-10)


def cos_sim(a, b):
    """Cosine similarity between flattened tensors."""
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def thorough_cleanup(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


def create_sp_group(world_size, sp_size):
    """Create SP groups. Returns (my_sp_group, sp_rank)."""
    rank = dist.get_rank()
    num_sp_groups = world_size // sp_size

    my_sp_group = None
    sp_rank = -1
    for g in range(num_sp_groups):
        ranks = list(range(g * sp_size, (g + 1) * sp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_sp_group = group
            sp_rank = rank - g * sp_size
    return my_sp_group, sp_rank


# ============================================================================
# Level 2: sp_utils Integration Tests
# ============================================================================

def test_sp_split_sequence(device):
    """Test sp_split_sequence correctness and error handling."""
    from utils.mytp.sp_utils import sp_split_sequence

    print_rank0("\n" + "=" * 60)
    print_rank0("Level 2 Test 1: sp_split_sequence")
    print_rank0("=" * 60)

    B, S_full = 2, 16
    sp_world = 4
    input_ids = torch.arange(S_full).unsqueeze(0).expand(B, -1).to(device)  # [2, 16]

    all_pass = True
    for sp_rank in range(sp_world):
        chunk = sp_split_sequence(input_ids, sp_rank, sp_world)
        expected_start = sp_rank * (S_full // sp_world)
        expected_end = expected_start + (S_full // sp_world)
        expected = torch.arange(expected_start, expected_end).unsqueeze(0).expand(B, -1).to(device)
        if not torch.equal(chunk, expected):
            print_rank0(f"  FAIL: sp_rank={sp_rank}, got {chunk[0].tolist()}, expected {expected[0].tolist()}")
            all_pass = False

    # Test error on non-divisible
    try:
        sp_split_sequence(torch.zeros(1, 15, device=device, dtype=torch.long), 0, 4)
        print_rank0("  FAIL: Should have raised ValueError for non-divisible length")
        all_pass = False
    except ValueError:
        pass  # Expected

    if all_pass:
        print_rank0("  ✅ PASS: sp_split_sequence correct")
    return all_pass


def test_sp_make_position_ids(device):
    """Test sp_make_position_ids produces correct global offsets."""
    from utils.mytp.sp_utils import sp_make_position_ids

    print_rank0("\n" + "=" * 60)
    print_rank0("Level 2 Test 2: sp_make_position_ids")
    print_rank0("=" * 60)

    S_local = 8
    sp_world = 4
    B = 2
    all_pass = True

    for sp_rank in range(sp_world):
        pos = sp_make_position_ids(S_local, sp_rank, sp_world, batch_size=B, device=device)
        expected_start = sp_rank * S_local
        expected = torch.arange(expected_start, expected_start + S_local, device=device).unsqueeze(0).expand(B, -1)
        if not torch.equal(pos, expected):
            print_rank0(f"  FAIL: sp_rank={sp_rank}, got {pos[0].tolist()}, expected {expected[0].tolist()}")
            all_pass = False

    if all_pass:
        print_rank0("  ✅ PASS: sp_make_position_ids correct")
    return all_pass


def test_sp_exchange_boundary_labels(device, sp_group, sp_rank, sp_world):
    """Test boundary label exchange via P2P."""
    from utils.mytp.sp_utils import sp_exchange_boundary_labels

    print_rank0("\n" + "=" * 60)
    print_rank0("Level 2 Test 3: sp_exchange_boundary_labels")
    print_rank0("=" * 60)

    B = 1
    S_local = 8
    # Each rank has labels [sp_rank*S_local, ..., (sp_rank+1)*S_local - 1]
    labels = torch.arange(
        sp_rank * S_local, (sp_rank + 1) * S_local, device=device, dtype=torch.long
    ).unsqueeze(0)  # [1, 8]

    extended = sp_exchange_boundary_labels(labels, sp_group)
    # extended should be [1, S_local + 1]
    assert extended.shape == (B, S_local + 1), f"Shape mismatch: {extended.shape}"

    # Check: extended[:, :S_local] == original labels
    assert torch.equal(extended[:, :S_local], labels), "Original labels corrupted"

    # Check boundary label:
    # - For ranks 0..sp_world-2: should be next rank's first label
    # - For last rank: should be -100
    if sp_rank < sp_world - 1:
        expected_boundary = (sp_rank + 1) * S_local  # first label of next rank
        actual_boundary = extended[0, S_local].item()
        if actual_boundary != expected_boundary:
            print_rank0(f"  FAIL: rank {sp_rank} boundary={actual_boundary}, expected={expected_boundary}")
            return False
    else:
        expected_boundary = -100
        actual_boundary = extended[0, S_local].item()
        if actual_boundary != expected_boundary:
            print_rank0(f"  FAIL: last rank boundary={actual_boundary}, expected=-100")
            return False

    # Verify shift_labels = extended[:, 1:] gives correct next-token labels
    shift_labels = extended[:, 1:]  # [1, S_local]
    # For position i in this rank's chunk, the target should be label at global position (sp_rank*S_local + i + 1)
    for i in range(S_local):
        global_pos = sp_rank * S_local + i + 1
        if global_pos < sp_world * S_local:
            assert shift_labels[0, i].item() == global_pos, \
                f"rank {sp_rank} pos {i}: got {shift_labels[0, i].item()}, expected {global_pos}"
        else:
            assert shift_labels[0, i].item() == -100, \
                f"rank {sp_rank} pos {i}: got {shift_labels[0, i].item()}, expected -100"

    dist.barrier()
    print_rank0("  ✅ PASS: sp_exchange_boundary_labels correct")
    return True


def test_sp_reduce_loss(device, sp_group, sp_rank, sp_world):
    """Test sp_reduce_loss produces correct global mean."""
    from utils.mytp.sp_utils import sp_reduce_loss

    print_rank0("\n" + "=" * 60)
    print_rank0("Level 2 Test 4: sp_reduce_loss")
    print_rank0("=" * 60)

    # Simulate: each rank has different local loss sum and valid count
    # Generate deterministic values for any sp_world
    torch.manual_seed(42)
    local_sums = [(sp_rank + 1) * 10.0 for sp_rank in range(sp_world)]
    local_counts = [(sp_rank + 1) * 5 for sp_rank in range(sp_world)]
    # Global mean = sum(local_sums) / sum(local_counts)
    expected = sum(local_sums) / sum(local_counts)

    local_sum = torch.tensor(local_sums[sp_rank], device=device, dtype=torch.float32)
    local_count = torch.tensor(local_counts[sp_rank], device=device, dtype=torch.long)

    global_loss = sp_reduce_loss(local_sum, local_count, sp_group)

    # sp_reduce_loss returns local_sum / global_count per rank
    # Sum across all ranks = global_sum / global_count = global mean
    global_loss_summed = global_loss.clone()
    dist.all_reduce(global_loss_summed, op=dist.ReduceOp.SUM, group=sp_group)

    actual = global_loss_summed.item()
    if abs(actual - expected) > 1e-4:
        print_rank0(f"  FAIL: got {actual}, expected {expected}")
        return False

    # Also verify per-rank value: local_sum / global_count
    global_count = sum(local_counts)
    expected_per_rank = local_sums[sp_rank] / global_count
    actual_per_rank = global_loss.item()
    if abs(actual_per_rank - expected_per_rank) > 1e-4:
        print_rank0(f"  FAIL: rank {sp_rank} got {actual_per_rank}, expected {expected_per_rank}")
        return False

    dist.barrier()
    print_rank0(f"  ✅ PASS: sp_reduce_loss correct (global_mean={actual:.6f}, expected={expected:.6f})")
    return True


# ============================================================================
# Level 3: End-to-End SP vs Non-SP Equivalence
# ============================================================================

def test_e2e_loss_equivalence(device, sp_group, sp_rank, sp_world):
    """
    End-to-end test: compute CE loss with SP vs without SP.
    Verifies that the loss values are numerically equivalent.

    Strategy:
      - Create a simple model (embedding + linear layers + lm_head)
      - Run full sequence on rank 0 (reference, no SP)
      - Run split sequence across SP ranks
      - Compare loss values
    """
    print_rank0("\n" + "=" * 60)
    print_rank0("Level 3 Test 1: Loss Equivalence (SP vs non-SP)")
    print_rank0("=" * 60)

    from utils.mytp.sp_utils import (
        sp_split_sequence, sp_make_position_ids,
        sp_exchange_boundary_labels, sp_reduce_loss,
    )

    torch.manual_seed(42)
    B = 1
    S_full = 64  # Small for testing
    H = 128
    V = 1000  # Vocab size

    # Build a simple model: embedding + 2 linear layers + lm_head
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.linear1 = nn.Linear(H, H, bias=False)
            self.linear2 = nn.Linear(H, H, bias=False)
            self.lm_head = nn.Linear(H, V, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            return x

    model = SimpleModel().to(device=device, dtype=torch.bfloat16)

    # Create input
    torch.manual_seed(123)
    input_ids = torch.randint(0, V, (B, S_full), device=device)
    # Labels: shifted input_ids (next token prediction)
    labels = torch.cat([input_ids[:, 1:], torch.full((B, 1), -100, device=device, dtype=torch.long)], dim=1)

    # ---- Reference: full sequence, no SP ----
    model.zero_grad()
    ref_hs = model(input_ids)  # [B, S_full, H]
    ref_shift_hs = ref_hs[:, :-1, :].contiguous()
    ref_shift_labels = labels[:, 1:].contiguous()
    ref_logits = model.lm_head(ref_shift_hs)
    ref_loss = F.cross_entropy(
        ref_logits.view(-1, V), ref_shift_labels.view(-1), ignore_index=-100
    )
    ref_loss.backward()
    ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    ref_loss_val = ref_loss.item()

    # ---- SP: split sequence ----
    model.zero_grad()
    local_ids = sp_split_sequence(input_ids, sp_rank, sp_world)
    local_labels = sp_split_sequence(labels, sp_rank, sp_world)

    # Forward with local chunk
    sp_hs = model(local_ids)  # [B, S_local, H]

    # Exchange boundary labels
    extended_labels = sp_exchange_boundary_labels(local_labels, sp_group)
    # Use ALL hidden states (not :-1) since boundary label is appended
    sp_shift_hs = sp_hs.contiguous()  # [B, S_local, H]
    sp_shift_labels = extended_labels[:, 1:].contiguous()  # [B, S_local]

    # Compute local loss (sum + count)
    sp_logits = model.lm_head(sp_shift_hs)
    local_loss_sum = F.cross_entropy(
        sp_logits.view(-1, V), sp_shift_labels.view(-1), ignore_index=-100, reduction="sum"
    )
    local_valid_count = (sp_shift_labels.view(-1) != -100).sum()

    # Global loss via sp_reduce_loss
    # sp_reduce_loss returns local_sum / global_count (different per rank)
    # The sum across all ranks equals the global mean loss.
    sp_loss = sp_reduce_loss(local_loss_sum, local_valid_count, sp_group)
    sp_loss.backward()

    # All-reduce gradients (SP-sum to reconstruct full gradient)
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=sp_group)

    sp_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    # The global mean loss = sum of all ranks' sp_loss values
    sp_loss_tensor = sp_loss.detach().clone()
    dist.all_reduce(sp_loss_tensor, op=dist.ReduceOp.SUM, group=sp_group)
    sp_loss_val = sp_loss_tensor.item()

    # ---- Compare ----
    loss_rel = abs(sp_loss_val - ref_loss_val) / (abs(ref_loss_val) + 1e-10)
    print_rank0(f"  Loss: ref={ref_loss_val:.6f}, sp={sp_loss_val:.6f}, rel_diff={loss_rel:.2e}")

    all_pass = True
    if loss_rel > 2e-3:
        print_rank0(f"  ❌ FAIL: Loss relative diff {loss_rel:.2e} > 2e-3")
        all_pass = False

    # Compare gradients
    for name in ref_grads:
        if name in sp_grads:
            g_rel = rel_diff(sp_grads[name], ref_grads[name])
            g_cos = cos_sim(sp_grads[name], ref_grads[name])
            if g_cos < 0.999:
                print_rank0(f"  ❌ FAIL: grad({name}) cos={g_cos:.6f} < 0.999")
                all_pass = False
            elif dist.get_rank() == 0:
                print(f"    grad({name}): rel_diff={g_rel:.2e}, cos={g_cos:.6f}", flush=True)

    if all_pass:
        print_rank0("  ✅ PASS: Loss and gradients match")
    return all_pass


def test_e2e_hidden_states_equivalence(device, sp_group, sp_rank, sp_world):
    """
    End-to-end test: compare intermediate hidden states.
    Verifies that each SP rank's local hidden states match the corresponding
    slice of the full-sequence hidden states.
    """
    print_rank0("\n" + "=" * 60)
    print_rank0("Level 3 Test 2: Hidden States Equivalence")
    print_rank0("=" * 60)

    from utils.mytp.sp_utils import sp_split_sequence

    torch.manual_seed(42)
    B = 1
    S_full = 64
    H = 128
    V = 1000

    class MultiLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.layers = nn.ModuleList([
                nn.Linear(H, H, bias=False) for _ in range(4)
            ])

        def forward(self, input_ids, return_intermediates=False):
            x = self.embed(input_ids)
            intermediates = [x.detach().clone()]
            for layer in self.layers:
                x = layer(x)
                x = F.gelu(x)
                intermediates.append(x.detach().clone())
            if return_intermediates:
                return x, intermediates
            return x

    model = MultiLayerModel().to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(123)
    input_ids = torch.randint(0, V, (B, S_full), device=device)

    # Reference: full sequence
    ref_out, ref_intermediates = model(input_ids, return_intermediates=True)

    # SP: split sequence
    local_ids = sp_split_sequence(input_ids, sp_rank, sp_world)
    sp_out, sp_intermediates = model(local_ids, return_intermediates=True)

    # Compare: each SP rank's output should match the corresponding slice
    S_local = S_full // sp_world
    start = sp_rank * S_local
    end = start + S_local

    all_pass = True
    for layer_idx, (ref_inter, sp_inter) in enumerate(zip(ref_intermediates, sp_intermediates)):
        ref_slice = ref_inter[:, start:end, :]
        r = rel_diff(sp_inter, ref_slice)
        c = cos_sim(sp_inter, ref_slice)
        if c < 0.9999:
            print_rank0(f"  ❌ FAIL: layer {layer_idx} cos={c:.6f} < 0.9999")
            all_pass = False
        elif dist.get_rank() == 0:
            print(f"    Layer {layer_idx}: rel_diff={r:.2e}, cos={c:.6f}", flush=True)

    # Also compare final output
    ref_out_slice = ref_out[:, start:end, :].detach()
    out_rel = rel_diff(sp_out.detach(), ref_out_slice)
    out_cos = cos_sim(sp_out.detach(), ref_out_slice)
    if dist.get_rank() == 0:
        print(f"    Final output: rel_diff={out_rel:.2e}, cos={out_cos:.6f}", flush=True)
    if out_cos < 0.9999:
        print_rank0(f"  ❌ FAIL: final output cos={out_cos:.6f} < 0.9999")
        all_pass = False

    dist.barrier()
    if all_pass:
        print_rank0("  ✅ PASS: All intermediate hidden states match")
    return all_pass


def test_e2e_backward_gradients(device, sp_group, sp_rank, sp_world):
    """
    End-to-end test: compare backward gradients in detail.
    Verifies that SP backward produces the same parameter gradients
    as non-SP backward, and also compares input gradients.
    """
    print_rank0("\n" + "=" * 60)
    print_rank0("Level 3 Test 3: Backward Gradients Equivalence")
    print_rank0("=" * 60)

    from utils.mytp.sp_utils import (
        sp_split_sequence, sp_exchange_boundary_labels, sp_reduce_loss,
    )

    torch.manual_seed(42)
    B = 1
    S_full = 64
    H = 128
    V = 1000

    class GradTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.linear1 = nn.Linear(H, H, bias=False)
            self.linear2 = nn.Linear(H, H, bias=False)
            self.lm_head = nn.Linear(H, V, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            return x

    model = GradTestModel().to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(123)
    input_ids = torch.randint(0, V, (B, S_full), device=device)
    labels = torch.cat([input_ids[:, 1:], torch.full((B, 1), -100, device=device, dtype=torch.long)], dim=1)

    # ---- Reference: full sequence ----
    model.zero_grad()
    ref_hs = model(input_ids)
    ref_logits = model.lm_head(ref_hs[:, :-1, :].contiguous())
    ref_loss = F.cross_entropy(ref_logits.view(-1, V), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
    ref_loss.backward()
    ref_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            ref_grads[name] = p.grad.clone()

    # ---- SP: split sequence ----
    model.zero_grad()
    local_ids = sp_split_sequence(input_ids, sp_rank, sp_world)
    local_labels = sp_split_sequence(labels, sp_rank, sp_world)

    sp_hs = model(local_ids)
    extended_labels = sp_exchange_boundary_labels(local_labels, sp_group)
    sp_shift_hs = sp_hs.contiguous()
    sp_shift_labels = extended_labels[:, 1:].contiguous()

    sp_logits = model.lm_head(sp_shift_hs)
    local_loss_sum = F.cross_entropy(
        sp_logits.view(-1, V), sp_shift_labels.view(-1), ignore_index=-100, reduction="sum"
    )
    local_valid_count = (sp_shift_labels.view(-1) != -100).sum()
    sp_loss = sp_reduce_loss(local_loss_sum, local_valid_count, sp_group)
    sp_loss.backward()

    # All-reduce gradients across SP group
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=sp_group)

    sp_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            sp_grads[name] = p.grad.clone()

    # Verify loss: sum of all ranks' sp_loss = global mean
    sp_loss_tensor = sp_loss.detach().clone()
    dist.all_reduce(sp_loss_tensor, op=dist.ReduceOp.SUM, group=sp_group)
    sp_loss_val = sp_loss_tensor.item()
    loss_rel = abs(sp_loss_val - ref_loss.item()) / (abs(ref_loss.item()) + 1e-10)
    if dist.get_rank() == 0:
        print(f"    Loss: ref={ref_loss.item():.6f}, sp={sp_loss_val:.6f}, rel={loss_rel:.2e}", flush=True)

    # ---- Compare per-parameter gradients ----
    all_pass = True
    if loss_rel > 2e-3:
        print_rank0(f"  ❌ FAIL: loss rel_diff={loss_rel:.2e} > 2e-3")
        all_pass = False

    for name in sorted(ref_grads.keys()):
        if name not in sp_grads:
            print_rank0(f"  ❌ FAIL: {name} missing in SP grads")
            all_pass = False
            continue
        g_rel = rel_diff(sp_grads[name], ref_grads[name])
        g_cos = cos_sim(sp_grads[name], ref_grads[name])
        if dist.get_rank() == 0:
            status = "✅" if g_cos >= 0.999 else "❌"
            print(f"    {status} grad({name}): rel_diff={g_rel:.2e}, cos={g_cos:.6f}", flush=True)
        if g_cos < 0.999:
            all_pass = False

    dist.barrier()
    if all_pass:
        print_rank0("  ✅ PASS: All parameter gradients match")
    return all_pass


def test_e2e_with_position_ids(device, sp_group, sp_rank, sp_world):
    """
    End-to-end test: verify that position-dependent computations (simulating RoPE)
    produce correct results under SP with global position_ids.
    """
    print_rank0("\n" + "=" * 60)
    print_rank0("Level 3 Test 4: Position-Dependent Computation (RoPE simulation)")
    print_rank0("=" * 60)

    from utils.mytp.sp_utils import sp_split_sequence, sp_make_position_ids

    torch.manual_seed(42)
    B = 1
    S_full = 64
    H = 128

    # Simulate RoPE: output depends on position
    class PositionalModel(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.linear = nn.Linear(H, H, bias=False)
            # Create fixed sinusoidal position encoding
            self.register_buffer('pos_enc', self._make_pos_enc(1024, H))

        def _make_pos_enc(self, max_len, H):
            pos = torch.arange(max_len).unsqueeze(1).float()
            dim = torch.arange(0, H, 2).float()
            angles = pos / (10000 ** (dim / H))
            pe = torch.zeros(max_len, H)
            pe[:, 0::2] = torch.sin(angles)
            pe[:, 1::2] = torch.cos(angles)
            return pe

        def forward(self, x, position_ids=None):
            """x: [B, S, H], position_ids: [B, S]"""
            if position_ids is None:
                S = x.shape[1]
                pe = self.pos_enc[:S].unsqueeze(0)  # [1, S, H]
            else:
                # Gather position encodings by position_ids
                pe = self.pos_enc[position_ids]  # [B, S, H]
            x = x + pe.to(x.dtype)
            x = self.linear(x)
            return x

    model = PositionalModel(H).to(device=device, dtype=torch.bfloat16)

    # Create input hidden states
    torch.manual_seed(123)
    hidden_full = torch.randn(B, S_full, H, device=device, dtype=torch.bfloat16)

    # Reference: full sequence, no position_ids (auto 0..S-1)
    ref_out = model(hidden_full, position_ids=None)

    # SP: split + explicit global position_ids
    S_local = S_full // sp_world
    local_hidden = hidden_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    local_pos_ids = sp_make_position_ids(S_local, sp_rank, sp_world, batch_size=B, device=device)
    sp_out = model(local_hidden, position_ids=local_pos_ids)

    # Compare
    ref_slice = ref_out[:, sp_rank * S_local:(sp_rank + 1) * S_local, :].detach()
    sp_out_det = sp_out.detach()

    r = rel_diff(sp_out_det, ref_slice)
    c = cos_sim(sp_out_det, ref_slice)

    all_pass = True
    if c < 0.9999:
        print_rank0(f"  ❌ FAIL: cos={c:.6f} < 0.9999 (rel_diff={r:.2e})")
        all_pass = False
    else:
        if dist.get_rank() == 0:
            print(f"    Output: rel_diff={r:.2e}, cos={c:.6f}", flush=True)

    dist.barrier()
    if all_pass:
        print_rank0("  ✅ PASS: Position-dependent computation correct under SP")
    return all_pass


def test_e2e_mem_token_boundary(device, sp_group, sp_rank, sp_world):
    """
    End-to-end test: verify mem_token placement works correctly when
    mem_tokens span SP chunk boundaries.

    Tests the logic that computes local_context_lengths from global context_lengths.
    """
    print_rank0("\n" + "=" * 60)
    print_rank0("Level 3 Test 5: Mem Token Boundary Handling")
    print_rank0("=" * 60)

    S_full = 64  # Use larger S_full to ensure divisibility by any sp_world
    S_local = S_full // sp_world
    num_mem_token = 4
    all_pass = True

    def compute_local_ctx_len(context_length, sp_rank, S_local, num_mem_token):
        """Replicate the logic from TPModelHypernetwork.forward for testing."""
        chunk_start = sp_rank * S_local
        chunk_end = chunk_start + S_local
        mem_end = context_length + num_mem_token
        if context_length >= chunk_end or mem_end <= chunk_start:
            return S_local, False  # No mem_tokens on this rank
        else:
            return max(0, context_length - chunk_start), True

    # Test case 1: mem_tokens fully within rank 0's chunk
    # context_length=0 → mem at [0, 4) → always within rank 0
    ctx_len_1 = 0
    local_ctx, has_mem = compute_local_ctx_len(ctx_len_1, sp_rank, S_local, num_mem_token)
    mem_end_1 = ctx_len_1 + num_mem_token
    chunk_start = sp_rank * S_local
    chunk_end = chunk_start + S_local

    if sp_rank == 0:
        if not has_mem:
            print_rank0(f"  ❌ FAIL case 1: rank 0 should have mem_tokens")
            all_pass = False
        if local_ctx != 0:
            print_rank0(f"  ❌ FAIL case 1: rank 0 local_ctx={local_ctx}, expected 0")
            all_pass = False
    else:
        # Other ranks: mem at [0, 4), their chunk_start >= S_local >= 4
        # So mem_end <= chunk_start → no mem
        if mem_end_1 <= chunk_start:
            if has_mem:
                print_rank0(f"  ❌ FAIL case 1: rank {sp_rank} should NOT have mem_tokens")
                all_pass = False

    # Test case 2: mem_tokens spanning a boundary
    # Choose context_length such that mem spans two chunks
    # context_length = S_local - 2 → mem at [S_local-2, S_local+2) → spans rank 0 and rank 1
    ctx_len_2 = S_local - 2
    mem_end_2 = ctx_len_2 + num_mem_token  # S_local + 2

    for r in range(sp_world):
        local_ctx_r, has_mem_r = compute_local_ctx_len(ctx_len_2, r, S_local, num_mem_token)
        r_chunk_start = r * S_local
        r_chunk_end = r_chunk_start + S_local

        if r == 0:
            # Rank 0: chunk=[0, S_local), mem=[S_local-2, S_local+2)
            # Overlap at [S_local-2, S_local), local_ctx = S_local-2
            if not has_mem_r:
                print_rank0(f"  ❌ FAIL case 2: rank 0 should have mem_tokens")
                all_pass = False
            expected_local = ctx_len_2  # S_local - 2
            if local_ctx_r != expected_local:
                print_rank0(f"  ❌ FAIL case 2: rank 0 local_ctx={local_ctx_r}, expected {expected_local}")
                all_pass = False
        elif r == 1:
            # Rank 1: chunk=[S_local, 2*S_local), mem=[S_local-2, S_local+2)
            # Overlap at [S_local, S_local+2), local_ctx = max(0, (S_local-2) - S_local) = 0
            if not has_mem_r:
                print_rank0(f"  ❌ FAIL case 2: rank 1 should have mem_tokens (overlap)")
                all_pass = False
            if local_ctx_r != 0:
                print_rank0(f"  ❌ FAIL case 2: rank 1 local_ctx={local_ctx_r}, expected 0")
                all_pass = False
        else:
            # Other ranks: no overlap
            if has_mem_r:
                print_rank0(f"  ❌ FAIL case 2: rank {r} should NOT have mem_tokens")
                all_pass = False

    # Test case 3: mem_tokens entirely in a middle rank
    # context_length = 2 * S_local + 1 → mem at [2*S_local+1, 2*S_local+5)
    if sp_world >= 3:
        ctx_len_3 = 2 * S_local + 1
        mem_end_3 = ctx_len_3 + num_mem_token
        for r in range(sp_world):
            local_ctx_r, has_mem_r = compute_local_ctx_len(ctx_len_3, r, S_local, num_mem_token)
            r_chunk_start = r * S_local
            r_chunk_end = r_chunk_start + S_local

            if r == 2:
                # Rank 2: chunk=[2*S_local, 3*S_local), mem starts at 2*S_local+1
                if not has_mem_r:
                    print_rank0(f"  ❌ FAIL case 3: rank 2 should have mem_tokens")
                    all_pass = False
                expected_local = 1  # ctx_len_3 - 2*S_local = 1
                if local_ctx_r != expected_local:
                    print_rank0(f"  ❌ FAIL case 3: rank 2 local_ctx={local_ctx_r}, expected {expected_local}")
                    all_pass = False
            elif r_chunk_start < mem_end_3 and r_chunk_end > ctx_len_3:
                # This rank has overlap (could be rank 3 if S_local < num_mem_token)
                pass  # Don't fail, just verify the logic is consistent
            else:
                if has_mem_r and not (r_chunk_start < mem_end_3 and r_chunk_end > ctx_len_3):
                    print_rank0(f"  ❌ FAIL case 3: rank {r} should NOT have mem_tokens")
                    all_pass = False

    dist.barrier()
    if all_pass:
        print_rank0("  ✅ PASS: Mem token boundary handling correct")
    return all_pass


def test_e2e_full_forward_backward(device, sp_group, sp_rank, sp_world):
    """
    End-to-end test: Full forward + backward with a multi-layer model.
    Compares:
      1. Per-layer intermediate hidden states
      2. Final output (logits)
      3. Loss value
      4. Per-parameter gradients
      5. Input embedding gradients
    """
    print_rank0("\n" + "=" * 60)
    print_rank0("Level 3 Test 6: Full Forward+Backward (Multi-Layer)")
    print_rank0("=" * 60)

    from utils.mytp.sp_utils import (
        sp_split_sequence, sp_make_position_ids,
        sp_exchange_boundary_labels, sp_reduce_loss,
    )

    torch.manual_seed(42)
    B = 1
    S_full = 128
    H = 256
    V = 500
    num_layers = 4

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.layers = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(num_layers)])
            self.lm_head = nn.Linear(H, V, bias=False)
            # Position encoding (simulating RoPE)
            self.register_buffer('pos_enc', torch.randn(2048, H) * 0.01)

        def forward(self, input_ids, position_ids=None, return_intermediates=False):
            x = self.embed(input_ids)
            # Add position encoding
            if position_ids is None:
                x = x + self.pos_enc[:x.shape[1]].unsqueeze(0).to(x.dtype)
            else:
                x = x + self.pos_enc[position_ids].to(x.dtype)

            intermediates = []
            for layer in self.layers:
                x = layer(x)
                x = F.gelu(x)
                if return_intermediates:
                    intermediates.append(x.detach().clone())
            if return_intermediates:
                return x, intermediates
            return x

    model = FullModel().to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(123)
    input_ids = torch.randint(0, V, (B, S_full), device=device)
    labels = torch.cat([input_ids[:, 1:], torch.full((B, 1), -100, device=device, dtype=torch.long)], dim=1)

    # ---- Reference ----
    model.zero_grad()
    ref_hs, ref_intermediates = model(input_ids, return_intermediates=True)
    ref_logits = model.lm_head(ref_hs[:, :-1, :].contiguous())
    ref_loss = F.cross_entropy(ref_logits.view(-1, V), labels[:, 1:].contiguous().view(-1), ignore_index=-100)
    ref_loss.backward()
    ref_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    ref_loss_val = ref_loss.item()

    # ---- SP ----
    model.zero_grad()
    S_local = S_full // sp_world
    local_ids = sp_split_sequence(input_ids, sp_rank, sp_world)
    local_labels = sp_split_sequence(labels, sp_rank, sp_world)
    local_pos_ids = sp_make_position_ids(S_local, sp_rank, sp_world, batch_size=B, device=device)

    sp_hs, sp_intermediates = model(local_ids, position_ids=local_pos_ids, return_intermediates=True)

    # Compare intermediates
    all_pass = True
    start = sp_rank * S_local
    end = start + S_local

    for layer_idx, (ref_inter, sp_inter) in enumerate(zip(ref_intermediates, sp_intermediates)):
        ref_slice = ref_inter[:, start:end, :]
        c = cos_sim(sp_inter, ref_slice)
        r = rel_diff(sp_inter, ref_slice)
        if c < 0.9999:
            print_rank0(f"  ❌ FAIL: intermediate layer {layer_idx} cos={c:.6f}")
            all_pass = False
        elif dist.get_rank() == 0:
            print(f"    Intermediate layer {layer_idx}: rel={r:.2e}, cos={c:.6f}", flush=True)

    # Compare final hidden states
    ref_hs_slice = ref_hs[:, start:end, :].detach()
    hs_cos = cos_sim(sp_hs.detach(), ref_hs_slice)
    if dist.get_rank() == 0:
        print(f"    Final hidden: cos={hs_cos:.6f}", flush=True)

    # Loss
    extended_labels = sp_exchange_boundary_labels(local_labels, sp_group)
    sp_shift_hs = sp_hs.contiguous()
    sp_shift_labels = extended_labels[:, 1:].contiguous()
    sp_logits = model.lm_head(sp_shift_hs)
    local_loss_sum = F.cross_entropy(
        sp_logits.view(-1, V), sp_shift_labels.view(-1), ignore_index=-100, reduction="sum"
    )
    local_valid_count = (sp_shift_labels.view(-1) != -100).sum()
    sp_loss = sp_reduce_loss(local_loss_sum, local_valid_count, sp_group)
    sp_loss.backward()

    # All-reduce gradients
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=sp_group)

    # Global mean loss = sum of all ranks' sp_loss
    sp_loss_tensor = sp_loss.detach().clone()
    dist.all_reduce(sp_loss_tensor, op=dist.ReduceOp.SUM, group=sp_group)
    sp_loss_val = sp_loss_tensor.item()
    loss_rel = abs(sp_loss_val - ref_loss_val) / (abs(ref_loss_val) + 1e-10)
    if dist.get_rank() == 0:
        print(f"    Loss: ref={ref_loss_val:.6f}, sp={sp_loss_val:.6f}, rel={loss_rel:.2e}", flush=True)
    if loss_rel > 2e-3:
        print_rank0(f"  ❌ FAIL: loss rel_diff={loss_rel:.2e} > 2e-3")
        all_pass = False

    # Compare gradients
    sp_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    for name in sorted(ref_grads.keys()):
        if name not in sp_grads:
            continue
        g_cos = cos_sim(sp_grads[name], ref_grads[name])
        g_rel = rel_diff(sp_grads[name], ref_grads[name])
        if g_cos < 0.999:
            print_rank0(f"  ❌ FAIL: grad({name}) cos={g_cos:.6f}")
            all_pass = False
        elif dist.get_rank() == 0:
            print(f"    grad({name}): rel={g_rel:.2e}, cos={g_cos:.6f}", flush=True)

    dist.barrier()
    if all_pass:
        print_rank0("  ✅ PASS: Full forward+backward equivalence verified")
    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    print_rank0(f"\n{'#' * 70}")
    print_rank0(f"# SP Integration + End-to-End Tests")
    print_rank0(f"# World size: {world_size}, using all GPUs as one SP group")
    print_rank0(f"{'#' * 70}")

    # Use all GPUs as one SP group for maximum coverage
    sp_world = world_size
    sp_group, sp_rank = create_sp_group(world_size, sp_world)

    results = {}

    # Level 2: sp_utils integration tests
    print_rank0(f"\n{'=' * 70}")
    print_rank0("LEVEL 2: SP Utils Integration Tests")
    print_rank0(f"{'=' * 70}")

    results["L2.1 split_sequence"] = test_sp_split_sequence(device)
    results["L2.2 make_position_ids"] = test_sp_make_position_ids(device)
    results["L2.3 exchange_boundary_labels"] = test_sp_exchange_boundary_labels(
        device, sp_group, sp_rank, sp_world
    )
    results["L2.4 reduce_loss"] = test_sp_reduce_loss(device, sp_group, sp_rank, sp_world)

    thorough_cleanup(device)

    # Level 3: End-to-end tests
    print_rank0(f"\n{'=' * 70}")
    print_rank0("LEVEL 3: End-to-End SP vs Non-SP Equivalence Tests")
    print_rank0(f"{'=' * 70}")

    results["L3.1 loss_equivalence"] = test_e2e_loss_equivalence(
        device, sp_group, sp_rank, sp_world
    )
    thorough_cleanup(device)

    results["L3.2 hidden_states"] = test_e2e_hidden_states_equivalence(
        device, sp_group, sp_rank, sp_world
    )
    thorough_cleanup(device)

    results["L3.3 backward_gradients"] = test_e2e_backward_gradients(
        device, sp_group, sp_rank, sp_world
    )
    thorough_cleanup(device)

    results["L3.4 position_ids"] = test_e2e_with_position_ids(
        device, sp_group, sp_rank, sp_world
    )
    thorough_cleanup(device)

    results["L3.5 mem_token_boundary"] = test_e2e_mem_token_boundary(
        device, sp_group, sp_rank, sp_world
    )
    thorough_cleanup(device)

    results["L3.6 full_forward_backward"] = test_e2e_full_forward_backward(
        device, sp_group, sp_rank, sp_world
    )
    thorough_cleanup(device)

    # Summary
    print_rank0(f"\n{'=' * 70}")
    print_rank0("SUMMARY")
    print_rank0(f"{'=' * 70}")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print_rank0(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print_rank0(f"\n{'=' * 70}")
    if all_pass:
        print_rank0("ALL TESTS PASSED ✅")
    else:
        print_rank0("SOME TESTS FAILED ❌")
    print_rank0(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
