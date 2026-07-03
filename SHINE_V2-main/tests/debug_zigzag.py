"""
Final correctness test for zigzag ring attention.
Tests forward correctness and self-consistency.
"""
import os
os.environ["PRECISION_ENHENCEMENT_FA2"] = "0"

import sys, torch, time
import torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.mytp.ring_attention import (
    _zigzag_indices, zigzag_split, zigzag_unsplit, sp_flash_attention
)
from flash_attn import flash_attn_func

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

SP = world_size
sp_group = dist.group.WORLD


def run_ring_attn(q_full, k_full, v_full, SP, sp_group, rank):
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


all_tests_pass = True

# ============================================================
# Test 1: Forward correctness (D=128)
# ============================================================
if rank == 0:
    print(f"\n{'='*60}")
    print(f"Test 1: Forward correctness D=128 (SP={SP})")
    print(f"{'='*60}")

for S in [256, 512, 1024, 2048, 4096]:
    torch.manual_seed(42)
    q = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)

    out_ref = flash_attn_func(q, k, v, causal=True)
    out_sp = run_ring_attn(q, k, v, SP, sp_group, rank)

    max_diff = (out_sp - out_ref).abs().max().item()
    mean_diff = (out_sp - out_ref).abs().mean().item()
    passed = max_diff < 0.01
    all_tests_pass = all_tests_pass and passed
    if rank == 0:
        print(f"  S={S}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {'✅' if passed else '❌'}")
    dist.barrier()

# ============================================================
# Test 2: Forward correctness (D=64)
# ============================================================
if rank == 0:
    print(f"\n{'='*60}")
    print(f"Test 2: Forward correctness D=64 (SP={SP})")
    print(f"{'='*60}")

for S in [512, 1024, 2048, 4096]:
    torch.manual_seed(42)
    q = torch.randn(1, S, 8, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, S, 8, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, S, 8, 64, dtype=torch.bfloat16, device=device)

    out_ref = flash_attn_func(q, k, v, causal=True)
    out_sp = run_ring_attn(q, k, v, SP, sp_group, rank)

    max_diff = (out_sp - out_ref).abs().max().item()
    passed = max_diff < 0.01
    all_tests_pass = all_tests_pass and passed
    if rank == 0:
        print(f"  S={S}: max_diff={max_diff:.6e} {'✅' if passed else '❌'}")
    dist.barrier()

# ============================================================
# Test 3: GQA (D=128)
# ============================================================
if rank == 0:
    print(f"\n{'='*60}")
    print(f"Test 3: GQA D=128 (SP={SP})")
    print(f"{'='*60}")

for S, Hq, Hkv in [(1024, 14, 2), (2048, 28, 4), (4096, 14, 2)]:
    torch.manual_seed(42)
    q = torch.randn(1, S, Hq, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, S, Hkv, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, S, Hkv, 128, dtype=torch.bfloat16, device=device)

    out_ref = flash_attn_func(q, k, v, causal=True)
    out_sp = run_ring_attn(q, k, v, SP, sp_group, rank)

    max_diff = (out_sp - out_ref).abs().max().item()
    passed = max_diff < 0.01
    all_tests_pass = all_tests_pass and passed
    if rank == 0:
        print(f"  S={S}, Hq={Hq}, Hkv={Hkv}: max_diff={max_diff:.6e} {'✅' if passed else '❌'}")
    dist.barrier()

# ============================================================
# Test 4: Self-consistency (D=128)
# ============================================================
if rank == 0:
    print(f"\n{'='*60}")
    print(f"Test 4: Self-consistency D=128 (SP={SP})")
    print(f"{'='*60}")

for S in [512, 1024, 2048]:
    torch.manual_seed(42)
    q = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)

    out1 = run_ring_attn(q, k, v, SP, sp_group, rank)
    out2 = run_ring_attn(q, k, v, SP, sp_group, rank)

    max_diff = (out1 - out2).abs().max().item()
    passed = max_diff == 0.0
    all_tests_pass = all_tests_pass and passed
    if rank == 0:
        print(f"  S={S}: diff={max_diff:.6e} {'✅' if passed else '❌'}")
    dist.barrier()

# ============================================================
# Test 5: Performance
# ============================================================
if rank == 0:
    print(f"\n{'='*60}")
    print(f"Test 5: Performance (SP={SP})")
    print(f"{'='*60}")

S, H, D = 8192, 14, 128
torch.manual_seed(42)
q = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
k = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
v = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)

indices = _zigzag_indices(S, SP)
q_local = q[:, indices[rank].to(device)]
k_local = k[:, indices[rank].to(device)]
v_local = v[:, indices[rank].to(device)]

for _ in range(3):
    sp_flash_attention(q_local, k_local, v_local, sp_group=sp_group, causal=True)
torch.cuda.synchronize()

N = 20
start = time.perf_counter()
for _ in range(N):
    sp_flash_attention(q_local, k_local, v_local, sp_group=sp_group, causal=True)
torch.cuda.synchronize()
ring_time = (time.perf_counter() - start) / N * 1000

if rank == 0:
    for _ in range(3):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    single_time = (time.perf_counter() - start) / N * 1000
    print(f"  S={S}, H={H}, D={D}")
    print(f"  Ring attention (SP={SP}): {ring_time:.2f} ms")
    print(f"  Single GPU flash_attn:    {single_time:.2f} ms")
    print(f"  Speedup: {single_time/ring_time:.2f}x")

# ============================================================
# Summary
# ============================================================
if rank == 0:
    print(f"\n{'='*60}")
    print(f"{'✅ ALL TESTS PASSED' if all_tests_pass else '❌ SOME TESTS FAILED'}")
    print(f"{'='*60}\n")

dist.barrier()
dist.destroy_process_group()
