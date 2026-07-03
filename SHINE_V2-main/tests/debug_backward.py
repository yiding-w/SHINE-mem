"""Debug backward for contiguous ring attention only (Scheme A)."""
import os
os.environ["PRECISION_ENHENCEMENT_FA2"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from utils.mytp.ring_attention import sp_flash_attention_contiguous, sp_flash_attention_alltoall_zigzag

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
sp_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

S, H, D = 512, 8, 128
chunk_size = S // sp_size

if rank == 0:
    print(f"Testing backward S={S}, H={H}, D={D}, SP={sp_size}", flush=True)

# Reference
torch.manual_seed(42)
q_ref = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
k_ref = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
v_ref = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
out_ref = flash_attn_func(q_ref, k_ref, v_ref, causal=True)
out_ref.sum().backward()

if rank == 0:
    print("Reference backward done", flush=True)

# SP Scheme A
torch.manual_seed(42)
q_full = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
k_full = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)
v_full = torch.randn(1, S, H, D, dtype=torch.bfloat16, device=device)

q_a = q_full[:, rank*chunk_size:(rank+1)*chunk_size].clone().requires_grad_(True)
k_a = k_full[:, rank*chunk_size:(rank+1)*chunk_size].clone().requires_grad_(True)
v_a = v_full[:, rank*chunk_size:(rank+1)*chunk_size].clone().requires_grad_(True)

out_a = sp_flash_attention_contiguous(q_a, k_a, v_a, sp_group=dist.group.WORLD, causal=True)
out_a.sum().backward()

if rank == 0:
    print("Scheme A backward done", flush=True)

# SP Scheme B
q_b = q_full[:, rank*chunk_size:(rank+1)*chunk_size].clone().requires_grad_(True)
k_b = k_full[:, rank*chunk_size:(rank+1)*chunk_size].clone().requires_grad_(True)
v_b = v_full[:, rank*chunk_size:(rank+1)*chunk_size].clone().requires_grad_(True)

out_b = sp_flash_attention_alltoall_zigzag(q_b, k_b, v_b, sp_group=dist.group.WORLD, causal=True)
out_b.sum().backward()

if rank == 0:
    print("Scheme B backward done", flush=True)

# Gather
def gather_grad(grad):
    gathered = [torch.empty_like(grad) for _ in range(sp_size)]
    dist.all_gather(gathered, grad.contiguous(), group=dist.group.WORLD)
    return torch.cat(gathered, dim=1)

dq_a_full = gather_grad(q_a.grad)
dk_a_full = gather_grad(k_a.grad)
dv_a_full = gather_grad(v_a.grad)

dq_b_full = gather_grad(q_b.grad)
dk_b_full = gather_grad(k_b.grad)
dv_b_full = gather_grad(v_b.grad)

if rank == 0:
    print(f"\nScheme A (contiguous ring):")
    print(f"  dQ diff: {(dq_a_full - q_ref.grad).abs().max().item():.4e}")
    print(f"  dK diff: {(dk_a_full - k_ref.grad).abs().max().item():.4e}")
    print(f"  dV diff: {(dv_a_full - v_ref.grad).abs().max().item():.4e}")
    
    print(f"\nScheme B (all-to-all + zigzag):")
    print(f"  dQ diff: {(dq_b_full - q_ref.grad).abs().max().item():.4e}")
    print(f"  dK diff: {(dk_b_full - k_ref.grad).abs().max().item():.4e}")
    print(f"  dV diff: {(dv_b_full - v_ref.grad).abs().max().item():.4e}")

dist.barrier()
dist.destroy_process_group()
