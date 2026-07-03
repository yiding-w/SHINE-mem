"""Quick test for all-to-all + zigzag ring attention (Scheme B) only."""
import os
os.environ["PRECISION_ENHENCEMENT_FA2"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from utils.mytp.ring_attention import sp_flash_attention_alltoall_zigzag

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

if rank == 0:
    print(f"Testing Scheme B (all-to-all + zigzag) with SP={world_size}")

S = 512
sp_size = world_size
chunk_size = S // sp_size

torch.manual_seed(42)
q_full = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
k_full = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)
v_full = torch.randn(1, S, 8, 128, dtype=torch.bfloat16, device=device)

out_ref = flash_attn_func(q_full, k_full, v_full, causal=True)

q_local = q_full[:, rank*chunk_size:(rank+1)*chunk_size].contiguous()
k_local = k_full[:, rank*chunk_size:(rank+1)*chunk_size].contiguous()
v_local = v_full[:, rank*chunk_size:(rank+1)*chunk_size].contiguous()

if rank == 0:
    print(f"  Running sp_flash_attention_alltoall_zigzag...")

out_b_local = sp_flash_attention_alltoall_zigzag(
    q_local, k_local, v_local, sp_group=dist.group.WORLD, causal=True
)

if rank == 0:
    print(f"  Forward done, gathering...")

out_b_gathered = [torch.empty_like(out_b_local) for _ in range(sp_size)]
dist.all_gather(out_b_gathered, out_b_local, group=dist.group.WORLD)
out_b_full = torch.cat(out_b_gathered, dim=1)

diff = (out_b_full - out_ref).abs().max().item()
if rank == 0:
    print(f"  Scheme B forward S={S}: max_diff={diff:.6e} {'PASS' if diff < 0.02 else 'FAIL'}")

dist.barrier()
dist.destroy_process_group()
