"""Minimal test: just verify build_sp_cp_context + fla_causal_conv1d work."""
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Started: world={world_size}", flush=True)

    # Create SP group of size=world_size (all ranks in one SP group)
    sp_group = dist.new_group(list(range(world_size)))
    sp_world = world_size
    sp_rank = rank

    if rank == 0:
        print(f"SP group created, sp_world={sp_world}", flush=True)

    # Test build_sp_cp_context
    from utils.mytp.fla_cp import build_sp_cp_context
    S_local = 256
    S_full = S_local * sp_world
    D, kernel_size = 64, 4

    cp_ctx = build_sp_cp_context(S_local, sp_group, conv1d_kernel_size=kernel_size, device=device)
    if rank == 0:
        print(f"cp_context built: type={type(cp_ctx).__name__}", flush=True)
        print(f"  is_first_rank={cp_ctx.is_first_rank}", flush=True)

    from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d

    # Generate full input on ALL ranks (same seed => same data)
    torch.manual_seed(42)
    x_full = torch.randn(1, S_full, D, dtype=torch.bfloat16, device=device)
    weight = torch.randn(D, kernel_size, dtype=torch.bfloat16, device=device)

    # Reference: full sequence computation (no CP)
    ref_out, _ = fla_causal_conv1d(x=x_full.clone(), weight=weight, activation='silu')

    # SP: each rank gets its local chunk from x_full
    x_local = x_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous().clone().requires_grad_(True)

    if rank == 0:
        print(f"Running fla_causal_conv1d with cp_context...", flush=True)

    out_local, _ = fla_causal_conv1d(x=x_local, weight=weight, activation='silu', cp_context=cp_ctx)

    if rank == 0:
        print(f"Conv1d output shape: {out_local.shape}", flush=True)

    # Backward
    torch.manual_seed(123)
    grad_full = torch.randn(1, S_full, D, dtype=torch.bfloat16, device=device)
    grad_local = grad_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    out_local.backward(grad_local)

    if rank == 0:
        print(f"Backward done. dx shape: {x_local.grad.shape}", flush=True)

    # Compare local chunk of output
    ref_local = ref_out[:, sp_rank * S_local:(sp_rank + 1) * S_local]
    diff = (out_local.detach().float() - ref_local.float()).abs()
    max_diff = diff.max().item()
    max_ref = ref_local.float().abs().max().item() + 1e-10
    rel_diff = max_diff / max_ref

    # Print per-rank results
    for r in range(world_size):
        if rank == r:
            print(f"  rank {r}: max_abs={max_diff:.6e}, max_rel={rel_diff:.6e} {'✅' if rel_diff < 0.01 else '❌'}", flush=True)
        dist.barrier()

    # Gather all ranks' max_rel_diff
    all_max = torch.tensor([rel_diff], device=device)
    dist.all_reduce(all_max, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\n  global max_rel_diff={all_max.item():.6e}", flush=True)
        print(f"  {'OVERALL PASS ✅' if all_max.item() < 0.01 else 'OVERALL FAIL ❌'}", flush=True)

    # Also test backward: compare dx
    x_full_ref = x_full.clone().requires_grad_(True)
    ref_out2, _ = fla_causal_conv1d(x=x_full_ref, weight=weight, activation='silu')
    ref_out2.backward(grad_full)
    dx_ref_local = x_full_ref.grad[:, sp_rank * S_local:(sp_rank + 1) * S_local]
    dx_diff = (x_local.grad.float() - dx_ref_local.float()).abs()
    dx_max_diff = dx_diff.max().item()
    dx_max_ref = dx_ref_local.float().abs().max().item() + 1e-10
    dx_rel = dx_max_diff / dx_max_ref

    all_dx_max = torch.tensor([dx_rel], device=device)
    dist.all_reduce(all_dx_max, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\n  dx global max_rel_diff={all_dx_max.item():.6e}", flush=True)
        print(f"  dx {'PASS ✅' if all_dx_max.item() < 0.05 else 'FAIL ❌'}", flush=True)

    dist.barrier()
    if rank == 0:
        print("\nAll done!", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
