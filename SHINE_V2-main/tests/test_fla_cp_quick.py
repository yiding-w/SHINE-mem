"""Quick smoke test for FLA CP conv1d and chunk_gated_delta_rule."""
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from utils.mytp.fla_cp import build_sp_cp_context

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()

    # Create SP group of 4
    sp_group = None
    for start in range(0, world_size, 4):
        g = dist.new_group(list(range(start, min(start+4, world_size))))
        if start <= rank < start+4:
            sp_group = g

    sp_world = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)

    if rank == 0:
        print(f"World={world_size}, SP_world={sp_world}")

    # ===== Test 1: Conv1d =====
    S_full, D, kernel_size = 2048, 512, 4
    S_local = S_full // sp_world

    torch.manual_seed(42)
    x_full = torch.randn(1, S_full, D, dtype=torch.bfloat16, device=device)
    weight = torch.randn(D, kernel_size, dtype=torch.bfloat16, device=device)

    # Reference
    x_ref = x_full.clone().requires_grad_(True)
    ref_out, _ = fla_causal_conv1d(x=x_ref, weight=weight, activation='silu')
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    dx_ref = x_ref.grad.clone()

    # SP
    x_local = x_full[:, sp_rank*S_local:(sp_rank+1)*S_local].contiguous().clone().requires_grad_(True)
    cp_ctx = build_sp_cp_context(S_local, sp_group, kernel_size, device)
    out_local, _ = fla_causal_conv1d(x=x_local, weight=weight.clone().requires_grad_(True), activation='silu', cp_context=cp_ctx)
    grad_local = grad_out[:, sp_rank*S_local:(sp_rank+1)*S_local].contiguous()
    out_local.backward(grad_local)
    dx_local = x_local.grad.clone()

    # Gather output
    gathered = [torch.empty_like(out_local.detach()) for _ in range(sp_world)]
    dist.all_gather(gathered, out_local.detach(), group=sp_group)
    out_sp = torch.cat(gathered, dim=1)

    # Gather dx
    dx_gathered = [torch.empty_like(dx_local) for _ in range(sp_world)]
    dist.all_gather(dx_gathered, dx_local, group=sp_group)
    dx_sp = torch.cat(dx_gathered, dim=1)

    if rank == 0:
        diff_out = (out_sp.float() - ref_out.detach().float()).abs()
        diff_dx = (dx_sp.float() - dx_ref.float()).abs()
        max_ref_out = ref_out.detach().float().abs().max().item() + 1e-10
        max_ref_dx = dx_ref.float().abs().max().item() + 1e-10
        print(f"\n[Test 1] Conv1d CP (S={S_full}, D={D}, K={kernel_size}, SP={sp_world}):")
        print(f"  output: max_rel={diff_out.max().item()/max_ref_out:.4e}")
        print(f"  dx:     max_rel={diff_dx.max().item()/max_ref_dx:.4e}")
        p1 = diff_out.max().item()/max_ref_out < 0.01 and diff_dx.max().item()/max_ref_dx < 0.05
        print(f"  {'PASS ✅' if p1 else 'FAIL ❌'}")

    dist.barrier()
    torch.cuda.empty_cache()

    # ===== Test 2: chunk_gated_delta_rule =====
    S_full, num_heads, head_k, head_v = 2048, 8, 128, 128
    S_local = S_full // sp_world

    torch.manual_seed(42)
    q_full = torch.randn(1, S_full, num_heads, head_k, dtype=torch.bfloat16, device=device, requires_grad=True)
    k_full = torch.randn(1, S_full, num_heads, head_k, dtype=torch.bfloat16, device=device, requires_grad=True)
    v_full = torch.randn(1, S_full, num_heads, head_v, dtype=torch.bfloat16, device=device, requires_grad=True)
    # g must be negative (decay factor) to avoid state explosion
    g_full = -torch.rand(1, S_full, num_heads, dtype=torch.float32, device=device).requires_grad_(True)
    beta_full = torch.sigmoid(torch.randn(1, S_full, num_heads, dtype=torch.bfloat16, device=device)).requires_grad_(True)

    ref_out2, _ = chunk_gated_delta_rule(q=q_full, k=k_full, v=v_full, g=g_full, beta=beta_full, use_qk_l2norm_in_kernel=True)
    grad_out2 = torch.randn_like(ref_out2)
    ref_out2.backward(grad_out2)
    dq_ref = q_full.grad.clone()
    dk_ref = k_full.grad.clone()
    dv_ref = v_full.grad.clone()

    # SP
    def lc(t): return t.detach()[:, sp_rank*S_local:(sp_rank+1)*S_local].contiguous().clone().requires_grad_(True)
    q_l, k_l, v_l, g_l, beta_l = lc(q_full), lc(k_full), lc(v_full), lc(g_full), lc(beta_full)

    cp_ctx2 = build_sp_cp_context(S_local, sp_group, conv1d_kernel_size=4, device=device)
    out_l2, _ = chunk_gated_delta_rule(q=q_l, k=k_l, v=v_l, g=g_l, beta=beta_l, use_qk_l2norm_in_kernel=True, cp_context=cp_ctx2)
    grad_l2 = grad_out2[:, sp_rank*S_local:(sp_rank+1)*S_local].contiguous()
    out_l2.backward(grad_l2)

    def gather(t):
        g = [torch.empty_like(t) for _ in range(sp_world)]
        dist.all_gather(g, t, group=sp_group)
        return torch.cat(g, dim=1)

    out_sp2 = gather(out_l2.detach())
    dq_sp = gather(q_l.grad)
    dk_sp = gather(k_l.grad)
    dv_sp = gather(v_l.grad)

    if rank == 0:
        def rel(a, b): return (a.float()-b.float()).abs().max().item() / (b.float().abs().max().item()+1e-10)
        def cos(a, b): return torch.nn.functional.cosine_similarity(a.float().reshape(1,-1), b.float().reshape(1,-1)).item()
        print(f"\n[Test 2] chunk_gated_delta_rule CP (S={S_full}, H={num_heads}, SP={sp_world}):")
        print(f"  output: max_rel={rel(out_sp2, ref_out2.detach()):.4e}, cos={cos(out_sp2, ref_out2.detach()):.8f}")
        print(f"  dQ:     max_rel={rel(dq_sp, dq_ref):.4e}, cos={cos(dq_sp, dq_ref):.8f}")
        print(f"  dK:     max_rel={rel(dk_sp, dk_ref):.4e}, cos={cos(dk_sp, dk_ref):.8f}")
        print(f"  dV:     max_rel={rel(dv_sp, dv_ref):.4e}, cos={cos(dv_sp, dv_ref):.8f}")
        # For bf16, 10% max_rel and cos > 0.999 is acceptable
        p2 = (rel(out_sp2, ref_out2.detach()) < 0.01 and
               cos(dq_sp, dq_ref) > 0.999 and
               cos(dk_sp, dk_ref) > 0.999 and
               cos(dv_sp, dv_ref) > 0.999)
        print(f"  {'PASS ✅' if p2 else 'FAIL ❌'}")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\nDone!")

if __name__ == "__main__":
    main()
