"""Test 3: Full GatedDeltaNet forward_sp (pure SP, no TP) + TP+SP."""
import os, sys, torch, torch.distributed as dist, torch.nn.functional as F
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

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from utils.mytp.tp_gated_deltanet import TPQwen3_5GatedDeltaNet, load_gated_deltanet_weights_from_full

    config_path = "/apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp/models/Qwen3.6-27B"
    config = AutoConfig.from_pretrained(config_path).text_config
    layer_idx = 0

    def rel(a, b): return (a.float()-b.float()).abs().max().item() / (b.float().abs().max().item()+1e-10)
    def cos(a, b): return F.cosine_similarity(a.float().reshape(1,-1), b.float().reshape(1,-1)).item()

    # ===== Test 3: Pure SP (TP=1, SP=4) =====
    if rank == 0:
        print(f"\n[Test 3] Full GatedDeltaNet forward_sp (TP=1, SP=4)", flush=True)

    # SP group of 4
    sp_group = None
    for start in range(0, world_size, 4):
        g = dist.new_group(list(range(start, min(start+4, world_size))))
        if start <= rank < start+4:
            sp_group = g
    sp_world = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)

    # Dummy TP group (single rank)
    tp_group = dist.new_group([rank])

    S_full = 2048
    S_local = S_full // sp_world

    # Reference model (single GPU, full sequence)
    torch.manual_seed(42)
    ref_model = Qwen3_5GatedDeltaNet(config, layer_idx).to(device=device, dtype=torch.bfloat16)
    ref_model.eval()

    # SP model (TP=1, SP=4)
    sp_model = TPQwen3_5GatedDeltaNet(
        config, layer_idx,
        tp_rank=0, tp_world=1, tp_process_group=tp_group,
        sp_group=sp_group, sp_world=sp_world,
    ).to(device=device, dtype=torch.bfloat16)
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
    ref_out_det = ref_out.detach().clone()

    # SP forward + backward
    hidden_local = hidden_full.detach()[:, sp_rank*S_local:(sp_rank+1)*S_local].contiguous().clone().requires_grad_(True)
    sp_out = sp_model(hidden_local)
    grad_local = grad_out_full[:, sp_rank*S_local:(sp_rank+1)*S_local].contiguous()
    sp_out.backward(grad_local)
    d_hidden_local = hidden_local.grad.clone()

    # Gather
    def gather_sp(t, group):
        sw = dist.get_world_size(group)
        gathered = [torch.empty_like(t) for _ in range(sw)]
        dist.all_gather(gathered, t, group=group)
        return torch.cat(gathered, dim=1)

    out_sp_full = gather_sp(sp_out.detach(), sp_group)
    d_hidden_sp_full = gather_sp(d_hidden_local, sp_group)

    if rank == 0:
        print(f"  output: max_rel={rel(out_sp_full, ref_out_det):.4e}, cos={cos(out_sp_full, ref_out_det):.8f}", flush=True)
        print(f"  d_hidden: max_rel={rel(d_hidden_sp_full, d_hidden_ref):.4e}, cos={cos(d_hidden_sp_full, d_hidden_ref):.8f}", flush=True)
        p3 = cos(out_sp_full, ref_out_det) > 0.999 and cos(d_hidden_sp_full, d_hidden_ref) > 0.999
        print(f"  {'PASS ✅' if p3 else 'FAIL ❌'}", flush=True)

    del ref_model, sp_model
    torch.cuda.empty_cache()
    dist.barrier()

    # ===== Test 4: TP+SP (TP=2, SP=4) =====
    if rank == 0:
        print(f"\n[Test 4] TP+SP (TP=2, SP=4)", flush=True)

    tp_size = 2
    sp_size = world_size // tp_size  # 4

    # Create TP and SP groups
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

    S_full = 2048
    S_local = S_full // sp_size

    # Reference model
    torch.manual_seed(42)
    ref_model2 = Qwen3_5GatedDeltaNet(config, layer_idx).to(device=device, dtype=torch.bfloat16)
    ref_model2.eval()

    # TP+SP model
    tp_sp_model = TPQwen3_5GatedDeltaNet(
        config, layer_idx,
        tp_rank=tp_rank_idx, tp_world=tp_size, tp_process_group=my_tp_group,
        sp_group=my_sp_group, sp_world=sp_size,
    ).to(device=device, dtype=torch.bfloat16)
    load_gated_deltanet_weights_from_full(tp_sp_model, ref_model2)
    tp_sp_model.eval()

    # Generate input
    torch.manual_seed(100)
    hidden_full2 = torch.randn(1, S_full, config.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Reference
    ref_out2 = ref_model2(hidden_full2)
    torch.manual_seed(200)
    grad_out_full2 = torch.randn_like(ref_out2)
    ref_out2.backward(grad_out_full2)
    d_hidden_ref2 = hidden_full2.grad.clone()
    ref_out_det2 = ref_out2.detach().clone()

    # TP+SP forward
    hidden_local2 = hidden_full2.detach()[:, sp_rank_idx*S_local:(sp_rank_idx+1)*S_local].contiguous().clone().requires_grad_(True)
    tp_sp_out = tp_sp_model(hidden_local2)
    grad_local2 = grad_out_full2[:, sp_rank_idx*S_local:(sp_rank_idx+1)*S_local].contiguous()
    tp_sp_out.backward(grad_local2)
    d_hidden_local2 = hidden_local2.grad.clone()

    # Gather across SP
    out_tp_sp_full = gather_sp(tp_sp_out.detach(), my_sp_group)
    d_hidden_tp_sp_full = gather_sp(d_hidden_local2, my_sp_group)

    if rank == 0:
        print(f"  output: max_rel={rel(out_tp_sp_full, ref_out_det2):.4e}, cos={cos(out_tp_sp_full, ref_out_det2):.8f}", flush=True)
        print(f"  d_hidden: max_rel={rel(d_hidden_tp_sp_full, d_hidden_ref2):.4e}, cos={cos(d_hidden_tp_sp_full, d_hidden_ref2):.8f}", flush=True)
        p4 = cos(out_tp_sp_full, ref_out_det2) > 0.999 and cos(d_hidden_tp_sp_full, d_hidden_ref2) > 0.999
        print(f"  {'PASS ✅' if p4 else 'FAIL ❌'}", flush=True)

    dist.barrier()
    if rank == 0:
        print("\nAll done!", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
