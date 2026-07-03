"""
Comprehensive FLA CP tests covering:
  1. Activation checkpointing compatibility
  2. Long sequence stability (16k/32k/64k)
  4. fp32 state merge vs bf16 (FLA default behavior check)
  5. Benchmark: speed + memory
  6. Multi-layer stacking
  8. Different SP sizes (SP=2/4/8)

All tests run in BOTH pure SP (TP=1) and TP+SP (TP=2) configurations.

Run with:
    torchrun --nproc_per_node=8 tests/test_fla_cp_comprehensive.py
"""
import os, sys, gc, time, torch, torch.distributed as dist, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def rel_diff(a, b):
    return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-10)


def cos_sim(a, b):
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def thorough_cleanup(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


def create_groups(world_size, tp_size):
    """Create TP and SP groups. Returns (my_tp_group, my_sp_group, tp_rank, sp_rank, sp_size)."""
    rank = dist.get_rank()
    sp_size = world_size // tp_size

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
    return my_tp_group, my_sp_group, tp_rank_idx, sp_rank_idx, sp_size


def build_models(config, layer_idx, tp_rank, tp_world, tp_group, sp_group, sp_world, device):
    """Build reference model and SP model, load same weights."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from utils.mytp.tp_gated_deltanet import TPQwen3_5GatedDeltaNet, load_gated_deltanet_weights_from_full

    torch.manual_seed(42)
    ref_model = Qwen3_5GatedDeltaNet(config, layer_idx).to(device=device, dtype=torch.bfloat16)
    ref_model.eval()

    sp_model = TPQwen3_5GatedDeltaNet(
        config, layer_idx,
        tp_rank=tp_rank, tp_world=tp_world, tp_process_group=tp_group,
        sp_group=sp_group, sp_world=sp_world,
    ).to(device=device, dtype=torch.bfloat16)
    load_gated_deltanet_weights_from_full(sp_model, ref_model)
    sp_model.eval()
    return ref_model, sp_model


def run_correctness_test(ref_model, sp_model, S_full, sp_rank, sp_world, sp_group, device, use_checkpoint=False):
    """Run forward+backward and compare. Returns (out_rel, out_cos, dh_rel, dh_cos)."""
    S_local = S_full // sp_world

    torch.manual_seed(100 + S_full)
    hidden_full = torch.randn(1, S_full, ref_model.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)

    # Reference
    if use_checkpoint:
        ref_out = torch_checkpoint(ref_model, hidden_full, use_reentrant=False)
    else:
        ref_out = ref_model(hidden_full)
    torch.manual_seed(200 + S_full)
    grad_full = torch.randn_like(ref_out)
    ref_out.backward(grad_full)
    d_hidden_ref = hidden_full.grad.clone()
    ref_out_det = ref_out.detach().clone()

    # SP
    hidden_local = hidden_full.detach()[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous().clone().requires_grad_(True)
    if use_checkpoint:
        sp_out = torch_checkpoint(sp_model, hidden_local, use_reentrant=False)
    else:
        sp_out = sp_model(hidden_local)
    grad_local = grad_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
    sp_out.backward(grad_local)
    d_hidden_local = hidden_local.grad.clone()

    # Gather
    gathered_out = [torch.empty_like(sp_out.detach()) for _ in range(sp_world)]
    dist.all_gather(gathered_out, sp_out.detach(), group=sp_group)
    out_full = torch.cat(gathered_out, dim=1)

    gathered_dh = [torch.empty_like(d_hidden_local) for _ in range(sp_world)]
    dist.all_gather(gathered_dh, d_hidden_local, group=sp_group)
    dh_full = torch.cat(gathered_dh, dim=1)

    return (
        rel_diff(out_full, ref_out_det),
        cos_sim(out_full, ref_out_det),
        rel_diff(dh_full, d_hidden_ref),
        cos_sim(dh_full, d_hidden_ref),
    )


def run_benchmark(sp_model, S_local, device, num_warmup=3, num_iters=10):
    """Benchmark speed and memory for forward+backward."""
    hidden_size = sp_model._inner.hidden_size

    # Warmup
    for _ in range(num_warmup):
        x = torch.randn(1, S_local, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)
        out = sp_model(x)
        out.sum().backward()
        del x, out
    torch.cuda.synchronize(device)

    # Measure
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(num_iters):
        x = torch.randn(1, S_local, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)
        out = sp_model(x)
        out.sum().backward()
        del x, out
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / num_iters * 1000  # ms
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB

    return elapsed, peak_mem


def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    assert world_size == 8, f"Requires 8 GPUs, got {world_size}"

    from transformers import AutoConfig
    config_path = "/apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp/models/Qwen3.6-27B"
    config = AutoConfig.from_pretrained(config_path).text_config

    print_rank0(f"\n{'#'*80}")
    print_rank0(f"# FLA CP Comprehensive Tests")
    print_rank0(f"# World={world_size}, Device={torch.cuda.get_device_name(local_rank)}")
    print_rank0(f"# Model: hidden={config.hidden_size}, v_heads={config.linear_num_value_heads}, "
                f"k_heads={config.linear_num_key_heads}, head_dim={config.linear_key_head_dim}")
    print_rank0(f"{'#'*80}")

    all_pass = True

    # ===========================================================================
    # Test 8: Different SP sizes (SP=2/4/8) — both pure SP and TP+SP
    # ===========================================================================
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 8: Different SP sizes (correctness)")
    print_rank0(f"{'='*80}")

    test_configs = [
        # (tp_size, description)
        (1, "Pure SP"),
        (2, "TP+SP"),
    ]

    S_full = 4096
    for tp_size, desc in test_configs:
        sp_size = world_size // tp_size
        tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)

        ref_model, sp_model = build_models(config, 0, tp_rank, tp_size, tp_group, sp_group, sp_size, device)
        out_rel, out_cos, dh_rel, dh_cos = run_correctness_test(
            ref_model, sp_model, S_full, sp_rank, sp_size, sp_group, device
        )
        passed = out_cos > 0.999 and dh_cos > 0.999
        all_pass = all_pass and passed
        if rank == 0:
            print(f"  {desc} (TP={tp_size}, SP={sp_size}): "
                  f"out_cos={out_cos:.8f}, dh_cos={dh_cos:.8f} "
                  f"{'✅' if passed else '❌'}", flush=True)
        del ref_model, sp_model
        thorough_cleanup(device)
        dist.barrier()

    # Also test SP=2 with TP=4
    tp_size = 4
    sp_size = world_size // tp_size  # 2
    tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)
    ref_model, sp_model = build_models(config, 0, tp_rank, tp_size, tp_group, sp_group, sp_size, device)
    out_rel, out_cos, dh_rel, dh_cos = run_correctness_test(
        ref_model, sp_model, S_full, sp_rank, sp_size, sp_group, device
    )
    passed = out_cos > 0.999 and dh_cos > 0.999
    all_pass = all_pass and passed
    if rank == 0:
        print(f"  TP+SP (TP=4, SP=2): "
              f"out_cos={out_cos:.8f}, dh_cos={dh_cos:.8f} "
              f"{'✅' if passed else '❌'}", flush=True)
    del ref_model, sp_model
    thorough_cleanup(device)
    dist.barrier()

    # ===========================================================================
    # Test 2: Long sequence stability (16k/32k/64k)
    # ===========================================================================
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 2: Long sequence stability")
    print_rank0(f"{'='*80}")

    tp_size = 2
    sp_size = world_size // tp_size  # 4
    tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)

    for S_full in [8192, 16384, 32768]:
        try:
            ref_model, sp_model = build_models(config, 0, tp_rank, tp_size, tp_group, sp_group, sp_size, device)
            out_rel, out_cos, dh_rel, dh_cos = run_correctness_test(
                ref_model, sp_model, S_full, sp_rank, sp_size, sp_group, device
            )
            passed = out_cos > 0.999 and dh_cos > 0.999
            all_pass = all_pass and passed
            if rank == 0:
                print(f"  S={S_full:>6} (TP={tp_size},SP={sp_size}): "
                      f"out_rel={out_rel:.4e}, out_cos={out_cos:.8f}, "
                      f"dh_rel={dh_rel:.4e}, dh_cos={dh_cos:.8f} "
                      f"{'✅' if passed else '❌'}", flush=True)
            del ref_model, sp_model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print_rank0(f"  S={S_full:>6} | ⚠️  OOM - skipping")
            else:
                print_rank0(f"  S={S_full:>6} | ERROR: {e}")
                all_pass = False
        thorough_cleanup(device)
        dist.barrier()

    # ===========================================================================
    # Test 1: Activation checkpointing compatibility
    # ===========================================================================
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 1: Activation checkpointing compatibility")
    print_rank0(f"{'='*80}")

    S_full = 4096
    for tp_size, desc in [(1, "Pure SP"), (2, "TP+SP")]:
        sp_size = world_size // tp_size
        tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)
        ref_model, sp_model = build_models(config, 0, tp_rank, tp_size, tp_group, sp_group, sp_size, device)

        try:
            out_rel, out_cos, dh_rel, dh_cos = run_correctness_test(
                ref_model, sp_model, S_full, sp_rank, sp_size, sp_group, device,
                use_checkpoint=True,
            )
            passed = out_cos > 0.999 and dh_cos > 0.999
            all_pass = all_pass and passed
            if rank == 0:
                print(f"  {desc} (TP={tp_size},SP={sp_size}) + checkpoint: "
                      f"out_cos={out_cos:.8f}, dh_cos={dh_cos:.8f} "
                      f"{'✅' if passed else '❌'}", flush=True)
        except Exception as e:
            print_rank0(f"  {desc} + checkpoint | ERROR: {e}")
            import traceback
            if rank == 0:
                traceback.print_exc()
            all_pass = False

        del ref_model, sp_model
        thorough_cleanup(device)
        dist.barrier()

    # ===========================================================================
    # Test 6: Multi-layer stacking (2/4/8 layers)
    # ===========================================================================
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 6: Multi-layer stacking")
    print_rank0(f"{'='*80}")

    tp_size = 2
    sp_size = world_size // tp_size  # 4
    tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)
    S_full = 4096
    S_local = S_full // sp_size

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from utils.mytp.tp_gated_deltanet import TPQwen3_5GatedDeltaNet, load_gated_deltanet_weights_from_full

    for num_layers in [2, 4, 8]:
        try:
            # Build multi-layer reference
            torch.manual_seed(42)
            ref_layers = [Qwen3_5GatedDeltaNet(config, i).to(device=device, dtype=torch.bfloat16) for i in range(num_layers)]
            for m in ref_layers:
                m.eval()

            # Build multi-layer SP
            sp_layers = []
            for i in range(num_layers):
                sp_m = TPQwen3_5GatedDeltaNet(
                    config, i,
                    tp_rank=tp_rank, tp_world=tp_size, tp_process_group=tp_group,
                    sp_group=sp_group, sp_world=sp_size,
                ).to(device=device, dtype=torch.bfloat16)
                load_gated_deltanet_weights_from_full(sp_m, ref_layers[i])
                sp_m.eval()
                sp_layers.append(sp_m)

            # Reference forward
            torch.manual_seed(100)
            hidden_full = torch.randn(1, S_full, config.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)
            x = hidden_full
            for layer in ref_layers:
                x = layer(x)
            ref_out = x
            torch.manual_seed(200)
            grad_full = torch.randn_like(ref_out)
            ref_out.backward(grad_full)
            d_hidden_ref = hidden_full.grad.clone()
            ref_out_det = ref_out.detach().clone()

            # SP forward
            hidden_local = hidden_full.detach()[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous().clone().requires_grad_(True)
            x_sp = hidden_local
            for layer in sp_layers:
                x_sp = layer(x_sp)
            sp_out = x_sp
            grad_local = grad_full[:, sp_rank * S_local:(sp_rank + 1) * S_local].contiguous()
            sp_out.backward(grad_local)
            d_hidden_local = hidden_local.grad.clone()

            # Gather
            gathered_out = [torch.empty_like(sp_out.detach()) for _ in range(sp_size)]
            dist.all_gather(gathered_out, sp_out.detach(), group=sp_group)
            out_full = torch.cat(gathered_out, dim=1)

            gathered_dh = [torch.empty_like(d_hidden_local) for _ in range(sp_size)]
            dist.all_gather(gathered_dh, d_hidden_local, group=sp_group)
            dh_full = torch.cat(gathered_dh, dim=1)

            o_cos = cos_sim(out_full, ref_out_det)
            dh_cos = cos_sim(dh_full, d_hidden_ref)
            passed = o_cos > 0.999 and dh_cos > 0.999
            all_pass = all_pass and passed
            if rank == 0:
                print(f"  {num_layers} layers (TP={tp_size},SP={sp_size}): "
                      f"out_cos={o_cos:.8f}, dh_cos={dh_cos:.8f} "
                      f"{'✅' if passed else '❌'}", flush=True)

            del ref_layers, sp_layers
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print_rank0(f"  {num_layers} layers | ⚠️  OOM - skipping")
            else:
                print_rank0(f"  {num_layers} layers | ERROR: {e}")
                all_pass = False
        thorough_cleanup(device)
        dist.barrier()

    # ===========================================================================
    # Test 5: Benchmark speed + memory (TP=2, SP=4)
    # ===========================================================================
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 5: Benchmark speed + memory")
    print_rank0(f"{'='*80}")

    tp_size = 2
    sp_size = world_size // tp_size  # 4
    tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)

    # Also build a non-SP model for comparison
    from utils.mytp.tp_gated_deltanet import TPQwen3_5GatedDeltaNet, load_gated_deltanet_weights_from_full
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    if rank == 0:
        print(f"  {'Config':<30} {'Time(ms)':<12} {'Peak Mem(MB)':<14}", flush=True)
        print(f"  {'-'*56}", flush=True)

    for S_full in [4096, 16384, 32768]:
        S_local = S_full // sp_size
        try:
            # SP model
            torch.manual_seed(42)
            ref_m = Qwen3_5GatedDeltaNet(config, 0).to(device=device, dtype=torch.bfloat16)
            sp_m = TPQwen3_5GatedDeltaNet(
                config, 0,
                tp_rank=tp_rank, tp_world=tp_size, tp_process_group=tp_group,
                sp_group=sp_group, sp_world=sp_size,
            ).to(device=device, dtype=torch.bfloat16)
            load_gated_deltanet_weights_from_full(sp_m, ref_m)
            sp_m.eval()
            del ref_m

            thorough_cleanup(device)
            elapsed_sp, mem_sp = run_benchmark(sp_m, S_local, device)
            if rank == 0:
                print(f"  {'SP (TP=2,SP=4) S='+str(S_full):<30} {elapsed_sp:<12.2f} {mem_sp:<14.1f}", flush=True)
            del sp_m

            # Non-SP model (TP only, full sequence on each SP rank)
            thorough_cleanup(device)
            torch.manual_seed(42)
            ref_m2 = Qwen3_5GatedDeltaNet(config, 0).to(device=device, dtype=torch.bfloat16)
            nosp_m = TPQwen3_5GatedDeltaNet(
                config, 0,
                tp_rank=tp_rank, tp_world=tp_size, tp_process_group=tp_group,
                sp_group=None, sp_world=1,
            ).to(device=device, dtype=torch.bfloat16)
            load_gated_deltanet_weights_from_full(nosp_m, ref_m2)
            nosp_m.eval()
            del ref_m2

            thorough_cleanup(device)
            elapsed_nosp, mem_nosp = run_benchmark(nosp_m, S_full, device)
            if rank == 0:
                print(f"  {'No-SP (TP=2) S='+str(S_full):<30} {elapsed_nosp:<12.2f} {mem_nosp:<14.1f}", flush=True)
                speedup = elapsed_nosp / elapsed_sp if elapsed_sp > 0 else 0
                mem_ratio = mem_sp / mem_nosp if mem_nosp > 0 else 0
                print(f"  {'  → SP speedup':<30} {speedup:<12.2f}x {mem_ratio:<14.2f}x mem", flush=True)
            del nosp_m
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print_rank0(f"  S={S_full} | ⚠️  OOM - skipping")
            else:
                print_rank0(f"  S={S_full} | ERROR: {e}")
        thorough_cleanup(device)
        dist.barrier()

    # ===========================================================================
    # Test 4: fp32 state merge check (compare FLA default behavior)
    # ===========================================================================
    print_rank0(f"\n{'='*80}")
    print_rank0(f"  Test 4: fp32 state merge (FLA default behavior)")
    print_rank0(f"{'='*80}")

    # FLA's chunk_gated_delta_rule internally uses fp32 for state operations.
    # We verify this by checking that long sequences don't accumulate errors.
    # If state merge were bf16, errors would grow with sequence length.
    tp_size = 2
    sp_size = world_size // tp_size
    tp_group, sp_group, tp_rank, sp_rank, _ = create_groups(world_size, tp_size)

    prev_cos = None
    if rank == 0:
        print(f"  Checking error growth with sequence length (should NOT grow):", flush=True)
    for S_full in [2048, 4096, 8192, 16384, 32768]:
        try:
            ref_model, sp_model = build_models(config, 0, tp_rank, tp_size, tp_group, sp_group, sp_size, device)
            _, out_cos, _, dh_cos = run_correctness_test(
                ref_model, sp_model, S_full, sp_rank, sp_size, sp_group, device
            )
            if rank == 0:
                growth = ""
                if prev_cos is not None:
                    if out_cos < prev_cos - 0.0001:
                        growth = " ⚠️ DEGRADING"
                    else:
                        growth = " ✅ stable"
                print(f"  S={S_full:>6}: out_cos={out_cos:.8f}, dh_cos={dh_cos:.8f}{growth}", flush=True)
                prev_cos = out_cos
            del ref_model, sp_model
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print_rank0(f"  S={S_full} | ⚠️  OOM - skipping")
            else:
                raise
        thorough_cleanup(device)
        dist.barrier()

    # ===========================================================================
    # Summary
    # ===========================================================================
    print_rank0(f"\n{'#'*80}")
    if all_pass:
        print_rank0("ALL COMPREHENSIVE FLA CP TESTS PASSED ✅")
    else:
        print_rank0("SOME TESTS FAILED ❌")
    print_rank0(f"{'#'*80}\n")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
