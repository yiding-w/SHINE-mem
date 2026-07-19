#!/usr/bin/env python
"""Compare official SHINE and this fork, stage by stage and tensor by tensor.

The two implementations are captured in separate child processes so an 8B
model from one source tree is released before the other one is loaded.  The
resulting CPU artifacts are then compared without importing either SHINE tree.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import gc
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


KEY_SOURCE_FILES = (
    "LoraQwen.py",
    "metanetwork_family.py",
    "utils/mysaveload.py",
    "utils/mydataset.py",
    "test_pretrain.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_file_manifest(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in KEY_SOURCE_FILES:
        path = root / relative
        result[relative] = {
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return result


def write_source_diff(official_root: Path, current_root: Path, output: Path) -> None:
    chunks: list[str] = []
    for relative in KEY_SOURCE_FILES:
        official_path = official_root / relative
        current_path = current_root / relative
        if not official_path.is_file() or not current_path.is_file():
            chunks.append(f"Missing file for comparison: {relative}\n")
            continue
        official_lines = official_path.read_text(encoding="utf-8").splitlines(keepends=True)
        current_lines = current_path.read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                official_lines,
                current_lines,
                fromfile=f"official/{relative}",
                tofile=f"current/{relative}",
            )
        )
    output.write_text("".join(chunks), encoding="utf-8")


def extract_official_chat_template(source_root: Path) -> str:
    path = source_root / "test_pretrain.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "chat_template":
                if isinstance(node.value.value, str):
                    return node.value.value
    raise RuntimeError(f"Could not find tokenizer.chat_template literal in {path}")


def _path_key(path: tuple[Any, ...]) -> str:
    return "/".join(str(part) for part in path)


def flatten_tensors(value: Any, path: tuple[Any, ...] = ()) -> dict[str, Any]:
    """Flatten an arbitrarily nested LoRA dictionary without losing tensors."""
    import torch

    if torch.is_tensor(value):
        return {_path_key(path): value}
    # LoRA leaves contain C=None when the wrapped linear has no bias.
    if value is None:
        return {}
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: (str(type(item)), str(item))):
            result.update(flatten_tensors(value[key], path + (key,)))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_tensors(item, path + (index,)))
        return result
    raise TypeError(f"Unsupported non-tensor value at {_path_key(path)}: {type(value)!r}")


def tensor_signature(tensor: Any) -> dict[str, Any]:
    """Hash every tensor byte and provide numerical diagnostics."""
    import torch

    detached = tensor.detach().contiguous().cpu()
    byte_view = detached.view(torch.uint8)
    digest = sha256_bytes(byte_view.numpy().tobytes())
    floating = detached.float()
    if detached.numel():
        minimum = float(floating.min())
        maximum = float(floating.max())
        mean = float(floating.mean())
        rms = float(floating.square().mean().sqrt())
    else:
        minimum = maximum = mean = rms = 0.0
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": detached.numel(),
        "sha256": digest,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "rms": rms,
    }


def named_parameter_signatures(module: Any) -> dict[str, Any]:
    return {name: tensor_signature(parameter) for name, parameter in module.named_parameters()}


def nested_tensor_signatures(value: Any) -> dict[str, Any]:
    return {name: tensor_signature(tensor) for name, tensor in flatten_tensors(value).items()}


def cpu_tensor(tensor: Any) -> Any:
    return tensor.detach().contiguous().cpu()


def set_all_seeds(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cfg(args: argparse.Namespace, config: Any) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "name": "source_parity",
            "run": {"seed": args.seed, "device": args.device},
            "paths": {"model_path": args.model_path},
            "optim": {"adapter_reg": 0.0},
            "model": {
                "lora_r": args.lora_r,
                "metalora_r": args.metalora_r,
                "ift_additional_metalora_r": -1,
                "num_mem_token": 4,
                "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
                "config_class_path": "LoraQwen.Qwen3Config",
                "tokenizer_from": args.model_path,
                "model_from": args.model_path,
            },
            "metanetwork": {
                "type": "transformer",
                "method": "rl",
                "transformer_cfg": {
                    "encoder_cfg": {
                        "d_model": config.hidden_size,
                        "nhead": args.hyper_nhead,
                        "dim_feedforward": args.hyper_ffn_dim,
                        "dropout": 0,
                        "activation": "gelu",
                        "layer_norm_eps": 0.00001,
                        "batch_first": True,
                        "norm_first": False,
                        "bias": True,
                    },
                    "couple_encoder_cfg": {
                        "d_model": config.hidden_size,
                        "nhead": args.hyper_nhead,
                        "dim_feedforward": args.hyper_ffn_dim,
                        "dropout": 0,
                        "activation": "gelu",
                        "layer_norm_eps": 0.00001,
                        "batch_first": True,
                        "norm_first": False,
                        "bias": True,
                    },
                    "layer_transformer_first": True,
                    "mean_pool_size": 1,
                    "num_layers": args.metanetwork_layers,
                    "couple_num_layers": 0,
                    "scale": args.generated_lora_scale,
                },
            },
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_mem_token": 4,
        }
    )


def select_device(device_arg: str) -> Any:
    import torch

    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({device_arg}) but torch.cuda.is_available() is false")
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def import_source_runtime(source_root: Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import only from source_root. Capture runs in a fresh process."""
    source_root = source_root.resolve()
    required = (source_root / "LoraQwen.py", source_root / "metanetwork_family.py")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(source_root))

    from transformers import AutoTokenizer
    from metanetwork_family import Metanetwork
    from utils.myfreeze import freeze
    from utils.myinit import _import_class
    from utils.mysaveload import load_checkpoint

    MetaModelCls = _import_class("LoraQwen.LoraQwen3ForCausalLM")
    ConfigCls = _import_class("LoraQwen.Qwen3Config")
    imported_files = {
        "LoraQwen": Path(sys.modules["LoraQwen"].__file__).resolve(),
        "metanetwork_family": Path(sys.modules["metanetwork_family"].__file__).resolve(),
        "utils.mysaveload": Path(sys.modules["utils.mysaveload"].__file__).resolve(),
    }
    wrong = {name: path for name, path in imported_files.items() if source_root not in path.parents}
    if wrong:
        raise RuntimeError(f"Source isolation failed for {source_root}: {wrong}")
    return AutoTokenizer, Metanetwork, freeze, load_checkpoint, MetaModelCls, ConfigCls


def build_official_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    """Reproduce official test_pretrain.py construction in its original order."""
    import torch

    source_root = Path(args.source_root).resolve()
    AutoTokenizer, Metanetwork, freeze, load_checkpoint, MetaModelCls, ConfigCls = import_source_runtime(
        source_root
    )
    device = select_device(args.device)
    set_all_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = ConfigCls.from_pretrained(args.model_path)
    config.num_mem_token = -1
    cfg = build_cfg(args, config)

    # This expensive temporary load is intentional: it exactly matches the RNG
    # consumption and construction sequence in official test_pretrain.py.
    print(f"[{args.label}] constructing temporary model to calculate num_mem_token", flush=True)
    tmp_model = MetaModelCls.from_pretrained(args.model_path, config=config)
    lora_numel = tmp_model.lora_params_numel(args.lora_r)
    denominator = config.hidden_size * config.num_hidden_layers
    if lora_numel % denominator:
        raise ValueError(f"lora_numel={lora_numel} is not divisible by hidden*layers={denominator}")
    config.num_mem_token = lora_numel // denominator
    cfg.num_mem_token = config.num_mem_token
    del tmp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", use_fast=True)
    tokenizer.add_tokens(["<RECON>", "<COMP>"])
    tokenizer.chat_template = extract_official_chat_template(source_root)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{args.label}] constructing final model in official FP32 mode", flush=True)
    metamodel = MetaModelCls.from_pretrained(args.model_path, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(args.lora_r))
    metanetwork.to(device)
    freeze(metamodel)
    metanetwork, metalora, _ = load_checkpoint(metanetwork, args.checkpoint_dir, device)
    metanetwork.eval()
    return metanetwork, metalora, tokenizer, cfg, device


def encode_official_reconstruction(tokenizer: Any, context: str, args: argparse.Namespace, device: Any) -> dict[str, Any]:
    """Use the exact TestPretrainCollator reconstruction layout."""
    import torch

    evidence = tokenizer(
        [context],
        max_length=args.context_max_length,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )
    prompt_messages = [[{"role": "user", "content": "<RECON>"}]]
    label_messages = [[
        {"role": "user", "content": "<RECON>"},
        {"role": "assistant", "content": context},
    ]]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=9,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        label_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        max_length=args.answer_max_length,
        truncation=True,
        return_dict=True,
        padding="max_length",
        enable_thinking=False,
    )
    # Copy BaseCollator.mask_label literally.  Reimplementing it here avoids
    # importing datasets/pandas just to create the parity input.
    labels = full["input_ids"].clone()
    masks = torch.zeros_like(labels)
    assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
    imstart_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    imend_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for batch_index, row in enumerate(labels):
        last_imend = args.answer_max_length
        for token_index in range(len(row) - 1, 0, -1):
            if row[token_index].item() == imend_token_id:
                last_imend = token_index
            elif (
                row[token_index].item() == assistant_token_id
                and row[token_index - 1].item() == imstart_token_id
            ):
                masks[batch_index, token_index + 2 : last_imend + 2] = 1
    labels = labels.masked_fill(masks == 0, -100)
    return {
        "evidence_ids": evidence["input_ids"].to(device),
        "evidence_attention_mask": evidence["attention_mask"].to(device),
        "prompt_ids": prompt["input_ids"].to(device),
        "prompt_attention_mask": prompt["attention_mask"].to(device),
        "full_ids": full["input_ids"].to(device),
        "full_attention_mask": full["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def extract_hidden(output: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def capture(args: argparse.Namespace) -> None:
    import torch
    import transformers

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root).resolve()
    metanetwork, metalora, tokenizer, cfg, device = build_official_runtime(args)
    encoded = encode_official_reconstruction(tokenizer, args.context, args, device)

    manifest: dict[str, Any] = {
        "label": args.label,
        "source_root": str(source_root),
        "source_git_revision": git_revision(source_root),
        "source_files": source_file_manifest(source_root),
        "import_origins": {
            name: str(Path(sys.modules[name].__file__).resolve())
            for name in ("LoraQwen", "metanetwork_family", "utils.mysaveload")
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": args.seed,
        "device": str(device),
        "context": args.context,
        "num_mem_token": int(cfg.num_mem_token),
        "tokenizer": {
            "length": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "recon_token_id": tokenizer.convert_tokens_to_ids("<RECON>"),
            "comp_token_id": tokenizer.convert_tokens_to_ids("<COMP>"),
            "chat_template_sha256": sha256_bytes(tokenizer.chat_template.encode("utf-8")),
        },
        "inputs": {},
        "parameters": {},
        "stages": {},
    }
    artifacts: dict[str, Any] = {}
    for name, tensor in encoded.items():
        manifest["inputs"][name] = tensor_signature(tensor)
        artifacts[f"input/{name}"] = cpu_tensor(tensor)

    print(f"[{args.label}] hashing loaded checkpoint parameters", flush=True)
    manifest["parameters"]["mem_tokens"] = tensor_signature(metanetwork.metamodel.model.mem_tokens)
    manifest["parameters"]["metanetwork"] = named_parameter_signatures(metanetwork.metanetwork)
    manifest["parameters"]["metalora"] = nested_tensor_signatures(metalora)
    if args.hash_backbone_parameters:
        manifest["parameters"]["backbone"] = named_parameter_signatures(metanetwork.metamodel)

    layer_signatures: dict[str, Any] = {}
    layer_outputs: dict[str, Any] = {}
    hooks = []

    def layer_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = extract_hidden(output)
            key = f"decoder_layer/{index:02d}"
            layer_signatures[key] = tensor_signature(hidden)
            memory_tail = hidden[:, -int(cfg.num_mem_token) :, :]
            layer_outputs[f"{key}/memory_tail"] = cpu_tensor(memory_tail)
            if args.save_full_layer_outputs:
                layer_outputs[f"{key}/full_hidden"] = cpu_tensor(hidden)

        return hook

    for index, layer in enumerate(metanetwork.metamodel.model.layers):
        hooks.append(layer.register_forward_hook(layer_hook(index)))

    print(f"[{args.label}] context encoder forward and decoder-layer capture", flush=True)
    with torch.no_grad():
        context_output = metanetwork.metamodel(
            input_ids=encoded["evidence_ids"],
            attention_mask=encoded["evidence_attention_mask"],
            loradict=metalora,
            use_gradient_checkpoint=False,
        )
    for hook in hooks:
        hook.remove()
    memory_states = context_output.memory_states
    manifest["stages"]["decoder_layers"] = layer_signatures
    manifest["stages"]["memory_states"] = tensor_signature(memory_states)
    artifacts.update(layer_outputs)
    artifacts["stage/memory_states"] = cpu_tensor(memory_states)

    hyper_signatures: dict[str, Any] = {}
    hooks = []

    def hyper_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = extract_hidden(output)
            hyper_signatures[name] = tensor_signature(hidden)
            if args.save_full_hyper_outputs:
                artifacts[f"stage/{name}"] = cpu_tensor(hidden)

        return hook

    for index, layer in enumerate(metanetwork.metanetwork.transformer_layers):
        name = f"hyper_transformer_layer/{index:02d}"
        hooks.append(layer.register_forward_hook(hyper_hook(name)))
    for index, layer in enumerate(metanetwork.metanetwork.couple_layers):
        name = f"hyper_couple_layer/{index:02d}"
        hooks.append(layer.register_forward_hook(hyper_hook(name)))

    print(f"[{args.label}] hypernetwork forward and generated-LoRA capture", flush=True)
    with torch.no_grad():
        plain_output = metanetwork.metanetwork(memory_states)
        generated_lora = metanetwork.metamodel.generate_lora_dict(
            args.lora_r,
            scale=args.generated_lora_scale,
            plain_tensor=plain_output,
        )
    for hook in hooks:
        hook.remove()
    manifest["stages"]["hyper_layers"] = hyper_signatures
    manifest["stages"]["plain_output"] = tensor_signature(plain_output)
    artifacts["stage/plain_output"] = cpu_tensor(plain_output)
    generated_flat = flatten_tensors(generated_lora)
    manifest["stages"]["generated_lora"] = {
        name: tensor_signature(tensor) for name, tensor in generated_flat.items()
    }
    for name, tensor in generated_flat.items():
        artifacts[f"generated_lora/{name}"] = cpu_tensor(tensor)

    projection_signatures: dict[str, Any] = {}
    hooks = []

    def projection_hook(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            projection_signatures[name] = tensor_signature(extract_hidden(output))

        return hook

    for name, module in metanetwork.metamodel.named_modules():
        if module.__class__.__name__ == "LoraLinear":
            hooks.append(module.register_forward_hook(projection_hook(name)))

    print(f"[{args.label}] teacher-forced logits with and without generated LoRA", flush=True)
    with torch.no_grad():
        with_lora = metanetwork.metamodel(
            input_ids=encoded["full_ids"],
            attention_mask=encoded["full_attention_mask"],
            loradict=generated_lora,
            ignore_mem_token=True,
            use_gradient_checkpoint=False,
        ).logits
    for hook in hooks:
        hook.remove()
    manifest["stages"]["lora_projection_outputs"] = projection_signatures
    with torch.no_grad():
        without_lora = metanetwork.metamodel(
            input_ids=encoded["full_ids"],
            attention_mask=encoded["full_attention_mask"],
            loradict=None,
            ignore_mem_token=True,
            use_gradient_checkpoint=False,
        ).logits
    shifted_mask = encoded["labels"][:, 1:].ne(-100)
    supervised_with = with_lora[:, :-1, :][shifted_mask]
    supervised_without = without_lora[:, :-1, :][shifted_mask]
    manifest["stages"]["supervised_logits_with_lora"] = tensor_signature(supervised_with)
    manifest["stages"]["supervised_logits_without_lora"] = tensor_signature(supervised_without)
    artifacts["stage/supervised_logits_with_lora"] = cpu_tensor(supervised_with)
    artifacts["stage/supervised_logits_without_lora"] = cpu_tensor(supervised_without)

    print(f"[{args.label}] deterministic generation with and without generated LoRA", flush=True)
    generation_results = {}
    for name, lora in (("with_lora", generated_lora), ("without_lora", None)):
        with torch.no_grad():
            generated = metanetwork.metamodel.generate(
                input_ids=encoded["prompt_ids"],
                attention_mask=encoded["prompt_attention_mask"],
                loradict=lora,
                ignore_mem_token=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[:, encoded["prompt_ids"].shape[1] :]
        artifacts[f"stage/generation_{name}_ids"] = cpu_tensor(new_ids)
        generation_results[name] = {
            "signature": tensor_signature(new_ids),
            "ids": new_ids[0].detach().cpu().tolist(),
            "text": tokenizer.decode(new_ids[0], skip_special_tokens=True),
        }
    manifest["stages"]["generation"] = generation_results

    torch.save(artifacts, output_dir / "tensors.pt")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{args.label}] wrote {output_dir / 'manifest.json'}", flush=True)


def signature_hashes(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if "sha256" in value and "shape" in value:
            yield "/".join(prefix), str(value["sha256"])
        else:
            for key in sorted(value):
                yield from signature_hashes(value[key], prefix + (str(key),))


def compare_tensor(left: Any, right: Any, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
    }
    if left.shape != right.shape:
        result.update({"exact": False, "allclose": False, "reason": "shape_mismatch"})
        return result
    exact = torch.equal(left, right)
    result["exact"] = bool(exact)
    if not (left.is_floating_point() or right.is_floating_point()):
        mismatch = left.ne(right).flatten()
        result["allclose"] = bool(exact)
        result["mismatch_count"] = int(mismatch.sum())
        if mismatch.any():
            index = int(mismatch.nonzero()[0])
            result["first_mismatch_flat_index"] = index
            result["left_value"] = int(left.flatten()[index])
            result["right_value"] = int(right.flatten()[index])
        return result

    left_float = left.float()
    right_float = right.float()
    close = torch.isclose(left_float, right_float, atol=atol, rtol=rtol, equal_nan=True)
    diff = (left_float - right_float).abs()
    result.update(
        {
            "allclose": bool(close.all()),
            "mismatch_count": int((~close).sum()),
            "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
            "mean_abs_diff": float(diff.mean()) if diff.numel() else 0.0,
            "rms_diff": float(diff.square().mean().sqrt()) if diff.numel() else 0.0,
        }
    )
    denominator = right_float.abs().clamp_min(1e-12)
    result["max_rel_diff"] = float((diff / denominator).max()) if diff.numel() else 0.0
    if not close.all():
        index = int((~close).flatten().nonzero()[0])
        result["first_mismatch_flat_index"] = index
        result["left_value"] = float(left_float.flatten()[index])
        result["right_value"] = float(right_float.flatten()[index])
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    official_dir = Path(args.official_artifacts).resolve()
    current_dir = Path(args.current_artifacts).resolve()
    official_manifest = json.loads((official_dir / "manifest.json").read_text(encoding="utf-8"))
    current_manifest = json.loads((current_dir / "manifest.json").read_text(encoding="utf-8"))
    official_tensors = torch.load(official_dir / "tensors.pt", map_location="cpu", weights_only=False)
    current_tensors = torch.load(current_dir / "tensors.pt", map_location="cpu", weights_only=False)

    official_hashes = dict(signature_hashes(official_manifest))
    current_hashes = dict(signature_hashes(current_manifest))
    all_hash_names = sorted(set(official_hashes) | set(current_hashes))
    hash_comparisons = {
        name: {
            "official": official_hashes.get(name),
            "current": current_hashes.get(name),
            "exact": official_hashes.get(name) == current_hashes.get(name),
        }
        for name in all_hash_names
    }

    all_tensor_names = sorted(set(official_tensors) | set(current_tensors))
    tensor_comparisons: dict[str, Any] = {}
    for name in all_tensor_names:
        if name not in official_tensors or name not in current_tensors:
            tensor_comparisons[name] = {
                "exact": False,
                "allclose": False,
                "reason": "missing_tensor",
                "official_present": name in official_tensors,
                "current_present": name in current_tensors,
            }
            continue
        tensor_comparisons[name] = compare_tensor(
            official_tensors[name], current_tensors[name], args.atol, args.rtol
        )

    first_hash_mismatch = next((name for name, row in hash_comparisons.items() if not row["exact"]), None)
    first_tensor_mismatch = next(
        (name for name, row in tensor_comparisons.items() if not row.get("allclose", False)), None
    )
    execution_prefixes = (
        "inputs/",
        "parameters/mem_tokens",
        "parameters/metanetwork/",
        "parameters/metalora/",
        "parameters/backbone/",
        "stages/decoder_layers/",
        "stages/memory_states",
        "stages/hyper_layers/",
        "stages/plain_output",
        "stages/generated_lora/",
        "stages/lora_projection_outputs/",
        "stages/supervised_logits_with_lora",
        "stages/supervised_logits_without_lora",
        "stages/generation/",
    )
    mismatched_hashes = {name for name, row in hash_comparisons.items() if not row["exact"]}
    first_runtime_divergence = None
    for prefix in execution_prefixes:
        candidates = sorted(name for name in mismatched_hashes if name.startswith(prefix))
        if candidates:
            first_runtime_divergence = candidates[0]
            break
    report = {
        "official_manifest": str(official_dir / "manifest.json"),
        "current_manifest": str(current_dir / "manifest.json"),
        "atol": args.atol,
        "rtol": args.rtol,
        "summary": {
            "hash_entries": len(hash_comparisons),
            "exact_hash_entries": sum(row["exact"] for row in hash_comparisons.values()),
            "tensor_entries": len(tensor_comparisons),
            "exact_tensor_entries": sum(row.get("exact", False) for row in tensor_comparisons.values()),
            "allclose_tensor_entries": sum(row.get("allclose", False) for row in tensor_comparisons.values()),
            "first_hash_mismatch": first_hash_mismatch,
            "first_runtime_divergence": first_runtime_divergence,
            "first_tensor_mismatch": first_tensor_mismatch,
            "parity": first_hash_mismatch is None and first_tensor_mismatch is None,
        },
        "metadata": {
            "official_git_revision": official_manifest.get("source_git_revision"),
            "current_git_revision": current_manifest.get("source_git_revision"),
            "official_tokenizer": official_manifest.get("tokenizer"),
            "current_tokenizer": current_manifest.get("tokenizer"),
            "official_generation": official_manifest.get("stages", {}).get("generation"),
            "current_generation": current_manifest.get("stages", {}).get("generation"),
        },
        "hash_comparisons": hash_comparisons,
        "tensor_comparisons": tensor_comparisons,
    }
    output = Path(args.report).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Full report: {output}")
    return report


def capture_command(args: argparse.Namespace, label: str, source_root: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "capture",
        "--label",
        label,
        "--source-root",
        str(source_root),
        "--model-path",
        args.model_path,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--output-dir",
        str(output_dir),
        "--context",
        args.context,
        "--context-max-length",
        str(args.context_max_length),
        "--answer-max-length",
        str(args.answer_max_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--lora-r",
        str(args.lora_r),
        "--metalora-r",
        str(args.metalora_r),
        "--metanetwork-layers",
        str(args.metanetwork_layers),
        "--hyper-nhead",
        str(args.hyper_nhead),
        "--hyper-ffn-dim",
        str(args.hyper_ffn_dim),
        "--generated-lora-scale",
        str(args.generated_lora_scale),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    for enabled, flag in (
        (args.hash_backbone_parameters, "--hash-backbone-parameters"),
        (args.save_full_layer_outputs, "--save-full-layer-outputs"),
        (args.save_full_hyper_outputs, "--save-full-hyper-outputs"),
    ):
        if enabled:
            command.append(flag)
    return command


def run(args: argparse.Namespace) -> None:
    official_root = Path(args.official_root).resolve()
    current_root = Path(args.current_root).resolve()
    output_root = Path(args.output_dir).resolve()
    if official_root == current_root:
        raise ValueError("--official-root and --current-root must be different source trees")
    for root in (official_root, current_root):
        if not (root / "LoraQwen.py").is_file():
            raise FileNotFoundError(f"Not a SHINE source root: {root}")
    output_root.mkdir(parents=True, exist_ok=True)
    static_manifest = {
        "official_root": str(official_root),
        "current_root": str(current_root),
        "official_git_revision": git_revision(official_root),
        "current_git_revision": git_revision(current_root),
        "official_files": source_file_manifest(official_root),
        "current_files": source_file_manifest(current_root),
    }
    (output_root / "source_manifest.json").write_text(
        json.dumps(static_manifest, indent=2), encoding="utf-8"
    )
    write_source_diff(official_root, current_root, output_root / "source_diff.patch")

    official_output = output_root / "official"
    current_output = output_root / "current"
    print("Running official capture in an isolated process", flush=True)
    subprocess.run(capture_command(args, "official", official_root, official_output), check=True)
    print("Running current capture in an isolated process", flush=True)
    subprocess.run(capture_command(args, "current", current_root, current_output), check=True)
    compare_args = argparse.Namespace(
        official_artifacts=str(official_output),
        current_artifacts=str(current_output),
        report=str(output_root / "parity_report.json"),
        atol=args.atol,
        rtol=args.rtol,
    )
    report = compare(compare_args)
    if not report["summary"]["parity"]:
        raise SystemExit(2)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--context", default="David plays chess.")
    parser.add_argument("--context-max-length", type=int, default=1141)
    parser.add_argument("--answer-max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--metalora-r", type=int, default=128)
    parser.add_argument("--metanetwork-layers", type=int, default=4)
    parser.add_argument("--hyper-nhead", type=int, default=32)
    parser.add_argument("--hyper-ffn-dim", type=int, default=8192)
    parser.add_argument("--generated-lora-scale", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hash-backbone-parameters", action="store_true")
    parser.add_argument("--save-full-layer-outputs", action="store_true")
    parser.add_argument("--save-full-hyper-outputs", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Capture official/current sequentially and compare")
    run_parser.add_argument("--official-root", required=True)
    run_parser.add_argument("--current-root", default=str(Path(__file__).resolve().parents[2]))
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--atol", type=float, default=0.0)
    run_parser.add_argument("--rtol", type=float, default=0.0)
    add_runtime_arguments(run_parser)
    run_parser.set_defaults(handler=run)

    capture_parser = subparsers.add_parser("capture", help="Capture one source tree")
    capture_parser.add_argument("--label", required=True)
    capture_parser.add_argument("--source-root", required=True)
    capture_parser.add_argument("--output-dir", required=True)
    add_runtime_arguments(capture_parser)
    capture_parser.set_defaults(handler=capture)

    compare_parser = subparsers.add_parser("compare", help="Compare two existing captures")
    compare_parser.add_argument("--official-artifacts", required=True)
    compare_parser.add_argument("--current-artifacts", required=True)
    compare_parser.add_argument("--report", required=True)
    compare_parser.add_argument("--atol", type=float, default=0.0)
    compare_parser.add_argument("--rtol", type=float, default=0.0)
    compare_parser.set_defaults(handler=compare)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
