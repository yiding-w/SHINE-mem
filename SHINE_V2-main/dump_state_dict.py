"""
Dump the state dict summary of a model from safetensors files.
Uses only metadata (no tensor loading) for speed.
"""

import os
import sys
import json
import struct
import argparse


def read_safetensors_metadata(filepath):
    """Read safetensors metadata without loading tensors."""
    with open(filepath, "rb") as f:
        # First 8 bytes: length of header (uint64 LE)
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        # Read header JSON
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
    return header


def dtype_str(dtype_name):
    """Convert safetensors dtype name to torch-like dtype string."""
    mapping = {
        "F16": "torch.float16",
        "BF16": "torch.bfloat16",
        "F32": "torch.float32",
        "F64": "torch.float64",
        "I8": "torch.int8",
        "I16": "torch.int16",
        "I32": "torch.int32",
        "I64": "torch.int64",
        "U8": "torch.uint8",
        "BOOL": "torch.bool",
    }
    return mapping.get(dtype_name, dtype_name)


def dump_state_dict(model_path, output_path):
    print(f"Loading model metadata from: {model_path}")
    print(f"Output will be saved to: {output_path}")

    # Find all safetensor files
    safetensor_files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    ])

    if not safetensor_files:
        print("ERROR: No safetensors files found!")
        sys.exit(1)

    print(f"Found {len(safetensor_files)} safetensor shard(s)")

    # Collect all tensor metadata
    all_tensors = {}
    for sf_file in safetensor_files:
        sf_path = os.path.join(model_path, sf_file)
        print(f"  Reading metadata from {sf_file}...")
        header = read_safetensors_metadata(sf_path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            all_tensors[key] = {
                "shape": tuple(meta["shape"]),
                "dtype": dtype_str(meta["dtype"]),
            }

    # Sort keys for consistent output
    sorted_keys = sorted(all_tensors.keys())

    print(f"\nTotal parameters: {len(sorted_keys)}")

    # Write output
    with open(output_path, "w") as out:
        out.write(f"=== STATE DICT SUMMARY ===\n")
        out.write(f"Model path: {model_path}\n")
        out.write(f"Total keys: {len(sorted_keys)}\n\n")

        for key in sorted_keys:
            info = all_tensors[key]
            out.write(f"{key}: shape={info['shape']}, dtype={info['dtype']}\n")

        out.write(f"\n=== END OF STATE DICT ===\n")

    print(f"State dict summary saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump model state dict summary from safetensors")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    args = parser.parse_args()

    dump_state_dict(args.model_path, args.output)
