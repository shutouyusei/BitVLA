"""
Offline quantizer for BitVLA models.

Converts BitLinear layers from bf16 master weights to 2-bit packed format
using the existing quantize_weights() method in BitLinear.

Usage:
    python quantize_model.py --checkpoint_path /path/to/bitvla-bf16 --output_path /path/to/bitvla-int2
"""

import argparse
import os
import shutil
import sys

import torch
from transformers import AutoModelForVision2Seq

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers.models.llava.modeling_bitnet import BitLinear


def quantize_and_save(checkpoint_path: str, output_path: str) -> None:
    """
    Load a BitVLA model in bf16, quantize all BitLinear layers
    to 2-bit packed format, and save the quantized model.

    BitLinear.quantize_weights() does:
    1. Pack bf16 weights into 2-bit (4 values per byte)
    2. Store scale factor (w_step)
    3. Delete original bf16 weight

    This reduces ~5.7GB -> ~1.5GB on GPU.
    """
    print(f"Loading model from: {checkpoint_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Quantizing BitLinear layers...")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and not module.enable_qlora:
            module.quantize_weights()
            count += 1
            print(f"  Quantized: {name}")

    print(f"\nQuantized {count} BitLinear layers")

    print(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path)

    # Copy tokenizer and processor files
    for f in os.listdir(checkpoint_path):
        if f.endswith((".json", ".txt", ".model", ".py")) and not os.path.exists(
            os.path.join(output_path, f)
        ):
            shutil.copy2(os.path.join(checkpoint_path, f), output_path)

    print(f"Done! Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize BitVLA model to 2-bit packed format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to bf16 model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save quantized model")
    args = parser.parse_args()
    quantize_and_save(args.checkpoint_path, args.output_path)
