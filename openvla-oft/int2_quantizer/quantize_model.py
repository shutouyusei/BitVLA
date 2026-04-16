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
from int2_quantizer import quantize_bitlinear_layers


def quantize_and_save(checkpoint_path: str, output_path: str) -> None:
    """
    Load a BitVLA model in bf16, quantize all BitLinear layers
    to 2-bit packed format, and save the quantized model.

    Reduces weight memory by ~8x for BitLinear layers.
    Non-BitLinear layers (embeddings, vision encoder, connector) remain in bf16.

    Side effects: Writes quantized model weights and copies
    tokenizer/processor files to output_path.
    """
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist or is not a directory: '{checkpoint_path}'"
        )
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading model from: {checkpoint_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Quantizing BitLinear layers...")
    count = quantize_bitlinear_layers(model)
    print(f"Quantized {count} BitLinear layers to 2-bit")

    print(f"Saving quantized model to: {output_path}")
    model.save_pretrained(output_path)

    # Copy tokenizer and processor files not already written by save_pretrained
    try:
        for f in os.listdir(checkpoint_path):
            if f.endswith((".json", ".txt", ".model", ".py")) and not os.path.exists(
                os.path.join(output_path, f)
            ):
                shutil.copy2(os.path.join(checkpoint_path, f), output_path)
    except OSError as e:
        print(
            f"WARNING: Model weights saved successfully but failed to copy "
            f"auxiliary files from '{checkpoint_path}': {e}"
        )
        print(f"You may need to manually copy tokenizer/processor files to '{output_path}'")

    print(f"Done! Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize BitVLA model to 2-bit packed format")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to bf16 model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save quantized model")
    args = parser.parse_args()
    quantize_and_save(args.checkpoint_path, args.output_path)
