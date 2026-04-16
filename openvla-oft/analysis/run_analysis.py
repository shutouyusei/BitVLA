"""
End-to-end attention analysis for BitVLA on LIBERO tasks.

Loads the model, creates a LIBERO environment, captures observations,
extracts attention maps, and generates visualizations.

Usage:
    cd openvla-oft
    python analysis/run_analysis.py \
        --checkpoint_path /path/to/bitvla \
        --task_suite original:libero_spatial \
        --task_suite perturbed:libero_spatial_swap \
        --task_id 0 \
        --output_dir ./analysis_output \
        --use_int2_quantization
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from analysis.extract_attention import extract_attention_maps, get_image_token_attention
from analysis.extract_hidden_states import extract_hidden_states, get_token_types
from analysis.visualize_attention import (
    compare_attention_maps,
    visualize_all_layers,
    visualize_attention_on_image,
)
from analysis.visualize_tsne import compare_tsne, overlay_tsne
from bitvla.constants import BITNET_DEFAULT_IMAGE_TOKEN_IDX
from int2_quantizer import quantize_bitlinear_layers


DEVICE = torch.device("cpu")  # Use CPU to avoid VRAM conflicts with other processes


def load_model(checkpoint_path, use_int2=False):
    """Load BitVLA model, optionally with int2 quantization."""
    from transformers import AutoModelForVision2Seq, AutoConfig, LlavaProcessor, SiglipImageProcessor
    from bitvla import Bitvla_Config, BitVLAForActionPrediction
    from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch

    AutoConfig.register("bitvla", Bitvla_Config)
    AutoModelForVision2Seq.register(Bitvla_Config, BitVLAForActionPrediction)

    update_auto_map(checkpoint_path)
    check_model_logic_mismatch(
        checkpoint_path,
        curr_files={"bitvla_for_action_prediction.py": None, "configuration_bit_vla.py": None},
        where_to_find_files_cur_codebase="./bitvla",
    )

    print(f"Loading model from: {checkpoint_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if use_int2:
        count = quantize_bitlinear_layers(model)
        print(f"Quantized {count} BitLinear layers to 2-bit at load time")

    model = model.to(DEVICE)
    model.eval()

    from bitvla.constants import (
        BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        BITNET_PROPRIO_PAD_IDX,
        BITNET_IGNORE_INDEX,
        BITNET_ACTION_TOKEN_BEGIN_IDX,
        BITNET_STOP_INDEX,
    )
    model.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )

    processor = LlavaProcessor.from_pretrained(checkpoint_path)

    return model, processor


def get_libero_observation(task_suite_name, task_id=0, seed=7):
    """Create a LIBERO environment and capture the initial observation image."""
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language

    initial_states = task_suite.get_task_init_states(task_id)
    env, _ = get_libero_env(task, model_family="bitnet", resolution=256)
    env.seed(seed)
    env.reset()
    env.set_init_state(initial_states[0])
    obs = env.step(np.zeros(7))[0]

    image = get_libero_image(obs)

    env.close()
    return image, task_description, task_name


def prepare_inputs(model, processor, image, task_label):
    """Prepare model inputs from an image and task label."""
    import copy
    from transformers.image_utils import get_image_size, to_numpy_array
    from bitvla.constants import BITNET_DEFAULT_IMAGE_TOKEN
    from bitvla.dataset.bitvla_transform import llava_to_openai

    all_images = [image]
    pixel_values = [
        processor.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
        for img in all_images
    ]
    patch_size = processor.patch_size
    num_image_tokens = []
    for img in pixel_values:
        height, width = get_image_size(to_numpy_array(img))
        num_image_tokens.append((height // patch_size) * (width // patch_size))

    sources = {
        "conversations": [
            {"from": "human", "value": f"<image>\n<proprio_pad>What action should the robot take to {task_label.lower()}?"},
            {"from": "gpt", "value": ""},
        ]
    }
    sources = copy.deepcopy(llava_to_openai(sources["conversations"]))
    prompt = sources[0]["content"]
    x = [{"role": "user", "content": prompt}]
    input_str = processor.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)

    placeholder = "<TEMP_IMAGE_TOKEN>"
    input_str = input_str.replace(BITNET_DEFAULT_IMAGE_TOKEN, placeholder)
    token_index = 0
    while placeholder in input_str and token_index < len(num_image_tokens):
        input_str = input_str.replace(
            placeholder, BITNET_DEFAULT_IMAGE_TOKEN * num_image_tokens[token_index], 1
        )
        token_index += 1

    input_ids = processor.tokenizer(input_str, add_special_tokens=True).input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pv = torch.stack(pixel_values, dim=0).unsqueeze(0)

    inputs = dict(
        input_ids=input_ids.to(DEVICE, dtype=torch.long),
        attention_mask=attention_mask.to(DEVICE),
        pixel_values=pv.to(DEVICE, dtype=torch.bfloat16),
    )
    return inputs


def run_attention_analysis(
    checkpoint_path,
    task_suites,
    task_id=0,
    output_dir="./analysis_output",
    use_int2=False,
    layer_indices=None,
):
    """
    End-to-end attention analysis.

    Args:
        checkpoint_path: path to BitVLA checkpoint
        task_suites: dict like {"original": "libero_spatial", "perturbed": "libero_spatial_swap"}
        task_id: which task in the suite (0-9)
        output_dir: where to save visualizations
        use_int2: whether to use int2 quantization
        layer_indices: which layers to analyze (default: every 5th layer)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    model, processor = load_model(checkpoint_path, use_int2=use_int2)

    if layer_indices is None:
        # Sample layers: first, every 5th, and last
        from transformers.models.llava.modeling_bitnet import BitNetAttention
        num_layers = sum(1 for _, m in model.named_modules() if isinstance(m, BitNetAttention))
        layer_indices = list(range(0, num_layers, 5))
        if (num_layers - 1) not in layer_indices:
            layer_indices.append(num_layers - 1)
        print(f"Analyzing layers: {layer_indices} (of {num_layers} total)")

    results = {}
    for label, suite_name in task_suites.items():
        print(f"\n{'='*60}")
        print(f"Capturing observation from: {suite_name} (task {task_id})")
        print(f"{'='*60}")

        image, task_description, task_name = get_libero_observation(suite_name, task_id)
        inputs = prepare_inputs(model, processor, image, task_description)

        print(f"Task: {task_description}")
        print(f"Extracting attention maps...")
        attention_maps = extract_attention_maps(model, inputs, layer_indices=layer_indices)
        image_attention = get_image_token_attention(
            attention_maps,
            inputs["input_ids"],
            image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        )

        # Extract hidden states for t-SNE
        print(f"Extracting hidden states...")
        hidden_states = extract_hidden_states(model, inputs, layer_indices=layer_indices)

        from bitvla.constants import (
            BITNET_PROPRIO_PAD_IDX,
            BITNET_ACTION_TOKEN_BEGIN_IDX,
        )
        token_types = get_token_types(
            inputs["input_ids"],
            image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
            proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
            action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        )

        results[label] = {
            "image": np.array(image),
            "attention": image_attention,
            "hidden_states": hidden_states,
            "token_types": token_types,
            "task_description": task_description,
            "task_name": task_name,
        }

        # Save per-suite all-layers visualization
        visualize_all_layers(
            np.array(image),
            image_attention,
            task_label=f"{task_description} ({label})",
            output_path=os.path.join(output_dir, f"layers_{label}_task{task_id}.png"),
        )

    # If we have both original and perturbed, generate comparison
    if "original" in results and "perturbed" in results:
        for layer_idx in layer_indices:
            if layer_idx in results["original"]["attention"] and layer_idx in results["perturbed"]["attention"]:
                compare_attention_maps(
                    results["original"]["image"],
                    results["perturbed"]["image"],
                    results["original"]["attention"][layer_idx],
                    results["perturbed"]["attention"][layer_idx],
                    layer_idx=layer_idx,
                    task_label=results["original"]["task_description"],
                    output_path=os.path.join(output_dir, f"compare_layer{layer_idx}_task{task_id}.png"),
                )

    # t-SNE comparison if we have both original and perturbed
    if "original" in results and "perturbed" in results:
        print("\nGenerating t-SNE visualizations...")
        compare_tsne(
            results["original"]["hidden_states"],
            results["perturbed"]["hidden_states"],
            results["original"]["token_types"],
            results["perturbed"]["token_types"],
            task_label=results["original"]["task_description"],
            output_path=os.path.join(output_dir, f"tsne_compare_task{task_id}.png"),
        )
        overlay_tsne(
            results["original"]["hidden_states"],
            results["perturbed"]["hidden_states"],
            results["original"]["token_types"],
            task_label=results["original"]["task_description"],
            output_path=os.path.join(output_dir, f"tsne_overlay_task{task_id}.png"),
        )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BitVLA attention analysis on LIBERO tasks")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--task_suite",
        type=str,
        nargs="+",
        required=True,
        help="Format: label:suite_name (e.g., original:libero_spatial perturbed:libero_spatial_swap)",
    )
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./analysis_output")
    parser.add_argument("--use_int2_quantization", action="store_true")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Layer indices to analyze")
    args = parser.parse_args()

    task_suites = {}
    for ts in args.task_suite:
        label, suite_name = ts.split(":")
        task_suites[label] = suite_name

    run_attention_analysis(
        checkpoint_path=args.checkpoint_path,
        task_suites=task_suites,
        task_id=args.task_id,
        output_dir=args.output_dir,
        use_int2=args.use_int2_quantization,
        layer_indices=args.layers,
    )
