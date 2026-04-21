"""Attention analysis entry point for BitVLA on LIBERO tasks.

Usage:
    cd openvla-oft
    python -m analysis.run_attention \
        --checkpoint_path /path/to/bitvla \
        --condition normal_success:libero_spatial \
        --condition normal_failed:libero_spatial \
        --condition pro_init:libero_spatial_swap \
        --condition pro_manip:libero_spatial_swap:mid_rollout \
        --task_id 0 --device cuda:0
"""

import argparse
import json
import os

import numpy as np

from analysis import common
from analysis.extract_attention import extract_attention_maps, get_image_token_attention
from analysis.visualize_attention import visualize_all_layers
from bitvla.constants import BITNET_DEFAULT_IMAGE_TOKEN_IDX


def analyze_condition(stack, label, suite, mode, task_id, layer_indices, capture_kwargs):
    print(f"\n{'=' * 60}\n[{label}] suite={suite} mode={mode} task={task_id}\n{'=' * 60}")
    image, task_desc, task_name, bboxes, meta = common.capture_observation(
        suite, task_id, mode, stack=stack, **capture_kwargs
    )
    inputs = common.prepare_inputs(stack.model, stack.processor, image, task_desc)

    print("Extracting attention maps...")
    attention_maps = extract_attention_maps(stack.model, inputs, layer_indices=layer_indices)
    image_attention = get_image_token_attention(
        attention_maps,
        inputs["input_ids"],
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    )

    return {
        "label": label,
        "image": np.array(image),
        "attention": image_attention,
        "bboxes": bboxes,
        "task_description": task_desc,
        "task_name": task_name,
        "meta": meta,
    }


def compute_bbox_attention_ratios(image_attention, bboxes, image_shape, patch_grid_size=16):
    """For each layer + object bbox, fraction of attention mass inside the bbox.

    Attention map has patch_grid_size**2 patches covering the image uniformly; we sum the
    patch values whose image-space rectangle overlaps the bbox, divided by the total sum.
    """
    H, W = image_shape[:2]
    scale_y = patch_grid_size / H
    scale_x = patch_grid_size / W
    results = {}
    for layer_idx, attn in image_attention.items():
        attn_grid = attn.float().cpu().numpy().reshape(patch_grid_size, patch_grid_size)
        total = float(attn_grid.sum())
        per_object = {}
        for b in bboxes:
            ymin, xmin, ymax, xmax = b["bbox"]
            gy0 = max(int(np.floor(ymin * scale_y)), 0)
            gy1 = min(int(np.ceil(ymax * scale_y)), patch_grid_size)
            gx0 = max(int(np.floor(xmin * scale_x)), 0)
            gx1 = min(int(np.ceil(xmax * scale_x)), patch_grid_size)
            if gy1 <= gy0 or gx1 <= gx0:
                ratio = 0.0
            else:
                inside = float(attn_grid[gy0:gy1, gx0:gx1].sum())
                ratio = inside / total if total > 0 else 0.0
            per_object[b["name"]] = {"ratio": ratio, "is_target": bool(b.get("is_target"))}
        results[int(layer_idx)] = per_object
    return results


def save_per_condition(result, output_dir, task_id):
    visualize_all_layers(
        result["image"],
        result["attention"],
        task_label=f"{result['task_description']} [{result['label']}]",
        output_path=os.path.join(output_dir, f"layers_{result['label']}_task{task_id}.png"),
        bboxes=result["bboxes"],
    )

    ratios = compute_bbox_attention_ratios(
        result["attention"], result["bboxes"], result["image"].shape
    )
    scores_path = os.path.join(output_dir, f"scores_{result['label']}_task{task_id}.json")
    with open(scores_path, "w") as f:
        json.dump({
            "label": result["label"],
            "task_description": result["task_description"],
            "meta": result["meta"],
            "bboxes": result["bboxes"],
            "per_layer_ratios": ratios,
        }, f, indent=2)
    print(f"Saved: {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="BitVLA attention map + score analysis")
    common.add_common_args(parser)
    args = parser.parse_args()

    common.set_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    conditions = common.parse_conditions(args.condition)
    first_suite = conditions[0][1]
    stack = common.load_rollout_stack(
        args.checkpoint_path,
        use_int2=args.use_int2_quantization,
        task_suite_name=first_suite,
    )

    if args.layers is None:
        layer_indices, num_layers = common.default_layer_indices(stack.model)
        print(f"Analyzing layers: {layer_indices} (of {num_layers})")
    else:
        layer_indices = args.layers

    capture_kwargs = dict(
        rollout_max_steps=args.rollout_max_steps,
        rollout_seed_candidates=tuple(args.rollout_seed_candidates),
        mid_rollout_step=args.mid_rollout_step,
    )

    for label, suite, mode in conditions:
        result = analyze_condition(
            stack, label, suite, mode, args.task_id, layer_indices, capture_kwargs
        )
        save_per_condition(result, args.output_dir, args.task_id)

    print(f"\nAttention outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
