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
from analysis.extract_attention import (
    extract_attention_maps,
    extract_siglip_attention_maps,
    get_image_token_attention,
    get_siglip_patch_saliency,
)
from analysis.visualize_attention import visualize_all_layers
from bitvla.constants import BITNET_DEFAULT_IMAGE_TOKEN_IDX


def analyze_condition(stack, label, suite, mode, task_id, layer_indices, capture_kwargs,
                      include_siglip=False, siglip_layer_indices=None):
    print(f"\n{'=' * 60}\n[{label}] suite={suite} mode={mode} task={task_id}\n{'=' * 60}")
    image, task_desc, task_name, bboxes, meta = common.capture_observation(
        suite, task_id, mode, stack=stack, **capture_kwargs
    )
    inputs = common.prepare_inputs(stack.model, stack.processor, image, task_desc)

    print("Extracting LLM attention maps...")
    attention_maps = extract_attention_maps(stack.model, inputs, layer_indices=layer_indices)
    image_attention = get_image_token_attention(
        attention_maps,
        inputs["input_ids"],
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    )

    siglip_attention = None
    if include_siglip:
        print("Extracting SigLIP attention maps...")
        siglip_maps = extract_siglip_attention_maps(
            stack.model, inputs, layer_indices=siglip_layer_indices
        )
        siglip_attention = get_siglip_patch_saliency(siglip_maps)

    return {
        "label": label,
        "image": np.array(image),
        "attention": image_attention,
        "siglip_attention": siglip_attention,
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


def save_per_condition(result, png_dir, json_dir, task_id):
    visualize_all_layers(
        result["image"],
        result["attention"],
        task_label=f"{result['task_description']} [{result['label']}] (LLM)",
        output_path=os.path.join(png_dir, f"layers_{result['label']}_task{task_id}.png"),
        bboxes=result["bboxes"],
    )

    ratios = compute_bbox_attention_ratios(
        result["attention"], result["bboxes"], result["image"].shape
    )
    payload = {
        "label": result["label"],
        "task_description": result["task_description"],
        "meta": result["meta"],
        "bboxes": result["bboxes"],
        "per_layer_ratios_llm": ratios,
    }

    if result.get("siglip_attention"):
        visualize_all_layers(
            result["image"],
            result["siglip_attention"],
            task_label=f"{result['task_description']} [{result['label']}] (SigLIP)",
            output_path=os.path.join(png_dir, f"layers_siglip_{result['label']}_task{task_id}.png"),
            bboxes=result["bboxes"],
        )
        siglip_ratios = compute_bbox_attention_ratios(
            result["siglip_attention"], result["bboxes"], result["image"].shape
        )
        payload["per_layer_ratios_siglip"] = siglip_ratios

    scores_path = os.path.join(json_dir, f"scores_{result['label']}_task{task_id}.json")
    with open(scores_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="BitVLA attention map + score analysis")
    common.add_common_args(parser)
    parser.add_argument("--include_siglip", action="store_true",
                        help="Also extract SigLIP patch-to-patch self-attention.")
    parser.add_argument("--siglip_layers", type=int, nargs="*", default=None,
                        help="SigLIP layer indices (default: every 5th + last of 26).")
    args = parser.parse_args()

    common.set_device(args.device)
    png_dir, json_dir = common.resolve_output_dirs(args.output_dir, args.output_subdir)
    print(f"Output dirs:\n  png:  {png_dir}\n  json: {json_dir}")

    conditions = common.parse_conditions(args.condition)
    first_suite = conditions[0][1]
    stack = common.load_rollout_stack(
        args.checkpoint_path,
        use_int2=args.use_int2_quantization,
        task_suite_name=first_suite,
    )

    if args.layers is None:
        layer_indices, num_layers = common.default_layer_indices(stack.model)
        print(f"Analyzing LLM layers: {layer_indices} (of {num_layers})")
    else:
        layer_indices = args.layers

    siglip_layer_indices = args.siglip_layers
    if args.include_siglip and siglip_layer_indices is None:
        from transformers.models.siglip.modeling_siglip import SiglipAttention
        n_siglip = sum(1 for _, m in stack.model.named_modules() if isinstance(m, SiglipAttention))
        siglip_layer_indices = list(range(0, n_siglip, 5))
        if (n_siglip - 1) not in siglip_layer_indices:
            siglip_layer_indices.append(n_siglip - 1)
        print(f"Analyzing SigLIP layers: {siglip_layer_indices} (of {n_siglip})")

    capture_kwargs = dict(
        rollout_max_steps=args.rollout_max_steps,
        rollout_seed_candidates=tuple(args.rollout_seed_candidates),
        mid_rollout_step=args.mid_rollout_step,
    )

    for label, suite, mode in conditions:
        result = analyze_condition(
            stack, label, suite, mode, args.task_id, layer_indices, capture_kwargs,
            include_siglip=args.include_siglip,
            siglip_layer_indices=siglip_layer_indices,
        )
        save_per_condition(result, png_dir, json_dir, args.task_id)

    print(f"\nAttention outputs saved under: {os.path.dirname(png_dir)}")


if __name__ == "__main__":
    main()
