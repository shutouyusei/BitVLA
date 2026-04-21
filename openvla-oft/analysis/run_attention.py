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
    get_image_token_attention_per_head,
    get_siglip_patch_saliency,
    get_siglip_patch_saliency_per_head,
)
from analysis.visualize_attention import visualize_all_layers
from bitvla.constants import BITNET_DEFAULT_IMAGE_TOKEN_IDX


def analyze_frame(stack, image, task_desc, layer_indices, include_siglip, siglip_layer_indices):
    inputs = common.prepare_inputs(stack.model, stack.processor, image, task_desc)

    attention_maps = extract_attention_maps(stack.model, inputs, layer_indices=layer_indices)
    image_attention = get_image_token_attention(
        attention_maps, inputs["input_ids"], image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    )
    image_attention_per_head = get_image_token_attention_per_head(
        attention_maps, inputs["input_ids"], image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    )

    siglip_attention = None
    siglip_attention_per_head = None
    if include_siglip:
        siglip_maps = extract_siglip_attention_maps(
            stack.model, inputs, layer_indices=siglip_layer_indices
        )
        siglip_attention = get_siglip_patch_saliency(siglip_maps)
        siglip_attention_per_head = get_siglip_patch_saliency_per_head(siglip_maps)

    return {
        "attention": image_attention,
        "attention_per_head": image_attention_per_head,
        "siglip_attention": siglip_attention,
        "siglip_attention_per_head": siglip_attention_per_head,
    }


def analyze_condition(stack, label, suite, mode, task_id, layer_indices, capture_kwargs,
                      include_siglip=False, siglip_layer_indices=None):
    print(f"\n{'=' * 60}\n[{label}] suite={suite} mode={mode} task={task_id}\n{'=' * 60}")
    capture = common.capture_frames(suite, task_id, mode, stack=stack, **capture_kwargs)

    frame_results = {}
    for frame_name, frame in capture["frames"].items():
        print(f"  frame {frame_name} (step={frame['step_idx']}, fallback={frame.get('fallback', False)})")
        attn = analyze_frame(
            stack, frame["image"], capture["task_description"],
            layer_indices, include_siglip, siglip_layer_indices,
        )
        frame_results[frame_name] = {
            "step_idx": frame["step_idx"],
            "fallback": frame.get("fallback", False),
            "image": frame["image"],
            "bboxes": frame["bboxes"],
            **attn,
        }

    return {
        "label": label,
        "task_description": capture["task_description"],
        "task_name": capture["task_name"],
        "meta": capture["meta"],
        "frames": frame_results,
    }


def _bbox_to_grid(bbox, H, W, grid):
    ymin, xmin, ymax, xmax = bbox
    gy0 = max(int(np.floor(ymin * grid / H)), 0)
    gy1 = min(int(np.ceil(ymax * grid / H)), grid)
    gx0 = max(int(np.floor(xmin * grid / W)), 0)
    gx1 = min(int(np.ceil(xmax * grid / W)), grid)
    return gy0, gy1, gx0, gx1


def compute_bbox_attention_ratios(image_attention, bboxes, image_shape, patch_grid_size=16):
    """Head-averaged per-layer, per-object attention mass ratio.

    Ratio denominator is attention summed over non-robot patches (so the
    robot never inflates the normalizer). Robot bboxes are still recorded
    with their raw mass for reference but flagged so downstream stats skip them.
    """
    H, W = image_shape[:2]
    results = {}
    for layer_idx, attn in image_attention.items():
        attn_grid = attn.float().cpu().numpy().reshape(patch_grid_size, patch_grid_size)
        # Exclude robot region from the denominator
        robot_mask = np.zeros_like(attn_grid, dtype=bool)
        for b in bboxes:
            if b.get("is_robot"):
                gy0, gy1, gx0, gx1 = _bbox_to_grid(b["bbox"], H, W, patch_grid_size)
                robot_mask[gy0:gy1, gx0:gx1] = True
        non_robot_total = float(attn_grid[~robot_mask].sum())
        per_object = {}
        for b in bboxes:
            gy0, gy1, gx0, gx1 = _bbox_to_grid(b["bbox"], H, W, patch_grid_size)
            if gy1 <= gy0 or gx1 <= gx0:
                ratio = 0.0
            elif b.get("is_robot"):
                ratio = float("nan")  # robot is excluded from target metric
            else:
                inside = float(attn_grid[gy0:gy1, gx0:gx1].sum())
                ratio = inside / non_robot_total if non_robot_total > 0 else 0.0
            per_object[b["name"]] = {
                "ratio": ratio,
                "is_target": bool(b.get("is_target")),
                "is_robot": bool(b.get("is_robot")),
            }
        results[int(layer_idx)] = per_object
    return results


def compute_bbox_attention_ratios_per_head(image_attention_per_head, bboxes, image_shape, patch_grid_size=16):
    """Per-head stats: for each layer + (non-robot) object, record max/std across heads.

    Auxiliary metric per Phase 1 design — surfaces whether a single head
    already localizes the target even when head-averaged attention is diffuse.
    """
    H, W = image_shape[:2]
    results = {}
    for layer_idx, attn_heads in image_attention_per_head.items():
        attn_np = attn_heads.float().cpu().numpy()  # (num_heads, num_patches)
        num_heads = attn_np.shape[0]
        # Build per-head robot-excluded denominators
        grid = patch_grid_size
        robot_mask = np.zeros((grid, grid), dtype=bool)
        for b in bboxes:
            if b.get("is_robot"):
                gy0, gy1, gx0, gx1 = _bbox_to_grid(b["bbox"], H, W, grid)
                robot_mask[gy0:gy1, gx0:gx1] = True
        per_object = {}
        for b in bboxes:
            if b.get("is_robot"):
                continue
            gy0, gy1, gx0, gx1 = _bbox_to_grid(b["bbox"], H, W, grid)
            if gy1 <= gy0 or gx1 <= gx0:
                per_object[b["name"]] = {"max": 0.0, "std": 0.0, "is_target": bool(b.get("is_target"))}
                continue
            ratios = []
            for h in range(num_heads):
                head_grid = attn_np[h].reshape(grid, grid)
                denom = float(head_grid[~robot_mask].sum())
                inside = float(head_grid[gy0:gy1, gx0:gx1].sum())
                ratios.append(inside / denom if denom > 0 else 0.0)
            ratios = np.asarray(ratios)
            per_object[b["name"]] = {
                "max": float(ratios.max()),
                "std": float(ratios.std()),
                "is_target": bool(b.get("is_target")),
            }
        results[int(layer_idx)] = per_object
    return results


def _frame_suffix(name):
    """Safe filename tag for a frame key like 't=max//3'."""
    return name.replace("/", "-").replace("=", "-")


def save_per_condition(result, png_dir, json_dir, task_id):
    frames_payload = {}
    for frame_name, fr in result["frames"].items():
        tag = _frame_suffix(frame_name)
        base_title = f"{result['task_description']} [{result['label']} / {frame_name}]"

        visualize_all_layers(
            fr["image"], fr["attention"],
            task_label=f"{base_title} (LLM)",
            output_path=os.path.join(png_dir, f"layers_{result['label']}_task{task_id}_{tag}.png"),
            bboxes=fr["bboxes"],
        )
        ratios = compute_bbox_attention_ratios(fr["attention"], fr["bboxes"], fr["image"].shape)
        per_head_stats = compute_bbox_attention_ratios_per_head(
            fr["attention_per_head"], fr["bboxes"], fr["image"].shape
        )
        frame_entry = {
            "step_idx": fr["step_idx"],
            "fallback": fr["fallback"],
            "bboxes": fr["bboxes"],
            "per_layer_ratios_llm": ratios,
            "per_layer_per_head_llm": per_head_stats,
        }

        if fr.get("siglip_attention"):
            visualize_all_layers(
                fr["image"], fr["siglip_attention"],
                task_label=f"{base_title} (SigLIP)",
                output_path=os.path.join(png_dir, f"layers_siglip_{result['label']}_task{task_id}_{tag}.png"),
                bboxes=fr["bboxes"],
            )
            frame_entry["per_layer_ratios_siglip"] = compute_bbox_attention_ratios(
                fr["siglip_attention"], fr["bboxes"], fr["image"].shape
            )
            frame_entry["per_layer_per_head_siglip"] = compute_bbox_attention_ratios_per_head(
                fr["siglip_attention_per_head"], fr["bboxes"], fr["image"].shape
            )

        frames_payload[frame_name] = frame_entry

    payload = {
        "label": result["label"],
        "task_description": result["task_description"],
        "task_name": result["task_name"],
        "meta": result["meta"],
        "frames": frames_payload,
    }
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
        siglip_layer_indices, n_siglip = common.default_siglip_layer_indices(stack.model)
        print(f"Analyzing SigLIP layers: 0..{n_siglip - 1} (all {n_siglip})")

    capture_kwargs = dict(
        rollout_max_steps=args.rollout_max_steps,
        rollout_seed_candidates=tuple(args.rollout_seed_candidates),
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
