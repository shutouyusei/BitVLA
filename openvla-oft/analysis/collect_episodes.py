"""Batch collection of Phase 1 episodes.

Two modes:
  collect   — run rollouts, extract attention on all 4 frames, save per-episode JSON.
  prescreen — run rollouts only, record success flag per (task_id, seed) to a file.

Usage:
    # Condition A: 20 successes across libero_spatial tasks 0-9
    python -m analysis.collect_episodes \\
        --mode collect \\
        --checkpoint_path /path/to/ft-ckpt \\
        --condition A_normal_success:libero_spatial:success \\
        --task_ids 0 1 2 3 4 5 6 7 8 9 \\
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \\
        --n 20 --include_siglip --device cuda:0 --use_int2_quantization

    # Prescreen libero_spatial for failing seeds
    python -m analysis.collect_episodes \\
        --mode prescreen \\
        --checkpoint_path /path/to/ft-ckpt \\
        --suite libero_spatial \\
        --task_ids 0 1 2 3 4 5 6 7 8 9 \\
        --seeds $(seq 0 99) \\
        --prescreen_output prescreen_libero_spatial.json \\
        --device cuda:0 --use_int2_quantization
"""

import argparse
import json
import os

from analysis import common
from analysis.run_attention import (
    analyze_frame,
    compute_bbox_attention_ratios,
    compute_bbox_attention_ratios_per_head,
    _frame_suffix,
)
from analysis.visualize_attention import visualize_all_layers


def _parse_condition(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Condition must be 'label:suite:outcome' (outcome ∈ success/failed/any); got '{spec}'"
        )
    label, suite, outcome = parts
    if outcome not in ("success", "failed", "any"):
        raise argparse.ArgumentTypeError(f"outcome must be success/failed/any, got '{outcome}'")
    return label, suite, outcome


def _matches_target(success_flag, target_outcome):
    if target_outcome == "any":
        return True
    return (target_outcome == "success") == bool(success_flag)


def _atomic_json_dump(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _episode_path(episodes_dir, label, task_id, seed):
    return os.path.join(episodes_dir, f"{label}_task{task_id}_seed{seed}.json")


def _build_frame_entry(attn, fr, include_siglip):
    entry = {
        "step_idx": fr["step_idx"],
        "fallback": fr["fallback"],
        "bboxes": fr["bboxes"],
        "per_layer_ratios_llm": compute_bbox_attention_ratios(
            attn["attention"], fr["bboxes"], fr["image"].shape
        ),
        "per_layer_per_head_llm": compute_bbox_attention_ratios_per_head(
            attn["attention_per_head"], fr["bboxes"], fr["image"].shape
        ),
    }
    if include_siglip:
        entry["per_layer_ratios_siglip"] = compute_bbox_attention_ratios(
            attn["siglip_attention"], fr["bboxes"], fr["image"].shape
        )
        entry["per_layer_per_head_siglip"] = compute_bbox_attention_ratios_per_head(
            attn["siglip_attention_per_head"], fr["bboxes"], fr["image"].shape
        )
    return entry


def run_collect(args, stack, layer_indices, siglip_layer_indices, episodes_dir, png_dir):
    label, suite, target_outcome = _parse_condition(args.condition)
    print(f"Condition: label={label} suite={suite} outcome={target_outcome} target N={args.n}")

    # Resume: count existing matching episodes
    collected = []
    for task_id in args.task_ids:
        for seed in args.seeds:
            path = _episode_path(episodes_dir, label, task_id, seed)
            if not os.path.exists(path):
                continue
            try:
                ep = json.load(open(path))
                if _matches_target(ep["meta"].get("success"), target_outcome):
                    collected.append((task_id, seed))
            except (OSError, json.JSONDecodeError):
                continue
    if collected:
        print(f"Resume: {len(collected)} matching episodes already on disk.")
    if len(collected) >= args.n:
        print("Target already met — nothing to do.")
        return

    attempts = 0
    for task_id in args.task_ids:
        for seed in args.seeds:
            if len(collected) >= args.n:
                break
            if (task_id, seed) in collected:
                continue
            path = _episode_path(episodes_dir, label, task_id, seed)

            attempts += 1
            print(f"\n[{len(collected)}/{args.n}] task={task_id} seed={seed} (attempt {attempts})")
            try:
                capture = common.capture_frames(
                    suite, task_id, "mid_rollout", stack=stack, seed=seed,
                    rollout_max_steps=args.rollout_max_steps,
                )
            except Exception as e:
                print(f"  rollout errored: {e}")
                continue

            success = capture["meta"]["success"]
            if not _matches_target(success, target_outcome):
                print(f"  outcome={success}, want {target_outcome} — skip")
                continue

            print(f"  matched, running attention analysis on {len(capture['frames'])} frames...")
            frames_payload = {}
            for frame_name, fr in capture["frames"].items():
                attn = analyze_frame(
                    stack, fr["image"], capture["task_description"],
                    layer_indices, args.include_siglip, siglip_layer_indices,
                )
                frames_payload[frame_name] = _build_frame_entry(attn, fr, args.include_siglip)
                if args.save_png:
                    tag = _frame_suffix(frame_name)
                    visualize_all_layers(
                        fr["image"], attn["attention"],
                        task_label=f"{capture['task_description']} [{label}/{frame_name}] (LLM)",
                        output_path=os.path.join(
                            png_dir, f"layers_{label}_task{task_id}_seed{seed}_{tag}.png"),
                        bboxes=fr["bboxes"],
                    )
                    if args.include_siglip:
                        visualize_all_layers(
                            fr["image"], attn["siglip_attention"],
                            task_label=f"{capture['task_description']} [{label}/{frame_name}] (SigLIP)",
                            output_path=os.path.join(
                                png_dir, f"layers_siglip_{label}_task{task_id}_seed{seed}_{tag}.png"),
                            bboxes=fr["bboxes"],
                        )

            payload = {
                "label": label,
                "task_id": task_id,
                "seed": seed,
                "suite": suite,
                "task_description": capture["task_description"],
                "task_name": capture["task_name"],
                "meta": capture["meta"],
                "frames": frames_payload,
            }
            _atomic_json_dump(path, payload)
            collected.append((task_id, seed))
            print(f"  saved {os.path.basename(path)}  [{len(collected)}/{args.n}]")
        if len(collected) >= args.n:
            break

    print(f"\nCollected {len(collected)}/{args.n} for {label} (attempted {attempts} new rollouts).")


def run_prescreen(args, stack, out_path):
    """Run rollouts only, record {task_id, seed, success}. No attention extraction."""
    existing = {}
    if os.path.exists(out_path):
        try:
            existing = json.load(open(out_path))
        except (OSError, json.JSONDecodeError):
            existing = {}

    results = existing if isinstance(existing, dict) else {}
    results.setdefault("suite", args.suite)
    results.setdefault("records", [])
    done_pairs = {(r["task_id"], r["seed"]) for r in results["records"]}

    suite = args.suite
    total = len(args.task_ids) * len(args.seeds)
    done = len(done_pairs)
    for task_id in args.task_ids:
        for seed in args.seeds:
            if (task_id, seed) in done_pairs:
                continue
            done += 1
            print(f"[{done}/{total}] task={task_id} seed={seed} ...", end=" ", flush=True)
            try:
                capture = common.capture_frames(
                    suite, task_id, "mid_rollout", stack=stack, seed=seed,
                    rollout_max_steps=args.rollout_max_steps,
                )
                success = bool(capture["meta"]["success"])
                final_step = capture["meta"].get("final_step_idx")
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            print(f"success={success} final_step={final_step}")
            results["records"].append({
                "task_id": task_id, "seed": seed,
                "success": success, "final_step_idx": final_step,
            })
            _atomic_json_dump(out_path, results)

    n_success = sum(1 for r in results["records"] if r["success"])
    n_failed = len(results["records"]) - n_success
    print(f"\nPrescreen {suite}: {n_success} success / {n_failed} failed across {len(results['records'])} rollouts.")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 episode collector")
    parser.add_argument("--mode", choices=["collect", "prescreen"], required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_int2_quantization", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./analysis_output")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Subdir under --output_dir (default: today's date).")
    parser.add_argument("--task_ids", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rollout_max_steps", type=int, default=None)

    # Collect-mode args
    parser.add_argument("--condition", type=str, default=None,
                        help="Collect mode: 'label:suite:outcome' where outcome ∈ success/failed/any.")
    parser.add_argument("--n", type=int, default=20, help="Target number of matching episodes.")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--siglip_layers", type=int, nargs="*", default=None)
    parser.add_argument("--include_siglip", action="store_true")
    parser.add_argument("--save_png", action="store_true",
                        help="Save per-frame attention PNG alongside the JSON (adds disk usage).")

    # Prescreen-mode args
    parser.add_argument("--suite", type=str, default=None,
                        help="Prescreen mode: suite name.")
    parser.add_argument("--prescreen_output", type=str, default=None,
                        help="Prescreen mode: where to write the {task_id, seed, success} records.")

    args = parser.parse_args()
    common.set_device(args.device)

    stack = common.load_rollout_stack(
        args.checkpoint_path,
        use_int2=args.use_int2_quantization,
        task_suite_name=(args.suite or "libero_spatial"),
    )

    if args.mode == "collect":
        if args.condition is None:
            parser.error("--condition is required for collect mode")
        png_dir, json_dir = common.resolve_output_dirs(args.output_dir, args.output_subdir)
        episodes_dir = os.path.join(json_dir, "episodes")
        os.makedirs(episodes_dir, exist_ok=True)

        if args.layers is None:
            layer_indices, n_llm = common.default_layer_indices(stack.model)
            print(f"LLM layers: all {n_llm}")
        else:
            layer_indices = args.layers

        siglip_layer_indices = args.siglip_layers
        if args.include_siglip and siglip_layer_indices is None:
            siglip_layer_indices, n_siglip = common.default_siglip_layer_indices(stack.model)
            print(f"SigLIP layers: all {n_siglip}")

        run_collect(args, stack, layer_indices, siglip_layer_indices, episodes_dir, png_dir)

    else:  # prescreen
        if args.suite is None or args.prescreen_output is None:
            parser.error("--suite and --prescreen_output are required for prescreen mode")
        run_prescreen(args, stack, args.prescreen_output)


if __name__ == "__main__":
    main()
