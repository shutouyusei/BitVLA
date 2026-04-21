"""t-SNE analysis entry point for BitVLA on LIBERO tasks.

Usage:
    cd openvla-oft
    python -m analysis.run_tsne \
        --checkpoint_path /path/to/bitvla \
        --condition normal_success:libero_spatial \
        --condition normal_failed:libero_spatial \
        --condition pro_init:libero_spatial_swap \
        --condition pro_manip:libero_spatial_swap:mid_rollout \
        --task_id 0 --device cuda:0
"""

import argparse
import math
import os

import matplotlib.pyplot as plt

from analysis import common
from analysis.extract_hidden_states import extract_hidden_states, get_token_types
from analysis.visualize_tsne import plot_tsne_single
from bitvla.constants import (
    BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    BITNET_PROPRIO_PAD_IDX,
    BITNET_ACTION_TOKEN_BEGIN_IDX,
)


def analyze_condition(stack, label, suite, mode, task_id, layer_indices, capture_kwargs):
    print(f"\n{'=' * 60}\n[{label}] suite={suite} mode={mode} task={task_id}\n{'=' * 60}")
    image, task_desc, task_name, meta = common.capture_observation(
        suite, task_id, mode, stack=stack, **capture_kwargs
    )
    inputs = common.prepare_inputs(stack.model, stack.processor, image, task_desc)

    print("Extracting hidden states...")
    hidden_states = extract_hidden_states(stack.model, inputs, layer_indices=layer_indices)
    token_types = get_token_types(
        inputs["input_ids"],
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
    )

    return {
        "label": label,
        "hidden_states": hidden_states,
        "token_types": token_types,
        "task_description": task_desc,
        "task_name": task_name,
        "meta": meta,
    }


def save_per_condition(result, layer_indices, output_dir, task_id):
    """Grid of t-SNE plots — one per layer — for a single condition."""
    hs = result["hidden_states"]
    layers = [l for l in layer_indices if l in hs]
    if not layers:
        print(f"[{result['label']}] no hidden states captured; skipping t-SNE.")
        return

    ncols = min(4, len(layers))
    nrows = math.ceil(len(layers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, layer_idx in enumerate(layers):
        ax = axes[idx // ncols][idx % ncols]
        plot_tsne_single(
            hs[layer_idx].squeeze(0).detach().cpu(),
            result["token_types"],
            ax=ax,
            title=f"layer {layer_idx}",
        )

    for idx in range(len(layers), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{result['task_description']} [{result['label']}]", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, f"tsne_{result['label']}_task{task_id}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="BitVLA t-SNE hidden-state analysis")
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
        save_per_condition(result, layer_indices, args.output_dir, args.task_id)

    print(f"\nt-SNE outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
