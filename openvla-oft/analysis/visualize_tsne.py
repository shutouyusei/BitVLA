"""
t-SNE visualization of BitVLA hidden state representations.

Projects token embeddings from different layers to 2D, colored by token type
(image, text, action). Compares original vs perturbed to reveal
representation collapse.

Usage:
    python visualize_tsne.py \
        --checkpoint_path /path/to/bitvla \
        --task_suite original:libero_spatial perturbed:libero_spatial_swap \
        --output_dir ./analysis_output
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Token type -> color mapping
TOKEN_COLORS = {
    "image": "#2196F3",    # blue
    "text": "#4CAF50",     # green
    "proprio": "#FF9800",  # orange
    "action": "#F44336",   # red
    "special": "#9E9E9E",  # gray
}


def plot_tsne_single(
    hidden_states: torch.Tensor,
    token_types: list,
    ax: plt.Axes,
    title: str = "",
    perplexity: int = 30,
) -> None:
    """
    Project hidden states to 2D with t-SNE and plot colored by token type.

    Args:
        hidden_states: (seq_len, hidden_dim) tensor
        token_types: list of token type strings per position
        ax: matplotlib axes
        title: subplot title
        perplexity: t-SNE perplexity (lower = more local structure)
    """
    X = hidden_states.float().numpy()
    n_samples = X.shape[0]

    # Adjust perplexity if too few samples
    perp = min(perplexity, max(5, n_samples // 4))

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    # Plot each token type
    for ttype in ["image", "text", "proprio", "action"]:
        mask = [i for i, t in enumerate(token_types) if t == ttype]
        if mask:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=TOKEN_COLORS[ttype],
                label=ttype,
                s=15,
                alpha=0.6,
            )

    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])


def compare_tsne(
    hidden_original: dict,
    hidden_perturbed: dict,
    token_types_original: list,
    token_types_perturbed: list,
    task_label: str = "",
    output_path: str = "tsne_comparison.png",
    perplexity: int = 30,
) -> None:
    """
    Side-by-side t-SNE comparison of original vs perturbed hidden states per layer.

    Args:
        hidden_original: dict layer_idx -> (seq_len, hidden_dim) for original
        hidden_perturbed: dict layer_idx -> (seq_len, hidden_dim) for perturbed
        token_types_original: token types for original input
        token_types_perturbed: token types for perturbed input
        task_label: task description
        output_path: where to save
        perplexity: t-SNE perplexity
    """
    common_layers = sorted(set(hidden_original.keys()) & set(hidden_perturbed.keys()))
    n_layers = len(common_layers)

    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 5 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, 2)

    for i, layer_idx in enumerate(common_layers):
        plot_tsne_single(
            hidden_original[layer_idx],
            token_types_original,
            axes[i, 0],
            title=f"Original — Layer {layer_idx}",
            perplexity=perplexity,
        )
        plot_tsne_single(
            hidden_perturbed[layer_idx],
            token_types_perturbed,
            axes[i, 1],
            title=f"Perturbed — Layer {layer_idx}",
            perplexity=perplexity,
        )

    fig.suptitle(f"BitVLA t-SNE: {task_label}", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def overlay_tsne(
    hidden_original: dict,
    hidden_perturbed: dict,
    token_types: list,
    task_label: str = "",
    output_path: str = "tsne_overlay.png",
    perplexity: int = 30,
) -> None:
    """
    Overlay t-SNE of original and perturbed on the same plot per layer.

    Original tokens are plotted as circles, perturbed as X markers.
    If representations are robust, clusters should overlap.
    If representations collapse, clusters will separate.

    Args:
        hidden_original: dict layer_idx -> (seq_len, hidden_dim) for original
        hidden_perturbed: dict layer_idx -> (seq_len, hidden_dim) for perturbed
        token_types: token type list (same structure assumed for both)
        task_label: task description
        output_path: where to save
        perplexity: t-SNE perplexity
    """
    common_layers = sorted(set(hidden_original.keys()) & set(hidden_perturbed.keys()))
    n_layers = len(common_layers)
    cols = min(4, n_layers)
    rows = math.ceil(n_layers / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer_idx in enumerate(common_layers):
        ax = axes[i]
        h_orig = hidden_original[layer_idx].float().numpy()
        h_pert = hidden_perturbed[layer_idx].float().numpy()

        # Concatenate and run t-SNE together for fair comparison
        X = np.concatenate([h_orig, h_pert], axis=0)
        n_orig = h_orig.shape[0]
        perp = min(perplexity, max(5, X.shape[0] // 4))

        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        X_2d = tsne.fit_transform(X)

        X_orig_2d = X_2d[:n_orig]
        X_pert_2d = X_2d[n_orig:]

        # Plot image tokens only (most informative for visual grounding)
        img_mask = [j for j, t in enumerate(token_types) if t == "image"]
        if img_mask and max(img_mask) < n_orig:
            ax.scatter(
                X_orig_2d[img_mask, 0], X_orig_2d[img_mask, 1],
                c=TOKEN_COLORS["image"], marker="o", s=20, alpha=0.5, label="Original (image)",
            )
            ax.scatter(
                X_pert_2d[img_mask, 0], X_pert_2d[img_mask, 1],
                c="#E91E63", marker="x", s=20, alpha=0.5, label="Perturbed (image)",
            )

        # Plot text tokens
        txt_mask = [j for j, t in enumerate(token_types) if t == "text"]
        if txt_mask and max(txt_mask) < n_orig:
            ax.scatter(
                X_orig_2d[txt_mask, 0], X_orig_2d[txt_mask, 1],
                c=TOKEN_COLORS["text"], marker="o", s=15, alpha=0.3, label="Original (text)",
            )
            ax.scatter(
                X_pert_2d[txt_mask, 0], X_pert_2d[txt_mask, 1],
                c="#8BC34A", marker="x", s=15, alpha=0.3, label="Perturbed (text)",
            )

        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(len(common_layers), len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"BitVLA t-SNE Overlay: {task_label}\n(circles=original, x=perturbed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
