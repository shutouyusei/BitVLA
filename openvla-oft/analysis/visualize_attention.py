"""
Attention map visualization for BitVLA models.

Overlays attention heatmaps on input images to show where the model focuses.
Supports single image, side-by-side comparison, and multi-layer grid views.

Usage:
    from analysis.visualize_attention import visualize_attention_on_image, compare_attention_maps
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def _draw_bboxes(ax, bboxes):
    """Overlay per-object bounding boxes. Target objects are drawn in red, others in lime."""
    if not bboxes:
        return
    from matplotlib.patches import Rectangle

    for b in bboxes:
        ymin, xmin, ymax, xmax = b["bbox"]
        color = "red" if b.get("is_target") else "lime"
        rect = Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            xmin, max(ymin - 2, 0), b["name"],
            color=color, fontsize=6,
            bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5),
        )


def visualize_attention_on_image(
    image: np.ndarray,
    attention_over_patches: torch.Tensor,
    patch_grid_size: int = 16,
    ax: plt.Axes = None,
    title: str = "",
    alpha: float = 0.5,
    cmap: str = "jet",
    bboxes=None,
) -> plt.Axes:
    """
    Overlay attention heatmap on the input image.

    Takes a (num_patches,) attention vector, reshapes to a grid,
    upscales to image resolution, and overlays as a semi-transparent heatmap.

    Args:
        image: input image as numpy array (H, W, 3)
        attention_over_patches: attention weights (num_patches,)
        patch_grid_size: sqrt(num_patches), default 16 for 256 patches
        ax: matplotlib axes to draw on (creates new figure if None)
        title: title for the subplot
        alpha: transparency of the heatmap overlay
        cmap: colormap name

    Returns:
        matplotlib Axes with the visualization
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    attn = attention_over_patches.float().numpy()
    attn_grid = attn.reshape(patch_grid_size, patch_grid_size)

    # Normalize to [0, 1]
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

    # Upscale to image resolution
    attn_image = Image.fromarray((attn_grid * 255).astype(np.uint8))
    attn_image = attn_image.resize((image.shape[1], image.shape[0]), Image.BICUBIC)
    attn_upscaled = np.array(attn_image).astype(np.float32) / 255.0

    ax.imshow(image)
    ax.imshow(attn_upscaled, alpha=alpha, cmap=cmap)
    _draw_bboxes(ax, bboxes)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    return ax


def compare_attention_maps(
    image_original: np.ndarray,
    image_perturbed: np.ndarray,
    attn_original: torch.Tensor,
    attn_perturbed: torch.Tensor,
    layer_idx: int,
    task_label: str = "",
    output_path: str = "attention_comparison.png",
    patch_grid_size: int = 16,
) -> None:
    """
    Side-by-side comparison of attention maps on original vs perturbed images.

    Args:
        image_original: original LIBERO image (H, W, 3)
        image_perturbed: position-perturbed image (H, W, 3)
        attn_original: attention over patches for original (num_patches,)
        attn_perturbed: attention over patches for perturbed (num_patches,)
        layer_idx: which layer these attentions are from
        task_label: task description for the title
        output_path: where to save the figure
        patch_grid_size: sqrt(num_patches)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    visualize_attention_on_image(
        image_original,
        attn_original,
        patch_grid_size=patch_grid_size,
        ax=axes[0],
        title=f"Original (Layer {layer_idx})",
    )

    visualize_attention_on_image(
        image_perturbed,
        attn_perturbed,
        patch_grid_size=patch_grid_size,
        ax=axes[1],
        title=f"Position Perturbed (Layer {layer_idx})",
    )

    fig.suptitle(f"BitVLA Attention: {task_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_single_layer_png(
    image: np.ndarray,
    attention_over_patches,
    layer_idx: int,
    output_path: str,
    patch_grid_size: int = 16,
    bboxes=None,
    title: str = None,
) -> None:
    """One-layer attention heatmap saved as its own PNG. For per-layer PNG outputs."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    visualize_attention_on_image(
        image, attention_over_patches,
        patch_grid_size=patch_grid_size,
        ax=ax,
        title=title if title is not None else f"Layer {layer_idx}",
        bboxes=bboxes,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_all_layers(
    image: np.ndarray,
    attention_maps: dict,
    task_label: str = "",
    output_path: str = "attention_all_layers.png",
    patch_grid_size: int = 16,
    max_layers: int = 12,
    bboxes=None,
    layer_indices: list = None,
) -> None:
    """
    Grid visualization of attention maps across multiple layers.

    Shows how attention evolves from early to late layers.

    Args:
        image: input image (H, W, 3)
        attention_maps: dict from get_image_token_attention(), layer_idx -> (num_patches,)
        task_label: task description
        output_path: where to save
        patch_grid_size: sqrt(num_patches)
        max_layers: maximum number of layers to show
    """
    if layer_indices is None:
        layer_indices = sorted(attention_maps.keys())[:max_layers]
    else:
        layer_indices = [l for l in layer_indices if l in attention_maps]
    n = len(layer_indices)
    if n == 0:
        return
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer_idx in enumerate(layer_indices):
        visualize_attention_on_image(
            image,
            attention_maps[layer_idx],
            patch_grid_size=patch_grid_size,
            ax=axes[i],
            title=f"Layer {layer_idx}",
            bboxes=bboxes,
        )

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"BitVLA Attention Across Layers: {task_label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
