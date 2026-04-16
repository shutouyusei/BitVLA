"""
Attention map extraction for BitVLA models.

Registers forward hooks on BitNetAttention layers to capture attention weights
during inference, without modifying the model code.

Usage:
    from analysis.extract_attention import extract_attention_maps
    attention_maps = extract_attention_maps(model, inputs)
"""

import torch
from typing import Dict, List, Optional


def extract_attention_maps(
    model: torch.nn.Module,
    inputs: dict,
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Run forward pass with hooks to capture attention weights from BitNetAttention layers.

    Uses register_forward_hook on each attention layer to intercept the returned
    attn_weights tensor. This avoids modifying model code or using output_attentions=True
    which may not propagate correctly through all model wrappers.

    Args:
        model: BitVLA model (on GPU, eval mode)
        inputs: dict with input_ids, attention_mask, pixel_values (already on device)
        layer_indices: which layers to extract (default: all). 0-indexed.

    Returns:
        dict mapping layer_idx -> attention weights tensor (num_heads, seq_len, seq_len)
    """
    from transformers.models.llava.modeling_bitnet import BitNetAttention

    # Find all attention layers
    attn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, BitNetAttention):
            attn_layers.append((name, module))

    if layer_indices is None:
        layer_indices = list(range(len(attn_layers)))

    # Storage for captured attention weights
    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # BitNetAttention.forward returns (attn_output, attn_weights, past_key_value)
            # attn_weights is None when output_attentions=False, so we need to
            # temporarily enable it
            pass
        return hook_fn

    # We need output_attentions=True for the model to return attention weights.
    # Override at the config level temporarily.
    original_output_attentions = model.config.output_attentions
    model.config.output_attentions = True

    # Also set on text_config if it exists (nested config for LLaVA models)
    original_text_output_attentions = None
    if hasattr(model.config, "text_config"):
        original_text_output_attentions = model.config.text_config.output_attentions
        model.config.text_config.output_attentions = True

    # We use a pre-hook to force output_attentions=True in the attention forward args,
    # and a post-hook to capture the returned attention weights.
    def make_pre_hook():
        def hook_fn(module, args, kwargs):
            # BitNetAttention.forward signature includes output_attentions as a kwarg
            kwargs["output_attentions"] = True
            return args, kwargs
        return hook_fn

    def make_capture_hook(layer_idx):
        def hook_fn(module, input, output):
            # output = (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured[layer_idx] = output[1].detach().cpu()
        return hook_fn

    for idx in layer_indices:
        if idx < len(attn_layers):
            name, module = attn_layers[idx]
            pre_hook = module.register_forward_pre_hook(make_pre_hook(), with_kwargs=True)
            post_hook = module.register_forward_hook(make_capture_hook(idx))
            hooks.append(pre_hook)
            hooks.append(post_hook)

    # Run a single forward pass with dummy labels to avoid the labels=None issue.
    # We create fake labels (all -100 = ignored) so the forward pass runs
    # through the full pipeline but doesn't compute any action loss.
    import copy
    fwd_inputs = copy.copy(inputs)
    seq_len = fwd_inputs["input_ids"].shape[1]
    fwd_inputs["labels"] = torch.full(
        (1, seq_len), -100, dtype=torch.long, device=fwd_inputs["input_ids"].device
    )
    with torch.inference_mode():
        model(**fwd_inputs)

    # Clean up hooks and restore config
    for hook in hooks:
        hook.remove()
    model.config.output_attentions = original_output_attentions
    if original_text_output_attentions is not None:
        model.config.text_config.output_attentions = original_text_output_attentions

    # Squeeze batch dimension (assuming batch_size=1)
    result = {}
    for layer_idx, attn in captured.items():
        result[layer_idx] = attn.squeeze(0)  # (num_heads, seq_len, seq_len)

    return result


def get_image_token_attention(
    attention_maps: Dict[int, torch.Tensor],
    input_ids: torch.Tensor,
    image_token_idx: int,
    num_patches_per_image: int = 256,
) -> Dict[int, torch.Tensor]:
    """
    Extract attention from action/text tokens to image tokens.

    For each layer, averages attention heads and extracts the columns
    corresponding to image tokens, giving a (seq_len, num_image_patches) matrix.
    We then take the mean across non-image query tokens to get a single
    attention distribution over image patches.

    Args:
        attention_maps: output from extract_attention_maps()
        input_ids: input token IDs (1, seq_len) or (seq_len,)
        image_token_idx: the token ID used for image patches
        num_patches_per_image: number of patches per image (default 256 for 224x224 with patch_size=14)

    Returns:
        dict mapping layer_idx -> attention over image patches (num_patches,)
        Can be reshaped to (sqrt(num_patches), sqrt(num_patches)) for visualization.
    """
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)

    # Find image token positions
    image_mask = input_ids == image_token_idx
    image_positions = torch.where(image_mask)[0]

    if len(image_positions) == 0:
        raise ValueError(f"No image tokens found with idx={image_token_idx}")

    # Take the first image's patches
    first_image_start = image_positions[0].item()
    first_image_end = first_image_start + num_patches_per_image

    result = {}
    for layer_idx, attn in attention_maps.items():
        # attn shape: (num_heads, seq_len, seq_len)
        # Average across heads
        attn_mean = attn.float().mean(dim=0)  # (seq_len, seq_len)

        # Extract columns for image tokens: how much each token attends to image patches
        image_attn = attn_mean[:, first_image_start:first_image_end]  # (seq_len, num_patches)

        # Average across all non-image query positions to get overall image attention
        non_image_mask = ~image_mask
        non_image_attn = image_attn[non_image_mask]  # (num_non_image_tokens, num_patches)
        avg_attn = non_image_attn.mean(dim=0)  # (num_patches,)

        result[layer_idx] = avg_attn

    return result
