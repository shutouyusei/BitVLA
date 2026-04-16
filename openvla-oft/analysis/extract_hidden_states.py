"""
Hidden state extraction for BitVLA models.

Registers forward hooks on BitNetDecoderLayer to capture hidden states
at each layer during inference.

Usage:
    from analysis.extract_hidden_states import extract_hidden_states
    hidden_states = extract_hidden_states(model, inputs)
"""

import torch
from typing import Dict, List, Optional


def extract_hidden_states(
    model: torch.nn.Module,
    inputs: dict,
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Run forward pass with hooks to capture hidden states from BitNetDecoderLayer.

    Each decoder layer outputs a hidden state tensor that represents the
    evolving representation at that depth. By comparing these across
    original vs perturbed inputs, we can see where representations diverge.

    Args:
        model: BitVLA model (eval mode)
        inputs: dict with input_ids, attention_mask, pixel_values
        layer_indices: which layers to extract (default: all)

    Returns:
        dict mapping layer_idx -> hidden states (seq_len, hidden_dim)
    """
    from transformers.models.llava.modeling_bitnet import BitNetDecoderLayer

    # Find all decoder layers
    decoder_layers = []
    for name, module in model.named_modules():
        if isinstance(module, BitNetDecoderLayer):
            decoder_layers.append((name, module))

    if not decoder_layers:
        raise RuntimeError(
            f"No BitNetDecoderLayer found in the model. "
            f"Model type: {type(model).__name__}"
        )

    if layer_indices is None:
        layer_indices = list(range(len(decoder_layers)))

    for idx in layer_indices:
        if idx >= len(decoder_layers):
            raise ValueError(
                f"Layer index {idx} out of range. Model has {len(decoder_layers)} "
                f"decoder layers (0-{len(decoder_layers)-1})."
            )

    captured = {}
    hooks = []

    def make_capture_hook(layer_idx):
        def hook_fn(module, input, output):
            # BitNetDecoderLayer.forward returns (hidden_states, ...) tuple
            if isinstance(output, tuple) and len(output) >= 1:
                captured[layer_idx] = output[0].detach().cpu().squeeze(0)  # (seq_len, hidden_dim)
        return hook_fn

    for idx in layer_indices:
        name, module = decoder_layers[idx]
        hook = module.register_forward_hook(make_capture_hook(idx))
        hooks.append(hook)

    try:
        import copy
        fwd_inputs = copy.copy(inputs)
        seq_len = fwd_inputs["input_ids"].shape[1]
        fwd_inputs["labels"] = torch.full(
            (1, seq_len), -100, dtype=torch.long, device=fwd_inputs["input_ids"].device
        )
        with torch.inference_mode():
            model(**fwd_inputs)
    finally:
        for hook in hooks:
            hook.remove()

    if not captured:
        raise RuntimeError(
            f"No hidden states captured from {len(layer_indices)} requested layers."
        )

    return captured


def get_token_types(
    input_ids: torch.Tensor,
    image_token_idx: int,
    proprio_pad_idx: int,
    action_token_begin_idx: int,
) -> List[str]:
    """
    Classify each token in the input sequence by type for coloring in t-SNE.

    Args:
        input_ids: (seq_len,) or (1, seq_len) token IDs
        image_token_idx: token ID for image patches
        proprio_pad_idx: token ID for proprioception
        action_token_begin_idx: token ID marking start of action tokens

    Returns:
        list of token type strings: "image", "text", "proprio", or "action"
    """
    if input_ids.dim() == 2:
        input_ids = input_ids.squeeze(0)

    # Note: proprio check must come before action check because
    # proprio_pad_idx == action_token_begin_idx (both 128011).
    # The single proprio sentinel is classified first; tokens with
    # IDs > action_token_begin_idx are classified as action.
    token_types = []
    for tid in input_ids.tolist():
        if tid == image_token_idx:
            token_types.append("image")
        elif tid == proprio_pad_idx:
            token_types.append("proprio")
        elif tid > action_token_begin_idx:
            token_types.append("action")
        else:
            token_types.append("text")

    return token_types
