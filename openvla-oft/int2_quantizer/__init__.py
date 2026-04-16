from transformers.models.llava.modeling_bitnet import BitLinear


def quantize_bitlinear_layers(model):
    """
    Quantize all BitLinear layers in a model to 2-bit packed format.

    Calls BitLinear.quantize_weights() on each unquantized layer, which:
    1. Packs bf16 weights into 2-bit (4 values per byte)
    2. Stores the scale factor (w_step)
    3. Sets the original bf16 weight to None (allowing garbage collection)

    This reduces weight memory by ~8x for BitLinear layers.
    Non-BitLinear layers (embeddings, vision encoder, connector) remain in bf16.

    Returns:
        int: Number of layers quantized.

    Raises:
        RuntimeError: If no BitLinear layers are found or if quantization fails on any layer.
    """
    count = 0
    for name, module in model.named_modules():
        # Skip layers already quantized (enable_qlora is True after quantize_weights())
        if isinstance(module, BitLinear) and not module.enable_qlora:
            try:
                module.quantize_weights()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to quantize BitLinear layer '{name}': {e}. "
                    f"Model is in a partially quantized state ({count} layers quantized "
                    f"before failure). Do not use this model."
                ) from e
            count += 1

    if count == 0:
        raise RuntimeError(
            "use_int2_quantization=True but no BitLinear layers were found to quantize. "
            "Verify the checkpoint contains BitLinear layers and was not already quantized."
        )

    # Post-quantization integrity check
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and module.enable_qlora:
            if not hasattr(module, "q_weight") or module.q_weight is None:
                raise RuntimeError(
                    f"Layer '{name}' marked as quantized but has no q_weight buffer. "
                    f"Quantization may have failed silently."
                )

    return count
