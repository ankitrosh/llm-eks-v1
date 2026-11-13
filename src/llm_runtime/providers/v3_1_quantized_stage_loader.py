"""V3.1 quantized linear adapter for Qwen3 decoder layers.

The provider builds executable replacement linear modules.
The framework loads remaining ordinary modules selectively.

This does not implement GPTQ kernels or checkpoint decoding.
Frozen V3 code is not modified.
"""

from __future__ import annotations

import torch
from torch import nn

from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    StageCheckpointError,
    materialize_module,
)


def _ensure_materialized(module, name, device):
    device = torch.device(device)

    for tensor_name, tensor in list(
        module.named_parameters()
    ) + list(module.named_buffers()):
        if tensor.is_meta:
            raise StageCheckpointError(
                f"Quantization adapter left meta tensor: "
                f"{name}.{tensor_name}"
            )

        if tensor.device != device:
            raise StageCheckpointError(
                f"Quantization adapter returned a tensor on "
                f"the wrong device: {name}.{tensor_name}"
            )


def materialize_quantized_decoder_layer(
    layer,
    checkpoint_prefix,
    reader,
    provider,
    device,
    dtype,
):
    """Replace selected linears and load remaining parameters.

    Provider contract:
        should_quantize(module_name) -> bool  [optional]

        build_linear(
            module_name=...,
            in_features=...,
            out_features=...,
            bias=...,
            reader=...,
            device=...,
            dtype=...,
        ) -> fully materialized nn.Module

    The module_name is the full checkpoint module prefix,
    without a trailing dot.

    Returns (ordinary_tensor_count, quantized_linear_count).
    """

    if not isinstance(layer, nn.Module):
        raise TypeError("Expected torch.nn.Module")

    build = getattr(provider, "build_linear", None)

    if not callable(build):
        raise StageCheckpointError(
            "Quantization provider requires build_linear"
        )

    selector = getattr(provider, "should_quantize", None)

    if selector is not None and not callable(selector):
        raise StageCheckpointError(
            "should_quantize must be callable"
        )

    # Snapshot names before modifying the module tree.
    candidates = [
        (name, module)
        for name, module in layer.named_modules()
        if name and isinstance(module, nn.Linear)
    ]

    replaced = []

    for relative_name, original in candidates:
        full_name = checkpoint_prefix + relative_name

        selected = (
            True if selector is None
            else selector(full_name)
        )

        if type(selected) is not bool:
            raise StageCheckpointError(
                "should_quantize must return bool"
            )

        if not selected:
            continue

        replacement = build(
            module_name=full_name,
            in_features=original.in_features,
            out_features=original.out_features,
            bias=original.bias is not None,
            reader=reader,
            device=device,
            dtype=dtype,
        )

        if not isinstance(replacement, nn.Module):
            raise StageCheckpointError(
                f"build_linear must return nn.Module: "
                f"{full_name}"
            )

        _ensure_materialized(
            replacement, full_name, device
        )

        parent_name, _, child_name = (
            relative_name.rpartition(".")
        )

        parent = (
            layer.get_submodule(parent_name)
            if parent_name else layer
        )

        setattr(parent, child_name, replacement)
        replaced.append(relative_name)

    # Materialize all ordinary, non-quantized leaves.
    # Never load through a replaced quantized module.
    ordinary_loaded = 0

    for name, module in layer.named_modules():
        if any(
            name == replaced_name
            or name.startswith(replaced_name + ".")
            for replaced_name in replaced
        ):
            continue

        has_own_state = any(
            p is not None
            for p in module._parameters.values()
        ) or any(
            b is not None
            for b in module._buffers.values()
            if b is not None
        )

        if not has_own_state:
            continue

        # The Qwen3 decoder's checkpoint-backed modules
        # are leaves. Fail closed if the structure changes.
        if any(module.children()):
            raise StageCheckpointError(
                f"Unsupported non-leaf checkpoint state: "
                f"{checkpoint_prefix}{name}"
            )

        if not name:
            raise StageCheckpointError(
                "Unsupported root-level checkpoint state"
            )

        ordinary_loaded += materialize_module(
            module,
            checkpoint_prefix + name + ".",
            reader,
            device,
            dtype,
        )

    _ensure_materialized(
        layer, checkpoint_prefix.rstrip("."), device
    )

    return ordinary_loaded, len(replaced)
