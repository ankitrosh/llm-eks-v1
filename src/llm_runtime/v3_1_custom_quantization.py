"""Custom quantization checkpoint planning for V3.1.

Loads trusted provider Python code to inspect checkpoint metadata.
Does not load quantized weights or initialize execution kernels.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

from llm_runtime.v3_1_quantization_api import (
    QuantizationContext,
    QuantizationProfile,
    QuantizationProviderError,
    QuantizationRegistry,
)
from llm_runtime.v3_1_runtime_config import resolve_v31_config


def resolve_custom_quantized_plan(
    config,
    *,
    gpu_memory_gib=None,
    registry=None,
):
    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be an object")

    data = copy.deepcopy(dict(config))
    model = data.get("model")
    pipeline = data.get("pipeline")

    if not isinstance(model, dict) or not isinstance(
        pipeline, dict
    ):
        raise ValueError("model and pipeline must be objects")

    selector = model.get("quantization")

    if (
        not isinstance(selector, dict)
        or selector.get("provider") != "custom"
    ):
        raise ValueError(
            "Expected custom quantization configuration"
        )

    factory = selector.get("factory")

    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError(
            "Custom quantization requires module:function factory"
        )

    options = selector.get("options", {})

    if not isinstance(options, dict):
        raise ValueError("options must be an object")

    checkpoint = model.get("checkpoint_dir")

    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint_dir is required")

    context = QuantizationContext(
        checkpoint_dir=checkpoint,
        device="cpu",
        activation_dtype=options.get(
            "activation_dtype", "float16"
        ),
        options=options,
    )

    provider_registry = (
        registry if registry is not None
        else QuantizationRegistry()
    )

    provider = provider_registry.create(factory, context)
    profile = provider.inspect_checkpoint(context)

    if not isinstance(profile, QuantizationProfile):
        raise QuantizationProviderError(
            "Custom provider must return QuantizationProfile"
        )

    execution_method = (
        "build_linear"
        if profile.execution_mode == "linear"
        else "load_stage"
    )

    if not callable(
        getattr(provider, execution_method, None)
    ):
        raise QuantizationProviderError(
            f"Provider missing {execution_method}"
        )

    for field, actual in (
        ("bits", profile.bits),
        ("group_size", profile.group_size),
    ):
        if field in options and options[field] != actual:
            raise QuantizationProviderError(
                f"Configured {field} disagrees with checkpoint"
            )

    if "num_layers" in model or "layer_weights_gib" in model:
        raise ValueError(
            "Custom quantization derives layer metadata"
        )

    estimates = profile.planner_inputs()

    model.setdefault("id", Path(checkpoint).name)
    model["num_layers"] = profile.num_layers
    model["layer_weights_gib"] = estimates[
        "layer_weights_gib"
    ]

    memory = pipeline.setdefault("memory", {})

    if not isinstance(memory, dict):
        raise ValueError("pipeline.memory must be an object")

    for name in (
        "first_stage_gib",
        "last_stage_gib",
        "kv_bytes_per_layer_per_token",
    ):
        if name in memory:
            raise ValueError(
                f"{name} must come from the quantization provider"
            )
        memory[name] = estimates[name]

    if gpu_memory_gib is not None:
        memory["gpu_memory_gib"] = gpu_memory_gib

    memory["reserve_gib"] = (
        memory.get("reserve_gib", 2.0)
        + estimates["additional_rank_reserve_gib"]
    )

    plan = resolve_v31_config(data)

    return plan, profile
