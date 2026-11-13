"""CPU-only Qwen3 GPTQ INT8 checkpoint inspection and placement.

This module estimates checkpoint-storage-derived memory.
It does not load or execute quantized layers.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from llm_runtime.v3_1_checkpoint_profile import (
    _checkpoint_tensor_sizes,
)
from llm_runtime.v3_1_quantization_api import (
    GIB,
    QuantizationProfile,
    QuantizationProviderError,
)
from llm_runtime.v3_1_runtime_config import (
    resolve_v31_config,
)


_LAYER = re.compile(r"^model\.layers\.(\d+)\.")

_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


def _positive_int(value, field):
    if type(value) is not int or value < 1:
        raise QuantizationProviderError(
            f"{field} must be a positive integer"
        )
    return value


def inspect_qwen3_gptq_int8(checkpoint_dir):
    """Inspect GPTQ INT8 storage; do not instantiate a model."""

    root = Path(checkpoint_dir).expanduser().resolve()
    config_file = root / "config.json"

    if not config_file.is_file():
        raise QuantizationProviderError(
            f"Missing checkpoint config: {config_file}"
        )

    config = json.loads(config_file.read_text())

    if config.get("model_type") != "qwen3":
        raise QuantizationProviderError(
            "Only standard Qwen3 is supported here"
        )

    quant = config.get("quantization_config")

    if not isinstance(quant, dict):
        raise QuantizationProviderError(
            "Missing quantization_config"
        )

    method = quant.get(
        "quant_method", quant.get("method")
    )

    if method != "gptq" or type(quant.get("bits")) is not int:
        raise QuantizationProviderError(
            "Expected a GPTQ checkpoint with explicit bits"
        )

    if quant["bits"] != 8:
        raise QuantizationProviderError(
            "This inspector supports GPTQ INT8 only"
        )

    official_format = quant.get("checkpoint_format")
    legacy_format = quant.get("format")

    if (
        official_format is not None
        and legacy_format is not None
        and official_format != legacy_format
    ):
        raise QuantizationProviderError(
            "Conflicting GPTQ checkpoint format metadata"
        )

    checkpoint_format = (
        official_format
        if official_format is not None
        else legacy_format
        if legacy_format is not None
        else "gptq"
    )

    if checkpoint_format not in ("gptq", "gptq_v2"):
        raise QuantizationProviderError(
            f"Unsupported GPTQ storage format: "
            f"{checkpoint_format}"
        )

    if quant.get("dynamic"):
        raise QuantizationProviderError(
            "Mixed/dynamic quantization requires "
            "a format-aware provider"
        )

    group_size = quant.get("group_size", 128)

    if (
        type(group_size) is not int
        or (group_size != -1 and group_size < 1)
    ):
        raise QuantizationProviderError(
            "Invalid GPTQ group_size"
        )

    layers = _positive_int(
        config.get("num_hidden_layers"),
        "num_hidden_layers",
    )

    heads = _positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )

    kv_heads = _positive_int(
        config.get("num_key_value_heads", heads),
        "num_key_value_heads",
    )

    hidden = _positive_int(
        config.get("hidden_size"),
        "hidden_size",
    )

    head_dim = config.get("head_dim")

    if head_dim is None:
        if hidden % heads:
            raise QuantizationProviderError(
                "Cannot derive attention head dimension"
            )
        head_dim = hidden // heads

    head_dim = _positive_int(head_dim, "head_dim")

    dtype = config.get("dtype") or config.get("torch_dtype")

    if dtype not in _DTYPE_BYTES:
        raise QuantizationProviderError(
            f"Unsupported KV-cache dtype: {dtype}"
        )

    # Reuses our existing SafeTensors/index validation.
    # Counts stored bytes, including GPTQ auxiliary tensors.
    tensors = _checkpoint_tensor_sizes(root)

    layer_bytes = [0] * layers
    first_bytes = 0
    last_bytes = 0

    for name, size in tensors.items():
        match = _LAYER.match(name)

        if match:
            index = int(match.group(1))

            if index >= layers:
                raise QuantizationProviderError(
                    f"Invalid layer index: {name}"
                )

            layer_bytes[index] += size

        elif name.startswith((
            "model.embed_tokens.",
            "model.rotary_emb.",
        )):
            first_bytes += size

        elif name.startswith((
            "model.norm.",
            "lm_head.",
        )):
            last_bytes += size

        else:
            raise QuantizationProviderError(
                f"Unrecognized checkpoint tensor: {name}"
            )

    # Validate that this is actually a packed GPTQ checkpoint.
    prefixes = {
        name[:-len(".qweight")]
        for name in tensors
        if name.endswith(".qweight")
    }

    if not prefixes:
        raise QuantizationProviderError(
            "No GPTQ packed qweight tensors found"
        )

    for prefix in prefixes:
        for suffix in ("scales", "qzeros"):
            key = f"{prefix}.{suffix}"

            if key not in tensors:
                raise QuantizationProviderError(
                    f"Missing GPTQ tensor: {key}"
                )

    if not all(layer_bytes):
        raise QuantizationProviderError(
            "One or more layers have no stored tensors"
        )

    embedding = "model.embed_tokens.weight"

    if embedding not in tensors:
        raise QuantizationProviderError(
            "Missing embedding weight"
        )

    if "lm_head.weight" not in tensors:
        if config.get("tie_word_embeddings") is True:
            # A separate final-stage projection copy may be
            # required when embedding and head are tied.
            last_bytes += tensors[embedding]
        elif not any(
            name.startswith("lm_head.")
            for name in tensors
        ):
            raise QuantizationProviderError(
                "Missing output projection"
            )

    kv_bytes = (
        2
        * kv_heads
        * head_dim
        * _DTYPE_BYTES[dtype]
    )

    return QuantizationProfile(
        provider="gptq",
        format_id=checkpoint_format,
        bits=8,
        group_size=group_size,
        layer_storage_bytes=tuple(layer_bytes),
        first_stage_bytes=first_bytes,
        last_stage_bytes=last_bytes,
        kv_bytes_per_layer_per_token=kv_bytes,
        workspace_bytes_per_rank=0,
        execution_mode="linear",
    )


def resolve_gptq_int8_plan(config, *, gpu_memory_gib=None):
    """Use GPTQ metadata with the existing N-rank planner.

    This only produces a proposed layout. It does not guarantee
    runtime memory fit or availability of GPTQ execution kernels.
    """

    data = copy.deepcopy(config)

    if not isinstance(data, dict):
        raise ValueError("Configuration must be an object")

    model = data.get("model")
    pipeline = data.get("pipeline")

    if not isinstance(model, dict) or not isinstance(pipeline, dict):
        raise ValueError("model and pipeline must be objects")

    quant = model.get("quantization", {})

    if not isinstance(quant, dict) or quant.get("provider") != "gptq":
        raise ValueError(
            "model.quantization.provider must be gptq"
        )

    options = quant.get("options", {})

    if not isinstance(options, dict):
        raise ValueError(
            "model.quantization.options must be an object"
        )

    checkpoint = model.get("checkpoint_dir")

    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("model.checkpoint_dir is required")

    profile = inspect_qwen3_gptq_int8(checkpoint)

    if "bits" in options and options["bits"] != profile.bits:
        raise ValueError(
            "Configured bit width disagrees with checkpoint"
        )

    if "group_size" in options and (
        options["group_size"] != profile.group_size
    ):
        raise ValueError(
            "Configured group size disagrees with checkpoint"
        )

    if "layer_weights_gib" in model or "num_layers" in model:
        raise ValueError(
            "Quantized checkpoint mode derives layer metadata"
        )

    model.setdefault("id", Path(checkpoint).name)
    model["num_layers"] = profile.num_layers
    model["layer_weights_gib"] = [
        size / GIB
        for size in profile.layer_storage_bytes
    ]

    memory = pipeline.setdefault("memory", {})

    if not isinstance(memory, dict):
        raise ValueError("pipeline.memory must be an object")

    derived = {
        "first_stage_gib": profile.first_stage_bytes / GIB,
        "last_stage_gib": profile.last_stage_bytes / GIB,
        "kv_bytes_per_layer_per_token":
            profile.kv_bytes_per_layer_per_token,
    }

    for name, value in derived.items():
        if name in memory:
            raise ValueError(
                f"{name} is checkpoint-derived"
            )
        memory[name] = value

    if gpu_memory_gib is not None:
        memory["gpu_memory_gib"] = gpu_memory_gib

    memory["reserve_gib"] = (
        memory.get("reserve_gib", 2.0)
        + profile.workspace_bytes_per_rank / GIB
    )

    plan = resolve_v31_config(data)

    return plan, profile
