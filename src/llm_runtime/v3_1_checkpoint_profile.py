"""CPU-only Qwen3 checkpoint inspection and placement resolution."""

from __future__ import annotations

import copy
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from llm_runtime.v3_1_runtime_config import resolve_v31_config


GIB = 1024 ** 3

STORAGE_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

KV_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}

LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class Qwen3CheckpointProfile:
    checkpoint_dir: str
    num_layers: int
    dtype: str
    tensor_count: int
    layer_weights_gib: tuple[float, ...]
    first_stage_gib: float
    last_stage_gib: float
    kv_bytes_per_layer_per_token: int

    def as_dict(self):
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "num_layers": self.num_layers,
            "dtype": self.dtype,
            "tensor_count": self.tensor_count,
            "layer_weights_gib": list(self.layer_weights_gib),
            "first_stage_gib": self.first_stage_gib,
            "last_stage_gib": self.last_stage_gib,
            "kv_bytes_per_layer_per_token":
                self.kv_bytes_per_layer_per_token,
        }


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_safetensors_header(path):
    """Read tensor sizes from metadata, not actual weight data."""

    file_size = path.stat().st_size

    with path.open("rb") as handle:
        raw = handle.read(8)

        if len(raw) != 8:
            raise ValueError(f"Invalid SafeTensors file: {path}")

        header_length = struct.unpack("<Q", raw)[0]

        if (
            header_length < 2
            or header_length > 32 * 1024 * 1024
            or header_length + 8 > file_size
        ):
            raise ValueError(
                f"Invalid SafeTensors header length: {path}"
            )

        header = json.loads(handle.read(header_length))

    if not isinstance(header, dict):
        raise ValueError(f"Malformed SafeTensors header: {path}")

    tensors = {}

    for name, entry in header.items():
        if name == "__metadata__":
            continue

        if not isinstance(entry, dict):
            raise ValueError(f"Invalid tensor metadata: {name}")

        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")

        if dtype not in STORAGE_DTYPE_BYTES:
            raise ValueError(
                f"Unsupported storage dtype {dtype!r}: {name}"
            )

        if (
            not isinstance(shape, list)
            or not shape
            or any(type(x) is not int or x < 1 for x in shape)
        ):
            raise ValueError(f"Invalid tensor shape: {name}")

        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int for x in offsets)
        ):
            raise ValueError(f"Invalid tensor offsets: {name}")

        start, end = offsets
        nbytes = math.prod(shape) * STORAGE_DTYPE_BYTES[dtype]

        if (
            start < 0
            or end - start != nbytes
            or end > file_size - 8 - header_length
        ):
            raise ValueError(
                f"Tensor size or offsets are invalid: {name}"
            )

        tensors[name] = nbytes

    return tensors


def _checkpoint_tensor_sizes(root):
    index_file = root / "model.safetensors.index.json"
    expected = None

    if index_file.exists():
        index = json.loads(
            index_file.read_text(encoding="utf-8")
        )

        expected = index.get("weight_map")

        if not isinstance(expected, dict) or not expected:
            raise ValueError("Invalid SafeTensors weight_map")

        shard_names = set(expected.values())

        if any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            for name in shard_names
        ):
            raise ValueError("Invalid shard name in checkpoint index")

        paths = [
            root / name for name in sorted(shard_names)
        ]
    else:
        paths = sorted(root.glob("*.safetensors"))

    if not paths:
        raise ValueError("No SafeTensors files found")

    tensors = {}

    for path in paths:
        for name, nbytes in _read_safetensors_header(path).items():
            if name in tensors:
                raise ValueError(
                    f"Duplicate checkpoint tensor: {name}"
                )

            if expected is not None and expected.get(name) != path.name:
                raise ValueError(
                    f"Checkpoint index mismatch: {name}"
                )

            tensors[name] = nbytes

    if expected is not None and set(tensors) != set(expected):
        raise ValueError(
            "Checkpoint tensors do not match the weight_map"
        )

    return tensors


def discover_qwen3_checkpoint(checkpoint_dir):
    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "config.json"

    if not config_path.is_file():
        raise ValueError(f"Missing config.json: {root}")

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if config.get("model_type") != "qwen3":
        raise ValueError(
            "Automatic checkpoint estimation currently supports Qwen3"
        )

    if config.get("quantization_config") is not None:
        raise ValueError(
            "Quantized checkpoints require a separate "
            "runtime-memory estimator"
        )

    num_layers = _positive_int(
        config.get("num_hidden_layers"),
        "num_hidden_layers",
    )

    num_heads = _positive_int(
        config.get("num_attention_heads"),
        "num_attention_heads",
    )

    num_kv_heads = _positive_int(
        config.get("num_key_value_heads", num_heads),
        "num_key_value_heads",
    )

    hidden_size = _positive_int(
        config.get("hidden_size"),
        "hidden_size",
    )

    head_dim = config.get("head_dim")

    if head_dim is None:
        if hidden_size % num_heads:
            raise ValueError("Cannot derive attention head_dim")

        head_dim = hidden_size // num_heads

    head_dim = _positive_int(head_dim, "head_dim")

    dtype = config.get("dtype") or config.get("torch_dtype")

    if dtype not in KV_DTYPE_BYTES:
        raise ValueError(
            f"Unsupported or missing runtime dtype: {dtype!r}"
        )

    tensors = _checkpoint_tensor_sizes(root)

    layer_bytes = [0] * num_layers
    first_bytes = 0
    last_bytes = 0

    for name, nbytes in tensors.items():
        match = LAYER_PATTERN.match(name)

        if match:
            layer = int(match.group(1))

            if layer >= num_layers:
                raise ValueError(
                    f"Out-of-range decoder layer: {name}"
                )

            layer_bytes[layer] += nbytes

        elif (
            name.startswith("model.embed_tokens.")
            or name.startswith("model.rotary_emb.")
        ):
            first_bytes += nbytes

        elif (
            name.startswith("model.norm.")
            or name.startswith("lm_head.")
        ):
            last_bytes += nbytes

        else:
            raise ValueError(
                f"Unrecognized Qwen3 checkpoint tensor: {name}"
            )

    if not all(layer_bytes):
        raise ValueError(
            "One or more decoder layers have no checkpoint weights"
        )

    if "model.embed_tokens.weight" not in tensors:
        raise ValueError("Missing input embedding weight")

    if "lm_head.weight" not in tensors:
        if config.get("tie_word_embeddings") is True:
            # The last rank still needs an output projection.
            # Budget a separate local copy of the embedding.
            last_bytes += tensors["model.embed_tokens.weight"]
        else:
            raise ValueError("Missing untied output head")

    kv_bytes = (
        2
        * num_kv_heads
        * head_dim
        * KV_DTYPE_BYTES[dtype]
    )

    return Qwen3CheckpointProfile(
        checkpoint_dir=str(root),
        num_layers=num_layers,
        dtype=dtype,
        tensor_count=len(tensors),
        layer_weights_gib=tuple(
            size / GIB for size in layer_bytes
        ),
        first_stage_gib=first_bytes / GIB,
        last_stage_gib=last_bytes / GIB,
        kv_bytes_per_layer_per_token=kv_bytes,
    )


def resolve_v31_checkpoint_config(
    config,
    *,
    gpu_memory_gib=None,
):
    """Discover model memory inputs and invoke the existing planner."""

    data = copy.deepcopy(config)

    model = data.get("model")
    pipeline = data.get("pipeline")

    if not isinstance(model, dict) or not isinstance(pipeline, dict):
        raise ValueError("model and pipeline must be objects")

    if "layer_weights_gib" in model:
        raise ValueError(
            "Checkpoint mode does not accept manual layer weights"
        )

    checkpoint_dir = model.get("checkpoint_dir")

    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("model.checkpoint_dir is required")

    profile = discover_qwen3_checkpoint(checkpoint_dir)

    if (
        "num_layers" in model
        and model["num_layers"] != profile.num_layers
    ):
        raise ValueError(
            "Configured num_layers disagrees with checkpoint"
        )

    model.setdefault("id", Path(checkpoint_dir).name)
    model["num_layers"] = profile.num_layers
    model["layer_weights_gib"] = list(
        profile.layer_weights_gib
    )

    memory = pipeline.setdefault("memory", {})

    if not isinstance(memory, dict):
        raise ValueError("pipeline.memory must be an object")

    discovered = {
        "kv_bytes_per_layer_per_token":
            profile.kv_bytes_per_layer_per_token,
        "first_stage_gib": profile.first_stage_gib,
        "last_stage_gib": profile.last_stage_gib,
    }

    for name, value in discovered.items():
        if name in memory:
            raise ValueError(
                f"{name} is checkpoint-derived; "
                "remove the manual override"
            )

        memory[name] = value

    if gpu_memory_gib is not None:
        memory["gpu_memory_gib"] = gpu_memory_gib

    plan = resolve_v31_config(data)

    return plan, profile
