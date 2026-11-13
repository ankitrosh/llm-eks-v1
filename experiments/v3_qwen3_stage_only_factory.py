"""Qwen3 stage-only loader for the V3 generic execution contract.

Supported:
  - Transformers 4.56.2 Qwen3
  - Two contiguous stages with an even number of layers
  - Full attention, eager backend, DynamicCache
  - Unquantized safetensors (single-file or sharded)

Only owned modules are constructed and materialized.
The complete Qwen3ForCausalLM is never instantiated here.
"""

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from experiments.v3_qwen3_partition import stage_bounds
from experiments.v3_qwen3_process_factory import (
    Qwen3StageExecutor,
)


class StageCheckpointError(RuntimeError):
    pass


class SelectiveTensorReader:
    """Read only requested named tensors from safetensors."""

    def __init__(self, checkpoint_dir):
        self.root = Path(checkpoint_dir).resolve()
        self.stack = ExitStack()
        self.handles = {}

        index = self.root / "model.safetensors.index.json"

        if index.exists():
            document = json.loads(index.read_text())
            self.weight_map = document["weight_map"]
            self.default_file = None
        else:
            self.weight_map = None
            self.default_file = "model.safetensors"

            if not (self.root / self.default_file).is_file():
                raise StageCheckpointError(
                    "Expected model.safetensors or its shard index"
                )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def get(self, name):
        if self.weight_map is None:
            filename = self.default_file
        else:
            filename = self.weight_map.get(name)

        if filename is None:
            raise StageCheckpointError(
                f"Missing checkpoint tensor: {name}"
            )

        path = (self.root / filename).resolve()

        if not path.is_relative_to(self.root):
            raise StageCheckpointError(
                "Checkpoint shard path escapes directory"
            )

        if not path.is_file():
            raise StageCheckpointError(
                f"Missing checkpoint shard: {filename}"
            )

        if filename not in self.handles:
            self.handles[filename] = (
                self.stack.enter_context(
                    safe_open(
                        str(path),
                        framework="pt",
                        device="cpu",
                    )
                )
            )

        handle = self.handles[filename]

        if name not in handle.keys():
            raise StageCheckpointError(
                f"Tensor {name} absent from {filename}"
            )

        return handle.get_tensor(name)


def materialize_module(
    module, prefix, reader, device, dtype, *, tied=False
):
    """Load only one assigned module's parameters/buffers."""

    expected = module.state_dict()
    loaded = {}

    for local_name, specification in expected.items():
        checkpoint_name = prefix + local_name

        if tied and checkpoint_name == "lm_head.weight":
            try:
                value = reader.get(checkpoint_name)
            except StageCheckpointError:
                value = reader.get(
                    "model.embed_tokens.weight"
                )
        else:
            value = reader.get(checkpoint_name)

        if tuple(value.shape) != tuple(specification.shape):
            raise StageCheckpointError(
                f"Shape mismatch: {checkpoint_name}"
            )

        if value.is_floating_point():
            value = value.to(
                device=device,
                dtype=dtype,
            )
        else:
            value = value.to(device=device)

        loaded[local_name] = value

    module.load_state_dict(
        loaded, strict=True, assign=True
    )

    if any(p.is_meta for p in module.parameters()):
        raise StageCheckpointError(
            f"Unmaterialized parameters in {prefix}"
        )

    return len(loaded)


class PartialQwen3Model(nn.Module):
    """Qwen3-compatible module tree containing one partition."""

    def __init__(self, config, rank, device, dtype, reader, split_at=None):
        super().__init__()

        count = config.num_hidden_layers

        start, end = stage_bounds(count, rank, split_at)

        if any(
            kind != "full_attention"
            for kind in config.layer_types
        ):
            raise StageCheckpointError(
                "Sliding attention is not yet supported"
            )

        self.config = config
        self.model = nn.Module()

        self.owned_layers = tuple(range(start, end))
        self.loaded_tensor_count = 0

        # Non-owned positions contain parameter-free placeholders.
        # They are never executed.
        self.model.layers = nn.ModuleList(
            nn.Identity() for _ in range(count)
        )

        with torch.device("meta"):
            for idx in self.owned_layers:
                self.model.layers[idx] = Qwen3DecoderLayer(
                    config, idx
                )

            if rank == 0:
                self.model.embed_tokens = nn.Embedding(
                    config.vocab_size,
                    config.hidden_size,
                    config.pad_token_id,
                )
            else:
                self.model.norm = Qwen3RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                )
                self.lm_head = nn.Linear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                )

        # RoPE has a small, derived buffer, not checkpoint weights.
        self.model.rotary_emb = Qwen3RotaryEmbedding(
            config, device=device
        )

        if rank == 0:
            self.loaded_tensor_count += materialize_module(
                self.model.embed_tokens,
                "model.embed_tokens.",
                reader, device, dtype,
            )

        for idx in self.owned_layers:
            self.loaded_tensor_count += materialize_module(
                self.model.layers[idx],
                f"model.layers.{idx}.",
                reader, device, dtype,
            )

        if rank == 1:
            self.loaded_tensor_count += materialize_module(
                self.model.norm,
                "model.norm.",
                reader, device, dtype,
            )
            self.loaded_tensor_count += materialize_module(
                self.lm_head,
                "lm_head.",
                reader, device, dtype,
                tied=config.tie_word_embeddings,
            )

        for name, parameter in self.named_parameters():
            if parameter.is_meta:
                raise StageCheckpointError(
                    f"Parameter remains on meta device: {name}"
                )

        self.eval()


@dataclass(frozen=True)
class Qwen3StageOnlyFactory:
    checkpoint_dir: str
    device: str = "cpu"
    dtype: str = "float32"
    split_at: int | None = None

    def __call__(self, rank):
        if rank not in (0, 1):
            raise ValueError("Expected rank 0 or 1")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.dtype not in dtype_map:
            raise ValueError(
                f"Unsupported compute dtype: {self.dtype}"
            )

        config = Qwen3Config.from_pretrained(
            self.checkpoint_dir,
            local_files_only=True,
        )

        if getattr(config, "quantization_config", None):
            raise StageCheckpointError(
                "Quantized checkpoint requires a separate "
                "quantization-aware stage loader"
            )

        config._attn_implementation = "eager"

        with SelectiveTensorReader(
            self.checkpoint_dir
        ) as reader:
            partial = PartialQwen3Model(
                config,
                rank,
                torch.device(self.device),
                dtype_map[self.dtype],
                reader,
                self.split_at,
            )

        return Qwen3StageExecutor(
            partial, rank, self.device, self.split_at
        )
