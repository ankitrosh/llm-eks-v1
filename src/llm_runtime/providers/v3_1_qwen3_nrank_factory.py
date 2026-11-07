"""V3.1 Qwen3 N-rank selective checkpoint provider.

Supports contiguous N-rank stages with:
- rank 0 embeddings
- intermediate transformer-only ranks
- final-rank norm and LM head
- per-request DynamicCache
- eager full causal attention

Designed for Transformers 4.56.2 and unquantized safetensors.
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from llm_runtime.v3_pipeline_scheduler import StageResult
from llm_runtime.providers.v3_1_qwen3_layout import (
    Qwen3NRankLayout,
)
from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    SelectiveTensorReader,
    StageCheckpointError,
    materialize_module,
)


class NRankPartialQwen3Model(nn.Module):
    """Construct and load only the assigned model partition."""

    def __init__(
        self,
        config,
        layout,
        rank,
        device,
        dtype,
        reader,
    ):
        super().__init__()

        start, end = layout.rank_bounds(rank)

        if any(
            kind != "full_attention"
            for kind in config.layer_types
        ):
            raise StageCheckpointError(
                "Only full causal attention is supported"
            )

        self.config = config
        self.rank = rank
        self.layout = layout
        self.owned_layers = tuple(range(start, end))
        self.loaded_tensor_count = 0

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            nn.Identity()
            for _ in range(config.num_hidden_layers)
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

            if rank == layout.world_size - 1:
                self.model.norm = Qwen3RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                )
                self.lm_head = nn.Linear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                )

        self.model.rotary_emb = Qwen3RotaryEmbedding(
            config,
            device=device,
        )

        if rank == 0:
            self.loaded_tensor_count += materialize_module(
                self.model.embed_tokens,
                "model.embed_tokens.",
                reader,
                device,
                dtype,
            )

        for idx in self.owned_layers:
            self.loaded_tensor_count += materialize_module(
                self.model.layers[idx],
                f"model.layers.{idx}.",
                reader,
                device,
                dtype,
            )

        if rank == layout.world_size - 1:
            self.loaded_tensor_count += materialize_module(
                self.model.norm,
                "model.norm.",
                reader,
                device,
                dtype,
            )

            self.loaded_tensor_count += materialize_module(
                self.lm_head,
                "lm_head.",
                reader,
                device,
                dtype,
                tied=config.tie_word_embeddings,
            )

        for name, parameter in self.named_parameters():
            if parameter.is_meta:
                raise StageCheckpointError(
                    f"Unmaterialized parameter: {name}"
                )

        self.eval()


class Qwen3NRankExecutor:
    """Execute one Qwen3 stage using global layer indices."""

    def __init__(self, model, layout, rank, device):
        self.model = model.eval()
        self.layout = layout
        self.rank = rank
        self.device = torch.device(device)
        self.config = model.config
        self.start, self.end = layout.rank_bounds(rank)

    def _causal_mask(self, hidden, cache_position):
        length = hidden.shape[1]
        total = int(cache_position[-1].item()) + 1

        keys = torch.arange(
            total, device=self.device
        )

        invalid = (
            keys.unsqueeze(0)
            > cache_position.unsqueeze(1)
        )

        mask = torch.zeros(
            (1, 1, length, total),
            device=self.device,
            dtype=hidden.dtype,
        )

        return mask.masked_fill(
            invalid.unsqueeze(0).unsqueeze(0),
            torch.finfo(hidden.dtype).min,
        )

    def __call__(self, task, cache):
        if task.step == 0:
            if cache is not None or task.position != 0:
                raise ValueError(
                    "Initial stage call must have empty cache"
                )

            cache = DynamicCache(
                config=self.config
            )

        elif not isinstance(cache, DynamicCache):
            raise ValueError(
                "Expected existing DynamicCache"
            )

        if (
            cache.layers[self.start].get_seq_length()
            != task.position
        ):
            raise RuntimeError(
                "Stage KV-cache position mismatch"
            )

        with torch.inference_mode():
            if self.rank == 0:
                if isinstance(task.payload, int):
                    ids = torch.tensor(
                        [[task.payload]],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    ids = task.payload.to(
                        self.device,
                        dtype=torch.long,
                    )

                if ids.ndim != 2 or ids.shape[0] != 1:
                    raise ValueError(
                        "Expected token IDs with shape [1, T]"
                    )

                hidden = self.model.model.embed_tokens(
                    ids
                )

            else:
                hidden = task.payload

                if (
                    not isinstance(hidden, torch.Tensor)
                    or hidden.ndim != 3
                    or hidden.shape[0] != 1
                    or hidden.device != self.device
                ):
                    raise ValueError(
                        "Expected local hidden states [1, T, H]"
                    )

            length = hidden.shape[1]

            cache_position = torch.arange(
                task.position,
                task.position + length,
                device=self.device,
                dtype=torch.long,
            )

            position_ids = cache_position.unsqueeze(0)

            attention_mask = self._causal_mask(
                hidden,
                cache_position,
            )

            position_embeddings = (
                self.model.model.rotary_emb(
                    hidden,
                    position_ids,
                )
            )

            for idx in self.model.owned_layers:
                layer = self.model.model.layers[idx]

                hidden = layer(
                    hidden_states=hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                if isinstance(hidden, tuple):
                    hidden = hidden[0]

            for idx in self.model.owned_layers:
                if (
                    cache.layers[idx].get_seq_length()
                    != task.position + length
                ):
                    raise RuntimeError(
                        f"Incorrect KV length at layer {idx}"
                    )

            # Intermediate stages return hidden states.
            # Only the final stage produces logits.
            if self.rank == self.layout.world_size - 1:
                hidden = self.model.model.norm(
                    hidden
                )

                hidden = hidden[:, -1:, :]
                hidden = self.model.lm_head(
                    hidden
                )

            return StageResult(
                payload=hidden.contiguous(),
                cache_state=cache,
            )


@dataclass(frozen=True)
class Qwen3NRankStageOnlyFactory:
    checkpoint_dir: str
    layer_boundaries: tuple[int, ...]
    device: str = "cpu"
    dtype: str = "float32"

    def __call__(self, rank):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {self.dtype}"
            )

        config = Qwen3Config.from_pretrained(
            self.checkpoint_dir,
            local_files_only=True,
        )

        if getattr(config, "quantization_config", None):
            raise StageCheckpointError(
                "Quantized checkpoints are not supported"
            )

        layout = Qwen3NRankLayout(
            config.num_hidden_layers,
            tuple(self.layer_boundaries),
        )

        layout.rank_bounds(rank)
        config._attn_implementation = "eager"

        device = torch.device(self.device)

        with SelectiveTensorReader(
            self.checkpoint_dir
        ) as reader:
            partial = NRankPartialQwen3Model(
                config,
                layout,
                rank,
                device,
                dtype_map[self.dtype],
                reader,
            )

        return Qwen3NRankExecutor(
            partial,
            layout,
            rank,
            device,
        )
