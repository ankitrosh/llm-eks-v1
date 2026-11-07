"""Qwen3 stage provider for the generic V3 execution contract.

Initial supported subset:
- Decoder-only Qwen3
- Two contiguous stages
- Full causal attention
- Batch size 1 per request
- Eager attention and DynamicCache

This tiny-model validation loads a full checkpoint per worker,
but only executes the worker's assigned transformer layers.
Stage-only checkpoint loading is a separate release requirement.
"""

from dataclasses import dataclass

import torch
from transformers import Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from llm_runtime.v3_pipeline_scheduler import StageResult


from llm_runtime.providers.v3_qwen3_partition import stage_bounds

class Qwen3StageExecutor:
    def __init__(self, model, rank, device, split_at=None):
        self.model = model.eval()
        self.rank = rank
        self.device = torch.device(device)
        self.config = model.config

        count = self.config.num_hidden_layers

        self.start, self.end = stage_bounds(
            count, rank, split_at
        )

        if any(
            kind != "full_attention"
            for kind in self.config.layer_types
        ):
            raise ValueError(
                "Sliding attention is not supported by this provider"
            )

    def _causal_mask(self, hidden, cache_position):
        """Build a full-attention mask from global positions.

        This avoids relying on cache layer zero when a remote
        partition only owns layers two and above.
        """
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
            assert cache is None
            assert task.position == 0
            cache = DynamicCache(config=self.config)
        else:
            assert isinstance(cache, DynamicCache)

        if (
            cache.layers[self.start].get_seq_length()
            != task.position
        ):
            raise RuntimeError(
                "Qwen3 stage cache position mismatch"
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
                        self.device, dtype=torch.long
                    )

                assert ids.ndim == 2
                assert ids.shape[0] == 1

                hidden = self.model.model.embed_tokens(ids)

            else:
                hidden = task.payload

                assert isinstance(hidden, torch.Tensor)
                assert hidden.ndim == 3
                assert hidden.shape[0] == 1
                assert hidden.device == self.device

            length = hidden.shape[1]

            cache_position = torch.arange(
                task.position,
                task.position + length,
                device=self.device,
                dtype=torch.long,
            )

            position_ids = cache_position.unsqueeze(0)

            mask = self._causal_mask(
                hidden, cache_position
            )

            rope = self.model.model.rotary_emb(
                hidden, position_ids
            )

            for idx in range(self.start, self.end):
                layer = self.model.model.layers[idx]

                hidden = layer(
                    hidden_states=hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=rope,
                )

            for idx in range(self.start, self.end):
                assert (
                    cache.layers[idx].get_seq_length()
                    == task.position + length
                )

            if self.rank == 1:
                # For autoregressive generation, only the last
                # position's logits are needed. All prompt positions
                # have already passed through the decoder layers.
                hidden = self.model.model.norm(hidden)
                hidden = hidden[:, -1:, :]
                hidden = self.model.lm_head(hidden)

            return StageResult(
                payload=hidden.contiguous(),
                cache_state=cache,
            )


@dataclass(frozen=True)
class Qwen3ProcessFactory:
    checkpoint_dir: str
    device: str = "cpu"
    split_at: int | None = None

    def __call__(self, rank):
        if rank not in (0, 1):
            raise ValueError(
                "Qwen3 two-stage provider expects rank 0 or 1"
            )

        torch.set_num_threads(1)

        model = Qwen3ForCausalLM.from_pretrained(
            self.checkpoint_dir,
            attn_implementation="eager",
        ).to(self.device).eval()

        return Qwen3StageExecutor(
            model, rank, self.device, self.split_at
        )
