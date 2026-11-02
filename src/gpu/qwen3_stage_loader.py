"""Rank-local Qwen3 loader: Transformers 4.56.2, BF16, eager attention."""
import gc
import os
import json
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


class Qwen3Stage(nn.Module):
    def __init__(self, model_dir, rank, split=20):
        super().__init__()

        if rank not in (0, 1):
            raise ValueError('expected rank 0 or 1')

        self.root = Path(model_dir)
        self.cfg = AutoConfig.from_pretrained(str(self.root))

        if self.cfg.model_type != 'qwen3' or self.cfg.tie_word_embeddings:
            raise ValueError('expects untied, dense Qwen3 model')

        if not 0 < split < self.cfg.num_hidden_layers:
            raise ValueError('invalid split')

        # V2: runtime-selectable attention, with V1 as the default.
        backend = os.environ.get(
            "LLM_ATTENTION_BACKEND", "eager"
        ).strip().lower()

        if backend not in {"eager", "sdpa"}:
            raise ValueError(
                f"Unsupported attention backend: {backend}. "
                "Currently supported: eager, sdpa"
            )

        self.cfg._attn_implementation = backend
        print(
            f"rank{rank} attention backend: {backend}",
            flush=True,
        )
        self.rank = rank

        self.start, self.stop = (
            (0, split)
            if rank == 0
            else (split, self.cfg.num_hidden_layers)
        )

        self.device = torch.device('cuda:0')
        self.dtype = torch.bfloat16

        self.index = json.loads(
            (self.root / 'model.safetensors.index.json').read_text()
        )['weight_map']

        # Allocate no parameter storage during model construction.
        with torch.device('meta'):
            if rank == 0:
                self.embed = nn.Embedding(
                    self.cfg.vocab_size,
                    self.cfg.hidden_size,
                    padding_idx=self.cfg.pad_token_id,
                )

            self.layers = nn.ModuleList([
                Qwen3DecoderLayer(self.cfg, i)
                for i in range(self.start, self.stop)
            ])

            if rank == 1:
                self.norm = Qwen3RMSNorm(
                    self.cfg.hidden_size,
                    eps=self.cfg.rms_norm_eps,
                )
                self.head = nn.Linear(
                    self.cfg.hidden_size,
                    self.cfg.vocab_size,
                    bias=False,
                )

        self.rope = Qwen3RotaryEmbedding(self.cfg).to(self.device)

    def load_one(self, module, prefix):
        names = [
            name for name in self.index
            if name.startswith(prefix)
        ]

        if not names:
            raise RuntimeError(f'No tensors found for {prefix}')

        weights = {}

        for shard in sorted({
            self.index[name] for name in names
        }):
            with safe_open(
                str(self.root / shard),
                framework='pt',
                device='cpu',
            ) as reader:
                for name in names:
                    if self.index[name] == shard:
                        weights[name[len(prefix):]] = reader.get_tensor(name)

        module.load_state_dict(
            weights,
            strict=True,
            assign=True,
        )

        # Transfer only this one module to GPU.
        module = module.to(
            device=self.device,
            dtype=self.dtype,
        ).eval()

        del weights
        gc.collect()

        return module

    def load(self):
        if self.rank == 0:
            self.embed = self.load_one(
                self.embed,
                'model.embed_tokens.',
            )

        for local_i, global_i in enumerate(
            range(self.start, self.stop)
        ):
            self.layers[local_i] = self.load_one(
                self.layers[local_i],
                f'model.layers.{global_i}.',
            )

            if local_i % 4 == 0 or global_i == self.stop - 1:
                print(
                    f'rank{self.rank} loaded layer {global_i} '
                    f'VRAM={torch.cuda.memory_allocated()/2**30:.2f} GiB',
                    flush=True,
                )

        if self.rank == 1:
            self.norm = self.load_one(
                self.norm,
                'model.norm.',
            )
            self.head = self.load_one(
                self.head,
                'lm_head.',
            )

        return self.eval()

    def make_cache(self):
        # Global layer IDs are retained across partition boundaries.
        return DynamicCache(config=self.cfg)

    def forward_local(self, data, start_position, cache):
        h = self.embed(data) if self.rank == 0 else data

        b, q, d = h.shape
        assert b == 1 and d == self.cfg.hidden_size

        pos = torch.arange(
            start_position,
            start_position + q,
            device=self.device,
        )

        # Mask-free SDPA is safe for:
        # 1. Fresh causal prefill (no previous KV cache).
        # 2. Single-token decode (all cached keys are visible).
        #
        # Keep the explicit mask for cached multi-token queries,
        # where non-square causal alignment needs special care.
        use_unmasked_sdpa = (
            self.cfg._attn_implementation == 'sdpa'
            and (start_position == 0 or q == 1)
        )

        if use_unmasked_sdpa:
            mask = None
        else:
            key = torch.arange(
                start_position + q,
                device=self.device,
            )

            visible = key[None, :] <= pos[:, None]

            mask = torch.zeros(
                (1, 1, q, start_position + q),
                dtype=self.dtype,
                device=self.device,
            )

            mask.masked_fill_(
                ~visible[None, None, :, :],
                torch.finfo(self.dtype).min,
            )

        positions = pos.unsqueeze(0)
        rotary = self.rope(h, positions)

        for layer in self.layers:
            h = layer(
                h,
                attention_mask=mask,
                position_ids=positions,
                cache_position=pos,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=rotary,
            )

            if isinstance(h, tuple):
                h = h[0]

        if self.rank == 0:
            return h.contiguous()

        return self.head(
            self.norm(h[:, -1:, :])
        )[:, -1, :].contiguous()
