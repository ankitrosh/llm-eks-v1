"""Rank-local Qwen3 loader supporting BF16 and GPTQ INT8 checkpoints."""

import gc
import json
import os
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
            raise ValueError("expected rank 0 or 1")

        self.root = Path(model_dir)
        self.cfg = AutoConfig.from_pretrained(str(self.root))

        if self.cfg.model_type != "qwen3" or self.cfg.tie_word_embeddings:
            raise ValueError("expects untied, dense Qwen3 model")

        if not 0 < split < self.cfg.num_hidden_layers:
            raise ValueError("invalid split")

        backend = os.environ.get(
            "LLM_ATTENTION_BACKEND", "eager"
        ).strip().lower()

        if backend not in {"eager", "sdpa"}:
            raise ValueError(
                f"Unsupported attention backend: {backend}"
            )

        self.cfg._attn_implementation = backend

        self.rank = rank
        self.start, self.stop = (
            (0, split)
            if rank == 0
            else (split, self.cfg.num_hidden_layers)
        )

        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16

        self.index = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]

        # Detect quantization from the checkpoint itself.
        quant = getattr(self.cfg, "quantization_config", None)
        self.is_gptq = quant is not None

        if self.is_gptq:
            if not isinstance(quant, dict):
                raise ValueError("Expected dictionary quantization_config")

            supported = (
                quant.get("quant_method") == "gptq"
                and quant.get("checkpoint_format") == "gptq"
                and quant.get("bits") == 8
                and quant.get("group_size") == 128
                and quant.get("sym") is True
                and quant.get("desc_act") is False
                and quant.get("pack_dtype") == "int32"
                and quant.get("lm_head") is False
                and quant.get("hyb_act") is False
                and quant.get("meta", {}).get("v2") is False
            )

            if not supported:
                raise ValueError(
                    "Unsupported GPTQ checkpoint configuration: "
                    f"{quant}"
                )

        print(
            f"rank{rank} attention backend: {backend}",
            flush=True,
        )
        print(
            f"rank{rank} weight format: "
            f"{'gptq-int8-triton' if self.is_gptq else 'bf16'}",
            flush=True,
        )

        # Allocate no parameter storage during construction.
        with torch.device("meta"):
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

            # Replace only checkpoint-quantized projections.
            if self.is_gptq:
                for local_i, layer in enumerate(self.layers):
                    self._install_gptq_linears(
                        layer,
                        global_layer=self.start + local_i,
                    )

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

    @staticmethod
    def _gptq_backend():
        import os

        backend = os.environ.get(
            "LLM_GPTQ_BACKEND", "tritonv2"
        ).strip().lower()

        if backend not in ("tritonv2", "marlin"):
            raise ValueError(
                f"Unsupported GPTQ backend: {backend}. "
                "Expected tritonv2 or marlin."
            )

        return backend

    def _install_gptq_linears(self, layer, global_layer):
        """Replace quantized nn.Linear projections before weight loading."""

        backend = self._gptq_backend()

        if backend == "tritonv2":
            from gptqmodel.nn_modules.qlinear.tritonv2 import (
                TritonV2QuantLinear,
                TRITON_AVAILABLE,
            )

            if not TRITON_AVAILABLE:
                raise RuntimeError(
                    "GPTQModel Triton backend unavailable"
                )

            QuantLinear = TritonV2QuantLinear

        else:
            try:
                import gptqmodel_marlin_kernels
            except ImportError as exc:
                raise RuntimeError(
                    "Marlin CUDA extension is unavailable. "
                    "Use the Marlin-enabled Docker image or "
                    "LLM_GPTQ_BACKEND=tritonv2."
                ) from exc

            from gptqmodel.nn_modules.qlinear.marlin import (
                MarlinQuantLinear,
            )

            QuantLinear = MarlinQuantLinear

        replacements = []

        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            tensor_prefix = (
                f"model.layers.{global_layer}.{name}."
            )

            if tensor_prefix + "qweight" in self.index:
                required = (
                    "qweight", "qzeros", "scales", "g_idx"
                )

                missing = [
                    field for field in required
                    if tensor_prefix + field not in self.index
                ]

                if missing:
                    raise RuntimeError(
                        f"Incomplete GPTQ projection "
                        f"{tensor_prefix}: {missing}"
                    )

                replacements.append((name, module))

            elif tensor_prefix + "weight" not in self.index:
                raise RuntimeError(
                    f"No recognized weights for {tensor_prefix}"
                )

        if not replacements:
            raise RuntimeError(
                f"No quantized projections found in layer {global_layer}"
            )

        for name, original in replacements:
            parent_name, _, attribute = name.rpartition(".")
            parent = (
                layer.get_submodule(parent_name)
                if parent_name else layer
            )

            quantized = QuantLinear(
                bits=8,
                group_size=128,
                desc_act=False,
                sym=True,
                in_features=original.in_features,
                out_features=original.out_features,
                bias=original.bias is not None,
                pack_dtype=torch.int32,
            )

            setattr(parent, attribute, quantized)

    def load_one(self, module, prefix):
        names = [
            name for name in self.index
            if name.startswith(prefix)
        ]

        if not names:
            raise RuntimeError(f"No tensors found for {prefix}")

        weights = {}

        for shard in sorted({
            self.index[name] for name in names
        }):
            with safe_open(
                str(self.root / shard),
                framework="pt",
                device="cpu",
            ) as reader:
                for name in names:
                    if self.index[name] == shard:
                        weights[name[len(prefix):]] = (
                            reader.get_tensor(name)
                        )

        module.load_state_dict(
            weights,
            strict=True,
            assign=True,
        )

        # TritonV2 requires GPTQ v2 zero-point encoding.
        # Marlin consumes the original symmetric GPTQ v1
        # checkpoint format and performs its own repacking.
        if (
            self.is_gptq
            and isinstance(module, Qwen3DecoderLayer)
            and self._gptq_backend() == "tritonv2"
        ):
            from gptqmodel.nn_modules.qlinear.tritonv2 import (
                TritonV2QuantLinear,
            )
            from gptqmodel.utils.model import (
                convert_gptq_v1_to_v2_format_module,
            )

            converted = 0

            for child in module.modules():
                if isinstance(child, TritonV2QuantLinear):
                    convert_gptq_v1_to_v2_format_module(
                        module=child,
                        bits=8,
                        pack_dtype=torch.int32,
                    )
                    converted += 1

            if converted != 7:
                raise RuntimeError(
                    f"Expected 7 GPTQ projections for {prefix}, "
                    f"converted {converted}"
                )

        # For GPTQ layers, preserve packed INT32 tensors and
        # checkpoint scale dtypes. Only cast ordinary RMSNorm
        # weights to the BF16 activation dtype.
        if self.is_gptq and isinstance(module, Qwen3DecoderLayer):
            module = module.to(device=self.device)

            for child in module.modules():
                if isinstance(child, Qwen3RMSNorm):
                    child.to(dtype=self.dtype)

            if self._gptq_backend() == "marlin":
                from gptqmodel.nn_modules.qlinear.marlin import (
                    MarlinQuantLinear,
                )

                initialized = 0

                for child in module.modules():
                    if isinstance(child, MarlinQuantLinear):
                        child.post_init()
                        initialized += 1

                if initialized != 7:
                    raise RuntimeError(
                        f"Expected 7 Marlin projections for {prefix}, "
                        f"initialized {initialized}"
                    )
        else:
            module = module.to(
                device=self.device,
                dtype=self.dtype,
            )

        module = module.eval()

        del weights
        gc.collect()

        return module

    def load(self):
        if self.rank == 0:
            self.embed = self.load_one(
                self.embed,
                "model.embed_tokens.",
            )

        for local_i, global_i in enumerate(
            range(self.start, self.stop)
        ):
            self.layers[local_i] = self.load_one(
                self.layers[local_i],
                f"model.layers.{global_i}.",
            )

            if local_i % 4 == 0 or global_i == self.stop - 1:
                print(
                    f"rank{self.rank} loaded layer {global_i} "
                    f"VRAM={torch.cuda.memory_allocated()/2**30:.2f} GiB",
                    flush=True,
                )

        if self.rank == 1:
            self.norm = self.load_one(
                self.norm,
                "model.norm.",
            )
            self.head = self.load_one(
                self.head,
                "lm_head.",
            )

        return self.eval()

    def make_cache(self):
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

        use_unmasked_sdpa = (
            self.cfg._attn_implementation == "sdpa"
            and (start_position == 0 or q == 1)
        )

        prefix_mask = os.environ.get(
            "LLM_SDPA_PREFIX_MASK", "explicit"
        ).strip().lower()

        if prefix_mask not in {"explicit", "lower_right"}:
            raise ValueError(
                f"Unsupported SDPA prefix mask: {prefix_mask}"
            )

        use_lower_right = (
            self.cfg._attn_implementation == "sdpa"
            and start_position > 0
            and q > 1
            and prefix_mask == "lower_right"
        )

        if use_unmasked_sdpa:
            mask = None
        elif use_lower_right:
            from torch.nn.attention.bias import causal_lower_right

            mask = causal_lower_right(
                q, start_position + q
            )
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
