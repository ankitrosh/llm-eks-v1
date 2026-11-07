"""GPT-2 adapter and attention plugins for V3 validation."""

from pathlib import Path

import torch
from torch import nn
from transformers import GPT2LMHeadModel

from llm_runtime.v3_extension_contracts import (
    StageInputs,
    StageOutputs,
    StageSpec,
)


class GPT2Partition(nn.Module):
    def __init__(self, model, spec):
        super().__init__()

        self.spec = spec

        self.wte = (
            model.transformer.wte
            if spec.first_stage else None
        )
        self.wpe = (
            model.transformer.wpe
            if spec.first_stage else None
        )
        self.drop = (
            model.transformer.drop
            if spec.first_stage else None
        )

        self.blocks = nn.ModuleDict({
            str(i): model.transformer.h[i]
            for i in range(
                spec.start_layer, spec.end_layer
            )
        })

        self.ln_f = (
            model.transformer.ln_f
            if spec.final_stage else None
        )
        self.lm_head = (
            model.lm_head
            if spec.final_stage else None
        )


class GPT2Adapter:
    """External two-stage ModelAdapter implementation."""

    def __init__(self):
        self.stages = None

    def validate_partition(self, stages):
        if (
            len(stages) != 2
            or not all(
                isinstance(s, StageSpec) for s in stages
            )
        ):
            raise ValueError("Expected two StageSpec objects")

        a, b = stages

        if not (
            a.rank == 0
            and b.rank == 1
            and a.first_stage
            and not a.final_stage
            and not b.first_stage
            and b.final_stage
            and a.start_layer == 0
            and 0 < a.end_layer == b.start_layer
            < b.end_layer
        ):
            raise ValueError("Invalid GPT-2 partition")

        self.stages = tuple(stages)

    def create_stage(
        self, spec, *, checkpoint, device, dtype
    ):
        if self.stages is None or spec not in self.stages:
            raise ValueError(
                "Validate partition before loading"
            )

        if not Path(checkpoint).is_dir():
            raise ValueError("Checkpoint directory not found")

        if device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("Unsupported device")

        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        if dtype not in dtypes:
            raise ValueError("Unsupported dtype")

        model = GPT2LMHeadModel.from_pretrained(
            checkpoint,
            attn_implementation="eager",
        ).eval()

        if (
            model.config.n_layer
            != self.stages[-1].end_layer
        ):
            raise ValueError(
                "Partition does not cover checkpoint"
            )

        stage = GPT2Partition(model, spec)

        return stage.to(
            device=device,
            dtype=dtypes[dtype],
        ).eval()

    def forward_stage(self, stage, inputs):
        if not isinstance(inputs, StageInputs):
            raise TypeError("Expected StageInputs")

        if inputs.cache_state is not None:
            from transformers.cache_utils import Cache
            if not isinstance(inputs.cache_state, Cache):
                raise TypeError("Expected a Transformers Cache")

        offset = inputs.positions

        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                "positions must be a non-negative integer"
            )

        h = inputs.payload

        if stage.spec.first_stage:
            if h.dtype != torch.long:
                raise ValueError(
                    "Expected integer token IDs"
                )

            positions = torch.arange(
                offset,
                offset + h.shape[1],
                device=h.device,
            )

            h = stage.drop(
                stage.wte(h)
                + stage.wpe(positions.unsqueeze(0))
            )

        for block in stage.blocks.values():
            result = block(
                h,
                past_key_values=inputs.cache_state,
                use_cache=inputs.cache_state is not None,
                attention_mask=inputs.attention_mask,
            )

            h = (
                result[0]
                if isinstance(result, (tuple, list))
                else result
            )

        if stage.spec.final_stage:
            return StageOutputs(
                logits=stage.lm_head(stage.ln_f(h)),
                cache_state=inputs.cache_state,
            )

        return StageOutputs(
            hidden_states=h,
            cache_state=inputs.cache_state,
        )

    def get_native_attention(
        self, stage, *, global_layer_idx
    ):
        key = str(global_layer_idx)

        if key not in stage.blocks:
            raise ValueError(
                "Layer is not owned by this stage"
            )

        return stage.blocks[key].attn

    def replace_attention(
        self, stage, *, global_layer_idx, module
    ):
        if not isinstance(module, nn.Module):
            raise TypeError(
                "Replacement must be an nn.Module"
            )

        self.get_native_attention(
            stage,
            global_layer_idx=global_layer_idx,
        )

        stage.blocks[str(global_layer_idx)].attn = module


class AttentionWrapper(nn.Module):
    def __init__(self, native_module, perturb=0.0):
        super().__init__()
        self.native = native_module
        self.perturb = perturb
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        result = self.native(*args, **kwargs)

        if self.perturb == 0:
            return result

        output = result[0].clone()
        output[..., 0] += self.perturb

        return (output, *result[1:])


class IdentityAttention:
    def create(
        self, *, global_layer_idx,
        native_module, model_config, options
    ):
        return AttentionWrapper(native_module)


class PerturbAttention:
    def create(
        self, *, global_layer_idx,
        native_module, model_config, options
    ):
        return AttentionWrapper(
            native_module,
            perturb=float(options.get("delta", 0.5)),
        )
