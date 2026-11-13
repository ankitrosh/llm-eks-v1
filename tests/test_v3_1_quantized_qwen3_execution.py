"""CPU execution validation for a user-defined INT8 adapter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)
from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    StageCheckpointError,
)


class ReferenceINT8Linear(nn.Module):
    """CPU reference: per-output-channel symmetric INT8."""

    def __init__(self, weight, bias=None):
        super().__init__()

        weight = weight.detach().float()

        scale = (
            weight.abs()
            .amax(dim=1, keepdim=True)
            .clamp_min(1e-7) / 127.0
        )

        qweight = (
            (weight / scale)
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )

        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

        self.register_buffer(
            "bias",
            None if bias is None
            else bias.detach().float().clone(),
        )

    def forward(self, hidden):
        weight = (
            self.qweight.to(hidden.dtype)
            * self.scale.to(hidden.dtype)
        )

        bias = (
            None if self.bias is None
            else self.bias.to(hidden.dtype)
        )

        return F.linear(hidden, weight, bias)


class ReferenceINT8Provider:
    def __init__(self):
        self.loaded_names = []

    def build_linear(
        self, *,
        module_name,
        in_features,
        out_features,
        bias,
        reader,
        device,
        dtype,
    ):
        weight = reader.get(module_name + ".weight")

        assert tuple(weight.shape) == (
            out_features, in_features
        )

        bias_value = (
            reader.get(module_name + ".bias")
            if bias else None
        )

        result = ReferenceINT8Linear(
            weight, bias_value
        ).to(device=device)

        self.loaded_names.append(module_name)

        return result


def quantize_reference_model(model):
    """Apply identical INT8 conversion to full reference."""

    for layer in model.model.layers:
        candidates = [
            (name, module)
            for name, module in layer.named_modules()
            if name and isinstance(module, nn.Linear)
        ]

        for name, original in candidates:
            parent_name, _, child_name = (
                name.rpartition(".")
            )

            parent = (
                layer.get_submodule(parent_name)
                if parent_name else layer
            )

            setattr(
                parent,
                child_name,
                ReferenceINT8Linear(
                    original.weight,
                    original.bias,
                ),
            )

    return model.eval()


def create_checkpoint(tmp_path):
    torch.manual_seed(42)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )

    config._attn_implementation = "eager"

    model = Qwen3ForCausalLM(config).eval()

    model.save_pretrained(
        str(tmp_path),
        safe_serialization=True,
    )

    return quantize_reference_model(
        copy.deepcopy(model)
    )


def test_custom_int8_three_stage_reference_parity(tmp_path):
    reference = create_checkpoint(tmp_path)

    provider = ReferenceINT8Provider()

    factory = Qwen3NRankStageOnlyFactory(
        checkpoint_dir=str(tmp_path),
        layer_boundaries=(0, 2, 4, 6),
        device="cpu",
        dtype="float32",
        quantization_provider=provider,
    )

    executors = [
        factory(rank)
        for rank in range(3)
    ]

    # Qwen3 has seven checkpoint-backed linear
    # projections per decoder layer.
    assert len(provider.loaded_names) == 42

    assert all(
        executor.model.quantized_linear_count == 14
        for executor in executors
    )

    for rank, executor in enumerate(executors):
        start = rank * 2
        end = start + 2

        assert all(
            any(
                name.startswith(
                    f"model.layers.{layer}."
                )
                for layer in range(start, end)
            )
            for name in provider.loaded_names[
                rank * 14:(rank + 1) * 14
            ]
        )

        assert all(
            not param.is_meta
            for param in executor.model.parameters()
        )

    input_ids = torch.tensor(
        [[4, 7, 9]],
        dtype=torch.long,
    )

    payload = input_ids

    with torch.inference_mode():
        for executor in executors:
            task = SimpleNamespace(
                step=0,
                position=0,
                payload=payload,
            )

            result = executor(task, None)
            payload = result.payload

        expected = reference(
            input_ids=input_ids,
            use_cache=False,
        ).logits[:, -1:, :]

    assert payload.shape == expected.shape

    torch.testing.assert_close(
        payload,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_quantization_provider_must_materialize_modules(
    tmp_path,
):
    create_checkpoint(tmp_path)

    class BrokenProvider:
        def build_linear(
            self, *,
            in_features,
            out_features,
            bias,
            **kwargs,
        ):
            return nn.Linear(
                in_features,
                out_features,
                bias=bias,
                device="meta",
            )

    factory = Qwen3NRankStageOnlyFactory(
        checkpoint_dir=str(tmp_path),
        layer_boundaries=(0, 3, 6),
        device="cpu",
        dtype="float32",
        quantization_provider=BrokenProvider(),
    )

    with pytest.raises(
        StageCheckpointError,
        match="meta tensor",
    ):
        factory(0)
