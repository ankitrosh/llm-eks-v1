import json

import pytest

torch = pytest.importorskip("torch")

from llm_runtime.v3_1_quantization_api import (
    QuantizationContext,
    QuantizationProviderError,
)
from llm_runtime.providers.v3_1_gptq_marlin_provider import (
    GPTQMarlinProvider,
)

from test_v3_1_gptq_int8_qwen3_checkpoint import (
    create_packed_qwen3_checkpoint,
)


def prepare(tmp_path, fmt="gptq"):
    create_packed_qwen3_checkpoint(
        tmp_path, fmt, group_size=128
    )

    path = tmp_path / "config.json"
    config = json.loads(path.read_text())

    config["quantization_config"].update({
        "sym": True,
        "desc_act": False,
        "pack_dtype": "int32",
    })

    path.write_text(json.dumps(config))

    return QuantizationContext(
        checkpoint_dir=str(tmp_path),
        device="cpu",
        activation_dtype="float32",
    )


def test_marlin_checkpoint_inspection(tmp_path):
    context = prepare(tmp_path)
    provider = GPTQMarlinProvider(context)

    profile = provider.inspect_checkpoint(context)

    assert profile.bits == 8
    assert profile.group_size == 128
    assert provider.should_quantize(
        "model.layers.0.self_attn.q_proj"
    )
    assert not provider.should_quantize(
        "model.layers.0.input_layernorm"
    )


def test_marlin_rejects_gptq_v2_without_conversion(tmp_path):
    context = prepare(tmp_path, "gptq_v2")

    with pytest.raises(
        QuantizationProviderError,
        match="GPTQ v1",
    ):
        GPTQMarlinProvider(context)


def test_marlin_rejects_asymmetric_checkpoint(tmp_path):
    context = prepare(tmp_path)
    path = tmp_path / "config.json"

    config = json.loads(path.read_text())
    config["quantization_config"]["sym"] = False
    path.write_text(json.dumps(config))

    with pytest.raises(
        QuantizationProviderError,
        match="sym=true",
    ):
        GPTQMarlinProvider(context)


def test_marlin_rejects_cpu_execution(tmp_path):
    context = prepare(tmp_path)
    provider = GPTQMarlinProvider(context)

    with pytest.raises(
        QuantizationProviderError,
        match="CUDA",
    ):
        provider.validate_runtime(
            context,
            provider.inspect_checkpoint(context),
        )
