"""End-to-end GPTQ-format Qwen3 checkpoint execution tests.

Creates synthetic packed INT8 checkpoint tensors, then:
- compiles automatic N-rank placement
- selectively loads quantized layers
- executes prefill and cached decoding
- compares against independently dequantized full Qwen3

The quantization used to create the checkpoint is
round-to-nearest, not calibration-based GPTQ.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from safetensors.torch import save_file

from llm_runtime.v3_1_launch_spec import (
    compile_v31_launch_spec,
)
from llm_runtime.v3_1_quantization_api import (
    QuantizationContext,
)
from llm_runtime.providers.v3_1_gptq_int8_torch import (
    GPTQInt8TorchReferenceProvider,
)
from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)


GROUP_SIZE = 32
NUM_LAYERS = 6
LINEARS_PER_LAYER = 7


def pack_input_axis(codes):
    """Pack four unsigned INT8 codes into each INT32."""

    k, n = codes.shape
    assert k % 4 == 0

    words = torch.zeros(
        (k // 4, n), dtype=torch.int64
    )

    for lane in range(4):
        words |= (
            codes[lane::4].to(torch.int64)
            << (lane * 8)
        )

    return words.to(torch.int32).contiguous()


def pack_output_axis(codes):
    """Pack four zero points across output columns."""

    groups, n = codes.shape
    assert n % 4 == 0

    words = torch.zeros(
        (groups, n // 4), dtype=torch.int64
    )

    for lane in range(4):
        words |= (
            codes[:, lane::4].to(torch.int64)
            << (lane * 8)
        )

    return words.to(torch.int32).contiguous()


def quantize_and_pack(weight, checkpoint_format, group_size=GROUP_SIZE):
    """Return GPTQ-format tensors and independent dequantization."""

    # PyTorch Linear weights have shape [out_features, in_features].
    values = weight.detach().float().T.contiguous()

    k, n = values.shape
    assert k % 4 == 0 and n % 4 == 0

    groups = math.ceil(k / group_size)

    g_idx = (
        torch.arange(k, dtype=torch.int32)
        // group_size
    )

    codes = torch.empty((k, n), dtype=torch.int32)
    scales = torch.empty(
        (groups, n), dtype=torch.float32
    )

    for group in range(groups):
        start = group * group_size
        end = min(start + group_size, k)

        block = values[start:end]

        scale = (
            block.abs()
            .amax(dim=0)
            .clamp_min(1e-5)
            / 127.0
        )

        scales[group] = scale

        # Synthetic symmetric INT8 quantization:
        # unsigned midpoint 128 represents zero.
        codes[start:end] = (
            (block / scale)
            .round()
            .add(128)
            .clamp(0, 255)
            .to(torch.int32)
        )

    zeros = torch.full(
        (groups, n), 128, dtype=torch.int32
    )

    # Legacy GPTQ saves the zero point minus one.
    stored_zeros = (
        (zeros - 1) % 256
        if checkpoint_format == "gptq"
        else zeros
    )

    tensors = {
        "qweight": pack_input_axis(codes),
        "qzeros": pack_output_axis(stored_zeros),
        "scales": scales.contiguous(),
        "g_idx": g_idx.contiguous(),
    }

    # Independently reconstruct the reference weight.
    # We do not call GPTQInt8TorchLinear here.
    reconstructed = (
        (codes.float() - zeros[g_idx].float())
        * scales[g_idx]
    ).T.contiguous()

    return tensors, reconstructed


def create_packed_qwen3_checkpoint(root, checkpoint_format, group_size=GROUP_SIZE):
    torch.manual_seed(42)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )

    config._attn_implementation = "eager"

    reference = Qwen3ForCausalLM(config).eval()

    checkpoint_tensors = {
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in reference.state_dict().items()
    }

    quantized_names = []

    for layer_index, layer in enumerate(
        reference.model.layers
    ):
        for local_name, module in layer.named_modules():
            if not local_name or not isinstance(
                module, nn.Linear
            ):
                continue

            prefix = (
                f"model.layers.{layer_index}.{local_name}"
            )

            original = checkpoint_tensors.pop(
                prefix + ".weight"
            )

            packed, reconstructed = quantize_and_pack(
                original, checkpoint_format, group_size
            )

            for suffix, tensor in packed.items():
                checkpoint_tensors[
                    prefix + "." + suffix
                ] = tensor

            # Full reference model executes the same
            # dequantized weights, independently stored.
            with torch.no_grad():
                module.weight.copy_(reconstructed)

            quantized_names.append(prefix)

    assert len(quantized_names) == (
        NUM_LAYERS * LINEARS_PER_LAYER
    )

    # Ensure no original dense decoder linear weights remain
    # in the checkpoint. This forces the quantized load path.
    for name in quantized_names:
        assert name + ".weight" not in checkpoint_tensors

    config.quantization_config = {
        "quant_method": "gptq",
        "bits": 8,
        "group_size": group_size,
        "checkpoint_format": checkpoint_format,
    }
    config.dtype = "float32"

    config.save_pretrained(str(root))

    save_file(
        checkpoint_tensors,
        str(root / "model.safetensors"),
    )

    saved_config = json.loads(
        (root / "config.json").read_text()
    )

    assert saved_config["quantization_config"][
        "checkpoint_format"
    ] == checkpoint_format

    return reference, quantized_names


class RecordingGPTQProvider(
    GPTQInt8TorchReferenceProvider
):
    def __init__(self, context):
        super().__init__(context)
        self.loaded_names = []

    def build_linear(self, **kwargs):
        self.loaded_names.append(
            kwargs["module_name"]
        )
        return super().build_linear(**kwargs)


def make_launch_config(root):
    return {
        "model": {
            "checkpoint_dir": str(root),
            "quantization": {
                "provider": "gptq",
                "options": {
                    "bits": 8,
                    "group_size": GROUP_SIZE,
                },
            },
        },
        "pipeline": {
            "num_ranks": 3,
            "partition": "auto",
            "memory": {
                "gpu_memory_gib": [1.0] * 3,
                "reserve_gib": 0.25,
                "max_active_requests": 2,
                "max_total_tokens_per_request": 64,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "identity",
        },
    }


@pytest.mark.parametrize(
    "checkpoint_format",
    ["gptq", "gptq_v2"],
)
def test_packed_qwen3_three_stage_execution(
    tmp_path, checkpoint_format
):
    reference, quantized_names = (
        create_packed_qwen3_checkpoint(
            tmp_path, checkpoint_format
        )
    )

    # The real quantization-aware launch compiler.
    spec = compile_v31_launch_spec(
        make_launch_config(tmp_path)
    )

    assert spec.world_size == 3
    assert spec.quantization["bits"] == 8
    assert spec.quantization["format"] == checkpoint_format
    assert spec.layout.boundaries[0] == 0
    assert spec.layout.boundaries[-1] == NUM_LAYERS

    context = QuantizationContext(
        checkpoint_dir=str(tmp_path),
        device="cpu",
        activation_dtype="float32",
    )

    provider = RecordingGPTQProvider(context)

    profile = provider.inspect_checkpoint(context)
    provider.validate_runtime(context, profile)

    factory = Qwen3NRankStageOnlyFactory(
        checkpoint_dir=str(tmp_path),
        layer_boundaries=tuple(
            spec.layout.boundaries
        ),
        device="cpu",
        dtype="float32",
        quantization_provider=provider,
    )

    executors = [
        factory(rank)
        for rank in range(3)
    ]

    assert len(provider.loaded_names) == len(
        quantized_names
    )
    assert set(provider.loaded_names) == set(
        quantized_names
    )

    for rank, executor in enumerate(executors):
        start, end = spec.layout.rank_bounds(rank)

        assert executor.model.owned_layers == tuple(
            range(start, end)
        )

        assert (
            executor.model.quantized_linear_count
            == (end - start) * LINEARS_PER_LAYER
        )

        assert all(
            not parameter.is_meta
            for parameter in executor.model.parameters()
        )

    # Prefill three tokens, then decode one additional token.
    initial_tokens = torch.tensor(
        [[4, 7, 9]], dtype=torch.long
    )
    next_token = 11

    caches = [None, None, None]

    for step, position, initial_payload in (
        (0, 0, initial_tokens),
        (1, 3, next_token),
    ):
        payload = initial_payload

        with torch.inference_mode():
            for rank, executor in enumerate(executors):
                task = SimpleNamespace(
                    step=step,
                    position=position,
                    payload=payload,
                )

                result = executor(task, caches[rank])

                payload = result.payload
                caches[rank] = result.cache_state

            reference_ids = (
                initial_tokens
                if step == 0
                else torch.cat(
                    [
                        initial_tokens,
                        torch.tensor(
                            [[next_token]],
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
                )
            )

            expected = reference(
                input_ids=reference_ids,
                use_cache=False,
            ).logits[:, -1:, :]

        assert payload.shape == expected.shape

        torch.testing.assert_close(
            payload,
            expected,
            rtol=2e-5,
            atol=2e-5,
        )


@pytest.mark.parametrize(
    "checkpoint_format",
    ["gptq", "gptq_v2"],
)
def test_config_selected_gptq_execution(
    tmp_path, checkpoint_format
):
    from llm_runtime.providers.v3_1_configured_stage_factory import (
        create_v31_qwen3_stage_factory,
    )

    reference, _ = create_packed_qwen3_checkpoint(
        tmp_path, checkpoint_format
    )

    config = make_launch_config(tmp_path)
    config["model"]["quantization"]["options"][
        "execution_backend"
    ] = "torch-reference"

    spec = compile_v31_launch_spec(config)

    factory = create_v31_qwen3_stage_factory(
        config,
        launch_spec=spec,
        device="cpu",
        dtype="float32",
    )

    executors = [
        factory(rank)
        for rank in range(spec.world_size)
    ]

    assert sum(
        executor.model.quantized_linear_count
        for executor in executors
    ) == NUM_LAYERS * LINEARS_PER_LAYER

    input_ids = torch.tensor(
        [[4, 7, 9]], dtype=torch.long
    )

    caches = [None] * spec.world_size

    # Test both prefill and a cached decode step.
    for step, position, first_payload in (
        (0, 0, input_ids),
        (1, 3, 11),
    ):
        payload = first_payload

        with torch.inference_mode():
            for rank, executor in enumerate(executors):
                result = executor(
                    SimpleNamespace(
                        step=step,
                        position=position,
                        payload=payload,
                    ),
                    caches[rank],
                )

                payload = result.payload
                caches[rank] = result.cache_state

            reference_ids = (
                input_ids
                if step == 0
                else torch.tensor(
                    [[4, 7, 9, 11]],
                    dtype=torch.long,
                )
            )

            expected = reference(
                input_ids=reference_ids,
                use_cache=False,
            ).logits[:, -1:, :]

        torch.testing.assert_close(
            payload,
            expected,
            rtol=2e-5,
            atol=2e-5,
        )


def test_gptq_requires_explicit_execution_backend(tmp_path):
    from llm_runtime.providers.v3_1_configured_stage_factory import (
        create_v31_qwen3_stage_factory,
    )

    create_packed_qwen3_checkpoint(tmp_path, "gptq")
    config = make_launch_config(tmp_path)

    with pytest.raises(
        ValueError,
        match="execution_backend",
    ):
        create_v31_qwen3_stage_factory(
            config,
            device="cpu",
            dtype="float32",
        )


def test_gptq_cpu_reference_rejects_bfloat16(tmp_path):
    from llm_runtime.providers.v3_1_configured_stage_factory import (
        create_v31_qwen3_stage_factory,
    )

    create_packed_qwen3_checkpoint(tmp_path, "gptq")
    config = make_launch_config(tmp_path)

    config["model"]["quantization"]["options"][
        "execution_backend"
    ] = "torch-reference"

    with pytest.raises(ValueError, match="float32"):
        create_v31_qwen3_stage_factory(
            config,
            device="cpu",
            dtype="bfloat16",
        )


def test_custom_quantization_factory_from_config(
    tmp_path, monkeypatch
):
    import sys
    import types

    from llm_runtime.providers.v3_1_configured_stage_factory import (
        create_v31_qwen3_stage_factory,
    )

    create_packed_qwen3_checkpoint(tmp_path, "gptq_v2")

    module = types.ModuleType(
        "v31_test_custom_execution_provider"
    )

    module.create_provider = (
        lambda context: GPTQInt8TorchReferenceProvider(
            context
        )
    )

    monkeypatch.setitem(
        sys.modules,
        module.__name__,
        module,
    )

    config = make_launch_config(tmp_path)

    config["model"]["quantization"] = {
        "provider": "custom",
        "factory": (
            "v31_test_custom_execution_provider:"
            "create_provider"
        ),
        "options": {
            "bits": 8,
            "group_size": GROUP_SIZE,
            "activation_dtype": "float32",
        },
    }

    spec = compile_v31_launch_spec(config)

    factory = create_v31_qwen3_stage_factory(
        config,
        launch_spec=spec,
        device="cpu",
        dtype="float32",
    )

    executor = factory(0)

    start, end = spec.layout.rank_bounds(0)

    assert executor.model.quantized_linear_count == (
        (end - start) * LINEARS_PER_LAYER
    )
