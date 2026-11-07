import copy
import json
import struct

import pytest

from llm_runtime.v3_1_gptq_int8_inspection import (
    inspect_qwen3_gptq_int8,
    resolve_gptq_int8_plan,
)


def make_checkpoint(root, *, bits=8):
    config = {
        "model_type": "qwen3",
        "num_hidden_layers": 4,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "quantization_config": {
            "quant_method": "gptq",
            "bits": bits,
            "group_size": 128,
            "format": "gptq",
        },
    }

    (root / "config.json").write_text(
        json.dumps(config)
    )

    tensors = {
        "model.embed_tokens.weight": ("BF16", [32, 16]),
        "model.norm.weight": ("BF16", [16]),
        "lm_head.weight": ("BF16", [32, 16]),
    }

    for layer in range(4):
        prefix = f"model.layers.{layer}.self_attn.q_proj"

        tensors[prefix + ".qweight"] = ("I32", [4, 16])
        tensors[prefix + ".qzeros"] = ("I32", [1, 4])
        tensors[prefix + ".scales"] = ("F16", [1, 16])
        tensors[prefix + ".g_idx"] = ("I32", [16])

    bytes_per = {"BF16": 2, "F16": 2, "I32": 4}

    header = {}
    offset = 0

    for name, (dtype, shape) in tensors.items():
        count = 1

        for dim in shape:
            count *= dim

        nbytes = count * bytes_per[dtype]

        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }

        offset += nbytes

    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)

    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + bytes(offset)
    )


def make_config(root, ranks=2):
    return {
        "model": {
            "checkpoint_dir": str(root),
            "quantization": {
                "provider": "gptq",
                "options": {"bits": 8},
            },
        },
        "pipeline": {
            "num_ranks": ranks,
            "partition": "auto",
            "memory": {
                "gpu_memory_gib": [1.0] * ranks,
                "reserve_gib": 0.25,
                "max_active_requests": 2,
                "max_total_tokens_per_request": 128,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "identity",
        },
    }


def test_gptq_storage_and_kv_estimates(tmp_path):
    make_checkpoint(tmp_path)

    profile = inspect_qwen3_gptq_int8(tmp_path)

    assert profile.provider == "gptq"
    assert profile.bits == 8
    assert profile.group_size == 128
    assert profile.num_layers == 4
    assert profile.layer_storage_bytes == (368,) * 4
    assert profile.kv_bytes_per_layer_per_token == 32
    assert profile.first_stage_bytes == 1024
    assert profile.last_stage_bytes == 1056


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_gptq_auto_partition(tmp_path, ranks):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path, ranks)
    plan, profile = resolve_gptq_int8_plan(config)

    assert plan.num_ranks == ranks
    assert profile.num_layers == 4

    boundaries = plan.layout.boundaries

    assert boundaries[0] == 0
    assert boundaries[-1] == 4
    assert len(boundaries) == ranks + 1

    assert all(
        stage.estimated_gib <= stage.budget_gib + 1e-9
        for stage in plan.placements
    )


def test_manual_quantized_partition(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path, 3)
    config["pipeline"]["partition"] = "manual"
    config["pipeline"]["layer_counts"] = [1, 1, 2]

    plan, _ = resolve_gptq_int8_plan(config)

    assert tuple(plan.layout.boundaries) == (0, 1, 2, 4)


def test_wrong_bit_width_rejected(tmp_path):
    make_checkpoint(tmp_path, bits=4)

    with pytest.raises(ValueError, match="INT8"):
        inspect_qwen3_gptq_int8(tmp_path)


def test_configuration_bit_mismatch_rejected(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path)
    config["model"]["quantization"]["options"]["bits"] = 4

    with pytest.raises(ValueError):
        resolve_gptq_int8_plan(config)


def test_original_config_unchanged(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path)
    original = copy.deepcopy(config)

    resolve_gptq_int8_plan(config)

    assert config == original


def test_infeasible_memory_rejected(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path)
    config["pipeline"]["memory"]["reserve_gib"] = 1.0

    with pytest.raises(ValueError):
        resolve_gptq_int8_plan(config)
