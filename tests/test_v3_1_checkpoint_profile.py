import json
import struct

import pytest

from llm_runtime.v3_1_checkpoint_profile import (
    GIB,
    discover_qwen3_checkpoint,
    resolve_v31_checkpoint_config,
)


def make_checkpoint(root, *, tied=False):
    config = {
        "model_type": "qwen3",
        "num_hidden_layers": 6,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "dtype": "bfloat16",
        "tie_word_embeddings": tied,
    }

    (root / "config.json").write_text(
        json.dumps(config)
    )

    shapes = {
        "model.embed_tokens.weight": [32, 16],
        "model.norm.weight": [16],
    }

    if not tied:
        shapes["lm_head.weight"] = [32, 16]

    for i in range(6):
        shapes[
            f"model.layers.{i}.self_attn.q_proj.weight"
        ] = [16, 16]

    header = {}
    offset = 0

    for name, shape in shapes.items():
        nbytes = 2

        for dim in shape:
            nbytes *= dim

        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [
                offset,
                offset + nbytes,
            ],
        }

        offset += nbytes

    raw_header = json.dumps(header).encode()
    raw_header += b" " * (-len(raw_header) % 8)

    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw_header))
        + raw_header
        + bytes(offset)
    )

    return root


def make_config(path, ranks):
    return {
        "model": {
            "checkpoint_dir": str(path),
        },
        "pipeline": {
            "num_ranks": ranks,
            "partition": "auto",
            "memory": {
                "reserve_gib": 0.75,
                "max_active_requests": 2,
                "max_total_tokens_per_request": 512,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "identity",
        },
    }


def test_checkpoint_metadata(tmp_path):
    make_checkpoint(tmp_path)

    profile = discover_qwen3_checkpoint(tmp_path)

    assert profile.num_layers == 6
    assert profile.tensor_count == 9
    assert profile.kv_bytes_per_layer_per_token == 32

    assert profile.layer_weights_gib == (
        512 / GIB,
    ) * 6

    assert profile.first_stage_gib == 1024 / GIB
    assert profile.last_stage_gib == 1056 / GIB


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_automatic_checkpoint_partition(tmp_path, ranks):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path, ranks)

    plan, profile = resolve_v31_checkpoint_config(
        config,
        gpu_memory_gib=2.79,
    )

    assert plan.num_ranks == ranks
    assert plan.layout.world_size == ranks

    boundaries = plan.layout.boundaries

    assert boundaries[0] == 0
    assert boundaries[-1] == 6
    assert len(boundaries) == ranks + 1
    assert profile.kv_bytes_per_layer_per_token == 32

    assert all(
        placement.estimated_gib <= placement.budget_gib
        for placement in plan.placements
    )


def test_manual_checkpoint_partition(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path, 4)
    config["pipeline"]["partition"] = "manual"
    config["pipeline"]["layer_counts"] = [1, 1, 2, 2]

    plan, _ = resolve_v31_checkpoint_config(
        config,
        gpu_memory_gib=2.79,
    )

    assert tuple(plan.layout.boundaries) == (
        0, 1, 2, 4, 6
    )


def test_tied_output_head_is_budgeted(tmp_path):
    make_checkpoint(tmp_path, tied=True)

    profile = discover_qwen3_checkpoint(tmp_path)

    assert profile.last_stage_gib == 1056 / GIB


def test_quantized_checkpoint_requires_estimator(tmp_path):
    make_checkpoint(tmp_path)

    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())

    config["quantization_config"] = {
        "quant_method": "gptq"
    }

    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Quantized"):
        discover_qwen3_checkpoint(tmp_path)


def test_checkpoint_layer_mismatch_rejected(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path, 4)
    config["model"]["num_layers"] = 5

    with pytest.raises(ValueError):
        resolve_v31_checkpoint_config(
            config,
            gpu_memory_gib=2.79,
        )


def test_missing_gpu_budget_rejected(tmp_path):
    make_checkpoint(tmp_path)

    with pytest.raises(ValueError):
        resolve_v31_checkpoint_config(
            make_config(tmp_path, 4)
        )


def test_manual_memory_estimates_rejected(tmp_path):
    make_checkpoint(tmp_path)

    config = make_config(tmp_path, 4)
    config["pipeline"]["memory"][
        "kv_bytes_per_layer_per_token"
    ] = 0

    with pytest.raises(ValueError):
        resolve_v31_checkpoint_config(
            config,
            gpu_memory_gib=2.79,
        )
