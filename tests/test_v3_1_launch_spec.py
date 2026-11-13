import copy

import pytest

from llm_runtime.v3_1_launch_spec import (
    compile_v31_launch_spec,
)


def make_config(ranks=3):
    return {
        "model": {
            "id": "tiny-qwen3",
            "num_layers": 6,
            "layer_weights_gib": [0.25] * 6,
        },
        "pipeline": {
            "num_ranks": ranks,
            "partition": "auto",
            "memory": {
                "gpu_memory_gib": [2.79] * ranks,
                "reserve_gib": 0.75,
                "max_active_requests": 2,
                "max_total_tokens_per_request": 512,
                "kv_bytes_per_layer_per_token": 4096,
                "first_stage_gib": 0.15,
                "last_stage_gib": 0.15,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "torch-zlib-v1",
            "options": {
                "zlib_level": 3,
            },
        },
    }


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_unified_auto_launch_spec(ranks):
    spec = compile_v31_launch_spec(
        make_config(ranks)
    )

    result = spec.as_dict()

    assert result["format_version"] == 1
    assert result["world_size"] == ranks
    assert result["communication"]["codec"] == (
        "torch-zlib-v1"
    )
    assert len(result["ranks"]) == ranks

    boundaries = result["layer_boundaries"]

    assert boundaries[0] == 0
    assert boundaries[-1] == 6
    assert len(boundaries) == ranks + 1

    for rank in range(ranks):
        stage = spec.for_rank(rank)

        assert stage["rank"] == rank
        assert stage["start_layer"] == boundaries[rank]
        assert stage["end_layer"] == boundaries[rank + 1]
        assert stage["plan_sha256"] == spec.plan_sha256

        assert stage["estimated_gib"] <= (
            stage["budget_gib"] + 1e-9
        )


def test_manual_layout_uses_requested_counts():
    config = make_config(4)

    config["pipeline"]["partition"] = "manual"
    config["pipeline"]["layer_counts"] = [
        1, 1, 2, 2
    ]

    spec = compile_v31_launch_spec(config)

    assert tuple(spec.layout.boundaries) == (
        0, 1, 2, 4, 6
    )


def test_deterministic_plan_digest():
    config = make_config(3)

    first = compile_v31_launch_spec(config)
    second = compile_v31_launch_spec(
        copy.deepcopy(config)
    )

    assert first.plan_sha256 == second.plan_sha256
    assert len(first.plan_sha256) == 64


def test_changed_placement_changes_digest():
    config = make_config(3)

    first = compile_v31_launch_spec(config)

    config["model"]["layer_weights_gib"][0] = 0.3

    second = compile_v31_launch_spec(config)

    assert first.plan_sha256 != second.plan_sha256


def test_rejects_invalid_rank():
    spec = compile_v31_launch_spec(
        make_config(3)
    )

    with pytest.raises(ValueError):
        spec.for_rank(3)


def test_original_config_not_mutated():
    config = make_config(3)
    original = copy.deepcopy(config)

    compile_v31_launch_spec(config)

    assert config == original
