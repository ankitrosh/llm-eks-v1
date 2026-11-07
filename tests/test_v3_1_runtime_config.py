import copy

import pytest

from llm_runtime.v3_1_runtime_config import (
    resolve_v31_config,
)


def make_config(num_ranks=3):
    return {
        "model": {
            "id": "test-qwen3",
            "num_layers": 6,
            "layer_weights_gib": [0.25] * 6,
        },
        "pipeline": {
            "num_ranks": num_ranks,
            "partition": "auto",
            "memory": {
                "gpu_memory_gib": [2.79] * num_ranks,
                "reserve_gib": 0.75,
                "max_total_tokens_per_request": 512,
                "max_active_requests": 2,
                "kv_bytes_per_layer_per_token": 4096,
                "first_stage_gib": 0.15,
                "last_stage_gib": 0.15,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "identity",
        },
    }


@pytest.mark.parametrize("num_ranks", [2, 3, 4])
def test_automatic_partition(num_ranks):
    plan = resolve_v31_config(
        make_config(num_ranks)
    )

    assert plan.num_ranks == num_ranks
    assert plan.layout.world_size == num_ranks
    assert len(plan.placements) == num_ranks

    boundaries = plan.as_dict()["layer_boundaries"]

    assert boundaries[0] == 0
    assert boundaries[-1] == 6
    assert len(boundaries) == num_ranks + 1

    assert all(
        right > left
        for left, right in zip(
            boundaries, boundaries[1:]
        )
    )

    assert sum(
        p.layer_count for p in plan.placements
    ) == 6

    assert all(
        p.estimated_gib <= p.budget_gib + 1e-9
        for p in plan.placements
    )


def test_manual_partition():
    config = make_config(4)
    config["pipeline"]["partition"] = "manual"
    config["pipeline"]["layer_counts"] = [1, 1, 2, 2]

    plan = resolve_v31_config(config)

    assert tuple(
        plan.layout.boundaries
    ) == (0, 1, 2, 4, 6)


def test_heterogeneous_memory():
    config = make_config(3)

    config["pipeline"]["memory"][
        "gpu_memory_gib"
    ] = [1.2, 2.0, 1.2]

    config["pipeline"]["memory"][
        "reserve_gib"
    ] = 0.25

    plan = resolve_v31_config(config)

    assert len(plan.placements) == 3

    assert all(
        p.estimated_gib <= p.budget_gib + 1e-9
        for p in plan.placements
    )


def test_infeasible_memory_rejected():
    config = make_config(3)

    config["pipeline"]["memory"][
        "gpu_memory_gib"
    ] = [0.5, 0.5, 0.5]

    config["pipeline"]["memory"][
        "reserve_gib"
    ] = 0.5

    with pytest.raises(ValueError):
        resolve_v31_config(config)


def test_invalid_rank_count_rejected():
    config = make_config(3)
    config["pipeline"]["num_ranks"] = 7

    with pytest.raises(ValueError):
        resolve_v31_config(config)


def test_manual_count_mismatch_rejected():
    config = make_config(4)
    config["pipeline"]["partition"] = "manual"
    config["pipeline"]["layer_counts"] = [1, 1, 1, 1]

    with pytest.raises(ValueError):
        resolve_v31_config(config)


def test_auto_rejects_manual_counts():
    config = make_config(4)
    config["pipeline"]["layer_counts"] = [1, 1, 2, 2]

    with pytest.raises(ValueError):
        resolve_v31_config(config)


def test_unknown_backend_rejected():
    config = make_config()
    config["communication"]["backend"] = "unknown"

    with pytest.raises(ValueError):
        resolve_v31_config(config)


def test_scalar_memory_expansion():
    config = make_config(4)

    config["pipeline"]["memory"][
        "gpu_memory_gib"
    ] = 2.79

    plan = resolve_v31_config(config)

    assert all(
        p.gpu_memory_gib == 2.79
        for p in plan.placements
    )


def test_config_not_mutated():
    config = make_config(3)
    original = copy.deepcopy(config)

    resolve_v31_config(config)

    assert config == original
