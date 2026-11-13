import json
from pathlib import Path

import pytest

from llm_runtime.v3_2_scheduler_api import (
    create_scheduler,
    resolve_scheduler_config,
)


def make_config(active=4):
    return {
        "pipeline": {
            "memory": {
                "max_active_requests": active
            }
        }
    }


def test_existing_config_defaults_to_interleaved():
    settings = resolve_scheduler_config(make_config())
    assert settings.mode == "interleaved"
    assert settings.policy == "fcfs"
    assert settings.max_active_requests == 4
    assert settings.max_requests_per_batch == 1


def test_continuous_config_is_public_and_resolved():
    root = Path(__file__).resolve().parents[1]
    path = (
        root / "configs/v3_2/"
        "qwen3_0_6b_continuous_n2.json"
    )
    config = json.loads(path.read_text())

    settings = resolve_scheduler_config(config)

    assert settings.mode == "continuous"
    assert settings.policy == "fcfs"
    assert settings.max_active_requests == 4
    assert settings.max_requests_per_batch == 4
    assert settings.max_batch_tokens == 1024
    assert settings.max_queue_requests == 32


@pytest.mark.parametrize("policy", ["fcfs", "round_robin"])
def test_continuous_policy_options(policy):
    config = make_config()
    config["scheduler"] = {
        "mode": "continuous",
        "policy": policy,
    }

    assert resolve_scheduler_config(config).policy == policy


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_active_capacity(value):
    config = make_config(value)

    with pytest.raises(ValueError):
        resolve_scheduler_config(config)


def test_batch_cannot_exceed_resident_capacity():
    config = make_config(2)
    config["scheduler"] = {
        "mode": "continuous",
        "max_requests_per_batch": 4,
    }

    with pytest.raises(ValueError):
        resolve_scheduler_config(config)


def test_invalid_batch_token_budget():
    config = make_config(4)
    config["scheduler"] = {
        "mode": "continuous",
        "max_requests_per_batch": 4,
        "max_batch_tokens": 2,
    }

    with pytest.raises(ValueError):
        resolve_scheduler_config(config)


def test_unknown_setting_rejected():
    config = make_config()
    config["scheduler"] = {
        "mode": "continuous",
        "unknown_option": 123,
    }

    with pytest.raises(ValueError):
        resolve_scheduler_config(config)


def test_legacy_mode_rejects_batching_options():
    config = make_config()
    config["scheduler"] = {
        "mode": "interleaved",
        "max_requests_per_batch": 4,
    }

    with pytest.raises(ValueError):
        resolve_scheduler_config(config)


def test_continuous_execution_not_silently_downgraded():
    config = make_config()
    config["scheduler"] = {"mode": "continuous"}

    with pytest.raises(NotImplementedError):
        create_scheduler(config, [lambda t, c: None], lambda x: x)


def test_public_factory_preserves_existing_scheduler():
    from llm_runtime.v3_pipeline_scheduler import (
        StageResult,
    )

    config = make_config(1)

    def stage(task, cache):
        return StageResult(task.payload, cache)

    with create_scheduler(
        config,
        [stage],
        lambda payload: payload,
    ) as scheduler:
        result = scheduler.submit(
            7,
            prompt_length=1,
            max_new_tokens=1,
            timeout_s=2,
        ).result(timeout=3)

        assert result.tokens == (7,)
        assert scheduler.metrics()["completed_requests"] == 1
