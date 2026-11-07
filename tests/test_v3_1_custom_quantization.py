import copy
import sys
import types

import pytest

from llm_runtime.v3_1_launch_spec import (
    compile_v31_launch_spec,
)
from llm_runtime.v3_1_quantization_api import (
    QuantizationProfile,
    QuantizationProviderError,
)

MIB = 1024 ** 2


class FakeINT6Provider:
    def inspect_checkpoint(self, context):
        return QuantizationProfile(
            provider="example-int6",
            format_id=context.options.get(
                "format_id", "packed-int6-v1"
            ),
            bits=6,
            group_size=128,
            layer_storage_bytes=(40 * MIB,) * 4,
            first_stage_bytes=20 * MIB,
            last_stage_bytes=30 * MIB,
            kv_bytes_per_layer_per_token=1024,
            workspace_bytes_per_rank=64 * MIB,
            execution_mode="linear",
        )

    def validate_runtime(self, context, profile):
        # This cannot execute on CPU, but offline
        # checkpoint planning must still succeed.
        if context.device != "cuda":
            raise RuntimeError("CUDA required for execution")

    def build_linear(self, **kwargs):
        raise NotImplementedError(
            "Test provider does not execute weights"
        )


@pytest.fixture
def provider_factory(monkeypatch):
    module = types.ModuleType("v31_test_int6_provider")
    module.create_provider = lambda ctx: FakeINT6Provider()

    monkeypatch.setitem(
        sys.modules, module.__name__, module
    )

    return "v31_test_int6_provider:create_provider"


def make_config(tmp_path, factory, ranks=3):
    return {
        "model": {
            "checkpoint_dir": str(tmp_path / "fake-checkpoint"),
            "quantization": {
                "provider": "custom",
                "factory": factory,
                "options": {
                    "bits": 6,
                    "group_size": 128,
                    "implementation_version": "1.0",
                },
            },
        },
        "pipeline": {
            "num_ranks": ranks,
            "partition": "auto",
            "memory": {
                "gpu_memory_gib": [1.0] * ranks,
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


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_custom_provider_auto_launch(
    tmp_path, provider_factory, ranks
):
    config = make_config(
        tmp_path, provider_factory, ranks
    )

    original = copy.deepcopy(config)
    spec = compile_v31_launch_spec(config)

    assert spec.world_size == ranks
    assert spec.layout.boundaries[0] == 0
    assert spec.layout.boundaries[-1] == 4
    assert len(spec.layout.boundaries) == ranks + 1

    assert spec.quantization["provider"] == "custom"
    assert spec.quantization["implementation"] == "example-int6"
    assert spec.quantization["bits"] == 6
    assert spec.quantization["factory"] == provider_factory

    assert spec.checkpoint_profile.layer_storage_bytes == (
        40 * MIB,
    ) * 4

    assert spec.as_dict()["checkpoint_profile"][
        "workspace_bytes_per_rank"
    ] == 64 * MIB

    assert config == original


def test_custom_options_change_digest(tmp_path, provider_factory):
    config = make_config(tmp_path, provider_factory)

    first = compile_v31_launch_spec(config)

    modified = copy.deepcopy(config)

    modified["model"]["quantization"]["options"][
        "implementation_version"
    ] = "2.0"

    second = compile_v31_launch_spec(modified)

    assert first.layout.boundaries == second.layout.boundaries
    assert first.plan_sha256 != second.plan_sha256


def test_custom_format_changes_digest(tmp_path, provider_factory):
    config = make_config(tmp_path, provider_factory)

    first = compile_v31_launch_spec(config)

    modified = copy.deepcopy(config)
    modified["model"]["quantization"]["options"][
        "format_id"
    ] = "packed-int6-v2"

    second = compile_v31_launch_spec(modified)

    assert first.plan_sha256 != second.plan_sha256


def test_custom_bit_mismatch_rejected(
    tmp_path, provider_factory
):
    config = make_config(tmp_path, provider_factory)

    config["model"]["quantization"]["options"][
        "bits"
    ] = 4

    with pytest.raises(
        QuantizationProviderError,
        match="bits",
    ):
        compile_v31_launch_spec(config)


def test_missing_custom_factory_rejected(
    tmp_path, provider_factory
):
    config = make_config(tmp_path, provider_factory)

    del config["model"]["quantization"]["factory"]

    with pytest.raises(
        ValueError,
        match="factory",
    ):
        compile_v31_launch_spec(config)


def test_custom_provider_is_not_executed_during_planning(
    tmp_path, provider_factory
):
    config = make_config(tmp_path, provider_factory)

    # validate_runtime() rejects CPU, and build_linear()
    # raises NotImplementedError. Neither should be called
    # when generating a placement plan.
    spec = compile_v31_launch_spec(config)

    assert spec.world_size == 3
