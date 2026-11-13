"""V3.1 public GPTQ configuration contract.

The user specifies GPTQ INT8 without selecting Marlin.
These CPU tests exercise real planning and provider selection,
mocking only the GPU-dependent runtime validation.
"""

import copy
import json

import pytest
import yaml

from test_v3_1_gptq_int8_inspection import (
    make_checkpoint,
    make_config,
)

from llm_runtime.v3_1_launch_spec import (
    compile_v31_launch_spec,
)
from llm_runtime.providers.v3_1_configured_stage_factory import (
    create_v31_qwen3_stage_factory,
)
from llm_runtime.providers.v3_1_gptq_marlin_provider import (
    GPTQMarlinProvider,
)


@pytest.fixture
def gptq_config(tmp_path):
    make_checkpoint(tmp_path)

    # Add the GPTQ v1 properties required by Marlin.
    config_file = tmp_path / "config.json"
    metadata = json.loads(config_file.read_text())

    metadata["quantization_config"].update({
        "sym": True,
        "desc_act": False,
        "pack_dtype": "int32",
    })

    config_file.write_text(json.dumps(metadata))

    # YAML round trip represents the user-facing config.
    return yaml.safe_load(
        yaml.safe_dump(make_config(tmp_path, ranks=2))
    )


def test_auto_selects_internal_marlin(
    gptq_config,
    monkeypatch,
):
    selector = gptq_config["model"]["quantization"]

    assert selector["provider"] == "gptq"
    assert selector["options"] == {"bits": 8}
    assert "execution_backend" not in selector["options"]

    original = copy.deepcopy(gptq_config)

    spec = compile_v31_launch_spec(gptq_config)

    calls = []

    def validate_without_gpu(self, context, profile):
        # Exercise actual provider construction/profile matching,
        # but do not require CUDA on the CPU test host.
        assert context.device == "cuda:0"
        assert context.activation_dtype == "bfloat16"
        assert profile == self.profile
        calls.append("validated")

    monkeypatch.setattr(
        GPTQMarlinProvider,
        "validate_runtime",
        validate_without_gpu,
    )

    factory = create_v31_qwen3_stage_factory(
        gptq_config,
        launch_spec=spec,
        device="cuda:0",
        dtype="bfloat16",
    )

    assert isinstance(
        factory.quantization_provider,
        GPTQMarlinProvider,
    )

    assert tuple(factory.layer_boundaries) == tuple(
        spec.layout.boundaries
    )

    assert calls == ["validated", "validated"]
    assert gptq_config == original


def test_auto_rejects_cpu_execution(gptq_config):
    spec = compile_v31_launch_spec(gptq_config)

    # CPU execution must not silently switch to an
    # unrelated reference implementation.
    with pytest.raises(ValueError, match="CUDA"):
        create_v31_qwen3_stage_factory(
            gptq_config,
            launch_spec=spec,
            device="cpu",
            dtype="float32",
        )


def test_launch_spec_configuration_drift_rejected(
    gptq_config,
):
    spec = compile_v31_launch_spec(gptq_config)

    changed = copy.deepcopy(gptq_config)
    changed["model"]["quantization"]["options"][
        "bits"
    ] = 4

    with pytest.raises(
        ValueError,
        match="differs from launch spec",
    ):
        create_v31_qwen3_stage_factory(
            changed,
            launch_spec=spec,
            device="cuda:0",
            dtype="bfloat16",
        )


def test_incompatible_marlin_checkpoint_rejected(
    gptq_config,
):
    spec = compile_v31_launch_spec(gptq_config)

    config_file = (
        gptq_config["model"]["checkpoint_dir"]
        + "/config.json"
    )

    with open(config_file) as handle:
        metadata = json.load(handle)

    metadata["quantization_config"]["sym"] = False

    with open(config_file, "w") as handle:
        json.dump(metadata, handle)

    with pytest.raises(ValueError, match="sym=true"):
        create_v31_qwen3_stage_factory(
            gptq_config,
            launch_spec=spec,
            device="cuda:0",
            dtype="bfloat16",
        )
