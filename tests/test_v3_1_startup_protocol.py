import copy

import pytest

from llm_runtime.v3_1_launch_spec import (
    compile_v31_launch_spec,
)
from llm_runtime.v3_1_startup_protocol import (
    StartupProtocolError,
    make_local_record,
    validate_startup_records,
)


def make_spec(ranks=3):
    config = {
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
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "torch-zlib-v1",
        },
    }

    return compile_v31_launch_spec(config)


def reports(spec):
    return [
        make_local_record(
            spec,
            rank,
            process_backend="gloo",
            device="cpu",
        )
        for rank in range(spec.world_size)
    ]


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_valid_startup(ranks):
    spec = make_spec(ranks)
    assert validate_startup_records(spec, reports(spec))


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_version", 999),
        ("launch_format_version", 999),
        ("plan_sha256", "incorrect"),
        ("world_size", 4),
        ("codec", "identity"),
        ("transport", "nccl"),
        ("wire_kind", "tensor"),
        ("process_backend", "nccl"),
    ],
)
def test_mismatched_rank_rejected(field, value):
    spec = make_spec()
    records = reports(spec)

    records[1][field] = value

    with pytest.raises(StartupProtocolError):
        validate_startup_records(spec, records)


def test_duplicate_rank_rejected():
    spec = make_spec()
    records = reports(spec)
    records[2]["rank"] = 1

    with pytest.raises(StartupProtocolError):
        validate_startup_records(spec, records)


def test_missing_wire_capability_rejected():
    spec = make_spec()
    records = reports(spec)

    records[1]["supported_wire_kinds"] = ["tensor"]

    with pytest.raises(
        StartupProtocolError,
        match="wire capabilities",
    ):
        validate_startup_records(spec, records)


def test_local_failure_is_propagated():
    spec = make_spec()
    records = reports(spec)

    records[2] = {
        "rank": 2,
        "error": "Custom codec failed to load",
    }

    with pytest.raises(
        StartupProtocolError,
        match="failed to load",
    ):
        validate_startup_records(spec, records)


def test_validation_does_not_mutate_reports():
    spec = make_spec()
    records = reports(spec)
    original = copy.deepcopy(records)

    validate_startup_records(spec, records)

    assert records == original


def test_custom_codec_wire_kind_disagreement():
    config = {
        "model": {
            "id": "tiny-qwen3",
            "num_layers": 6,
            "layer_weights_gib": [0.25] * 6,
        },
        "pipeline": {
            "num_ranks": 3,
            "partition": "auto",
            "memory": {
                "gpu_memory_gib": [2.79] * 3,
                "reserve_gib": 0.75,
            },
        },
        "communication": {
            "backend": "custom",
            "transport_factory": "example.transport:create",
            "codec": "custom",
            "codec_factory": "example.codec:create",
        },
    }

    spec = compile_v31_launch_spec(config)

    records = [
        make_local_record(
            spec,
            rank,
            process_backend="gloo",
            device="cpu",
            codec_wire_kind=(
                "tensor" if rank == 1 else "bytes"
            ),
            custom_transport_wire_kinds=[
                "tensor", "bytes"
            ],
            plugins_loaded=True,
        )
        for rank in range(3)
    ]

    with pytest.raises(
        StartupProtocolError,
        match="wire payload type",
    ):
        validate_startup_records(spec, records)
