import pytest

from llm_runtime.v3_1_quantization_api import (
    GIB,
    QuantizationContext,
    QuantizationProfile,
    QuantizationProviderError,
    QuantizationRegistry,
)


def profile(mode="linear"):
    return QuantizationProfile(
        provider="fake-int8",
        format_id="fake-packed-v1",
        bits=8,
        group_size=128,
        layer_storage_bytes=(
            300 * 1024**2,
            400 * 1024**2,
            500 * 1024**2,
        ),
        first_stage_bytes=100 * 1024**2,
        last_stage_bytes=150 * 1024**2,
        kv_bytes_per_layer_per_token=4096,
        workspace_bytes_per_rank=64 * 1024**2,
        execution_mode=mode,
    )


class FakeLinearProvider:
    def inspect_checkpoint(self, context):
        return profile()

    def validate_runtime(self, context, profile):
        if context.device != "cpu":
            raise QuantizationProviderError(
                "Fake provider only supports CPU"
            )

    def build_linear(self, **kwargs):
        return object()


class FakeStageProvider(FakeLinearProvider):
    def inspect_checkpoint(self, context):
        return profile("stage")

    def load_stage(self, **kwargs):
        return lambda task, cache: None


def test_custom_linear_provider():
    registry = QuantizationRegistry()

    registry.register(
        "fake-int8",
        lambda ctx: FakeLinearProvider(),
    )

    context = QuantizationContext(
        checkpoint_dir="/synthetic",
        device="cpu",
    )

    provider = registry.create("fake-int8", context)
    result = registry.inspect(provider, context)

    assert result.bits == 8
    assert result.num_layers == 3
    assert result.execution_mode == "linear"


def test_custom_stage_provider():
    registry = QuantizationRegistry()

    registry.register(
        "fake-stage",
        lambda ctx: FakeStageProvider(),
    )

    context = QuantizationContext("/synthetic")

    result = registry.inspect(
        registry.create("fake-stage", context),
        context,
    )

    assert result.execution_mode == "stage"


def test_memory_planner_inputs():
    result = profile().planner_inputs()

    assert result["layer_weights_gib"] == [
        300 / 1024,
        400 / 1024,
        500 / 1024,
    ]

    assert result["additional_rank_reserve_gib"] == (
        64 / 1024
    )

    assert result["kv_bytes_per_layer_per_token"] == 4096


@pytest.mark.parametrize(
    "changes",
    [
        {"bits": 0},
        {"bits": True},
        {"group_size": 0},
        {"layer_storage_bytes": (100,)},
        {"layer_storage_bytes": (100, -1)},
        {"workspace_bytes_per_rank": -1},
        {"execution_mode": "unknown"},
    ],
)
def test_invalid_profiles_rejected(changes):
    base = dict(
        provider="fake",
        format_id="test",
        bits=8,
        group_size=128,
        layer_storage_bytes=(100, 100),
        first_stage_bytes=0,
        last_stage_bytes=0,
        kv_bytes_per_layer_per_token=0,
    )

    base.update(changes)

    with pytest.raises(QuantizationProviderError):
        QuantizationProfile(**base)


def test_missing_execution_adapter_rejected():
    class MissingAdapter:
        def inspect_checkpoint(self, context):
            return profile()

        def validate_runtime(self, context, profile):
            pass

    context = QuantizationContext("/synthetic")

    with pytest.raises(
        QuantizationProviderError,
        match="build_linear",
    ):
        QuantizationRegistry.inspect(
            MissingAdapter(), context
        )


def test_runtime_incompatibility_rejected():
    context = QuantizationContext(
        "/synthetic",
        device="cuda:0",
    )

    with pytest.raises(
        QuantizationProviderError,
        match="only supports CPU",
    ):
        QuantizationRegistry.inspect(
            FakeLinearProvider(), context
        )


def test_duplicate_registration_rejected():
    registry = QuantizationRegistry()

    registry.register("fake", FakeLinearProvider)

    with pytest.raises(QuantizationProviderError):
        registry.register("fake", FakeLinearProvider)


def test_unknown_provider_rejected():
    registry = QuantizationRegistry()

    with pytest.raises(QuantizationProviderError):
        registry.create(
            "missing",
            QuantizationContext("/synthetic"),
        )


def test_invalid_import_path_rejected():
    registry = QuantizationRegistry()

    with pytest.raises(QuantizationProviderError):
        registry.create(
            "invalid-factory:",
            QuantizationContext("/synthetic"),
        )
