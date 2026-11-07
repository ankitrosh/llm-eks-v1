"""V3.1 quantization provider contract.

CPU-only interfaces for checkpoint inspection, memory estimates,
and optional model-specific execution adapters.

No quantization kernels or checkpoint loading are implemented here.
"""

from __future__ import annotations

import importlib
import math

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

GIB = 1024 ** 3


class QuantizationProviderError(ValueError):
    pass


@dataclass(frozen=True)
class QuantizationContext:
    checkpoint_dir: str
    device: str = "cpu"
    activation_dtype: str = "float16"
    options: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.checkpoint_dir, str)
            or not self.checkpoint_dir
        ):
            raise QuantizationProviderError(
                "checkpoint_dir is required"
            )

        if self.device.split(":")[0] not in ("cpu", "cuda"):
            raise QuantizationProviderError(
                "Unsupported device type"
            )

        if self.activation_dtype not in (
            "float16", "bfloat16", "float32"
        ):
            raise QuantizationProviderError(
                "Unsupported activation dtype"
            )

        if not isinstance(self.options, Mapping):
            raise QuantizationProviderError(
                "options must be a mapping"
            )


@dataclass(frozen=True)
class QuantizationProfile:
    """Provider-reported checkpoint and runtime-memory metadata.

    Storage bytes should include packed weights, scales,
    zero-points and other per-layer checkpoint tensors.

    Runtime workspace is estimated separately.
    """

    provider: str
    format_id: str
    bits: int
    group_size: int | None
    layer_storage_bytes: tuple[int, ...]
    first_stage_bytes: int
    last_stage_bytes: int
    kv_bytes_per_layer_per_token: int
    workspace_bytes_per_rank: int = 0
    execution_mode: str = "linear"

    def __post_init__(self):
        if not self.provider or not self.format_id:
            raise QuantizationProviderError(
                "Provider and checkpoint format are required"
            )

        if (
            type(self.bits) is not int
            or not 1 <= self.bits <= 16
        ):
            raise QuantizationProviderError(
                "Invalid quantization bit width"
            )

        if (
            self.group_size is not None
            and (
                type(self.group_size) is not int
                or (
                    self.group_size != -1
                    and self.group_size < 1
                )
            )
        ):
            raise QuantizationProviderError(
                "Invalid group size"
            )

        sizes = self.layer_storage_bytes

        if (
            not isinstance(sizes, (tuple, list))
            or len(sizes) < 2
            or any(
                type(size) is not int or size < 1
                for size in sizes
            )
        ):
            raise QuantizationProviderError(
                "Invalid per-layer storage sizes"
            )

        for name in (
            "first_stage_bytes",
            "last_stage_bytes",
            "kv_bytes_per_layer_per_token",
            "workspace_bytes_per_rank",
        ):
            value = getattr(self, name)

            if type(value) is not int or value < 0:
                raise QuantizationProviderError(
                    f"Invalid {name}"
                )

        if self.execution_mode not in ("linear", "stage"):
            raise QuantizationProviderError(
                "execution_mode must be linear or stage"
            )

        object.__setattr__(
            self, "layer_storage_bytes", tuple(sizes)
        )

    @property
    def num_layers(self):
        return len(self.layer_storage_bytes)

    def planner_inputs(self):
        """Memory inputs for the existing partition planner.

        These are estimates, not measured peak CUDA allocations.
        Workspace is additional per-rank reserve.
        """
        return {
            "layer_weights_gib": [
                value / GIB
                for value in self.layer_storage_bytes
            ],
            "first_stage_gib": self.first_stage_bytes / GIB,
            "last_stage_gib": self.last_stage_bytes / GIB,
            "kv_bytes_per_layer_per_token":
                self.kv_bytes_per_layer_per_token,
            "additional_rank_reserve_gib":
                self.workspace_bytes_per_rank / GIB,
        }


@runtime_checkable
class QuantizationProvider(Protocol):
    """Custom implementation supplied by a trusted Python plugin."""

    def inspect_checkpoint(
        self, context: QuantizationContext
    ) -> QuantizationProfile:
        ...

    def validate_runtime(
        self,
        context: QuantizationContext,
        profile: QuantizationProfile,
    ) -> None:
        ...

    # For execution_mode="linear":
    # build_linear(...) returns a quantized linear module.
    #
    # For execution_mode="stage":
    # load_stage(...) returns a callable stage executor.
    #
    # These methods are checked according to execution_mode.


class QuantizationRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, name, factory):
        if (
            not isinstance(name, str)
            or not name
            or ":" in name
            or not callable(factory)
        ):
            raise QuantizationProviderError(
                "Invalid provider registration"
            )

        if name in self._factories:
            raise QuantizationProviderError(
                f"Provider already registered: {name}"
            )

        self._factories[name] = factory

    @staticmethod
    def _import_factory(spec):
        module, separator, symbol = spec.partition(":")

        if (
            not separator
            or not module
            or not symbol
            or not all(
                part.isidentifier()
                for part in (
                    module.split(".") + symbol.split(".")
                )
            )
        ):
            raise QuantizationProviderError(
                "Provider factory must use module:function"
            )

        try:
            obj = importlib.import_module(module)

            for part in symbol.split("."):
                obj = getattr(obj, part)

        except (ImportError, AttributeError) as exc:
            raise QuantizationProviderError(
                f"Unable to import provider {spec}"
            ) from exc

        if not callable(obj):
            raise QuantizationProviderError(
                "Provider factory is not callable"
            )

        return obj

    def create(self, name, context):
        if not isinstance(context, QuantizationContext):
            raise QuantizationProviderError(
                "Expected QuantizationContext"
            )

        if name in self._factories:
            factory = self._factories[name]
        elif isinstance(name, str) and ":" in name:
            factory = self._import_factory(name)
        else:
            raise QuantizationProviderError(
                f"Unknown quantization provider: {name}"
            )

        provider = factory(context)

        for method in (
            "inspect_checkpoint",
            "validate_runtime",
        ):
            if not callable(getattr(provider, method, None)):
                raise QuantizationProviderError(
                    f"Provider missing {method}"
                )

        return provider

    @staticmethod
    def inspect(provider, context):
        profile = provider.inspect_checkpoint(context)

        if not isinstance(profile, QuantizationProfile):
            raise QuantizationProviderError(
                "Provider must return QuantizationProfile"
            )

        required_method = (
            "build_linear"
            if profile.execution_mode == "linear"
            else "load_stage"
        )

        if not callable(
            getattr(provider, required_method, None)
        ):
            raise QuantizationProviderError(
                f"Provider missing {required_method}"
            )

        provider.validate_runtime(context, profile)

        return profile
