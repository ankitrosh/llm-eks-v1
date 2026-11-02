"""Conservative optimization eligibility checks, independent of model families.

This module does not load weights, probe installed packages, or enable kernels.
A positive result is only permission to attempt real runtime validation.
"""

from dataclasses import dataclass
from typing import Callable

from .model_metadata import ModelMetadata


@dataclass(frozen=True)
class CapabilityContext:
    """Facts supplied by model/backend discovery and hardware inspection.

    model_supported_backends must come from a validated model interface,
    not merely from recognizing the model family name.
    """

    device_type: str
    compute_capability: tuple[int, int] | None = None
    model_supported_backends: frozenset[str] = frozenset()
    runtime_available_backends: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CapabilityResult:
    name: str
    eligible: bool
    reason: str
    runtime_validation_required: bool


OptimizationCheck = Callable[
    [ModelMetadata, CapabilityContext],
    CapabilityResult,
]


class OptimizationRegistry:
    def __init__(self) -> None:
        self._checks: dict[str, OptimizationCheck] = {}

    def register(self, name: str, check: OptimizationCheck) -> None:
        if not name or name in self._checks:
            raise ValueError(
                f"Invalid or duplicate optimization: {name!r}"
            )
        self._checks[name] = check

    def evaluate(
        self,
        name: str,
        metadata: ModelMetadata,
        context: CapabilityContext,
    ) -> CapabilityResult:
        if name not in self._checks:
            raise KeyError(f"Unregistered optimization: {name}")
        return self._checks[name](metadata, context)

    def evaluate_all(
        self,
        metadata: ModelMetadata,
        context: CapabilityContext,
    ) -> dict[str, CapabilityResult]:
        return {
            name: self.evaluate(name, metadata, context)
            for name in sorted(self._checks)
        }


def _reject(name: str, reason: str) -> CapabilityResult:
    return CapabilityResult(name, False, reason, False)


def _candidate(name: str) -> CapabilityResult:
    return CapabilityResult(
        name,
        True,
        "Preliminary checks passed; GPU execution and output validation required",
        True,
    )


def _common_attention_check(
    name: str,
    metadata: ModelMetadata,
    context: CapabilityContext,
) -> CapabilityResult | None:

    if name not in context.model_supported_backends:
        return _reject(
            name,
            "Model interface has not advertised this backend",
        )

    if name not in context.runtime_available_backends:
        return _reject(
            name,
            "Backend not confirmed available in the runtime",
        )

    if metadata.num_attention_heads % metadata.num_key_value_heads != 0:
        return _reject(
            name,
            "Unsupported Q/KV head grouping",
        )

    return None


def check_sdpa(
    metadata: ModelMetadata,
    context: CapabilityContext,
) -> CapabilityResult:
    name = "sdpa"

    problem = _common_attention_check(name, metadata, context)
    if problem is not None:
        return problem

    # SDPA may select a math fallback, not necessarily fused attention.
    return _candidate(name)


def check_flash_attention_2(
    metadata: ModelMetadata,
    context: CapabilityContext,
) -> CapabilityResult:
    name = "flash_attention_2"

    problem = _common_attention_check(name, metadata, context)
    if problem is not None:
        return problem

    if context.device_type != "cuda":
        return _reject(
            name, "FlashAttention-2 requires a supported CUDA GPU"
        )

    if context.compute_capability is None:
        return _reject(
            name, "CUDA compute capability has not been inspected"
        )

    if context.compute_capability < (8, 0):
        return _reject(
            name, "This FlashAttention-2 backend requires Ampere or newer"
        )

    if metadata.dtype not in {"float16", "bfloat16"}:
        return _reject(
            name, "This FlashAttention-2 backend requires FP16 or BF16"
        )

    if metadata.head_dim > 256:
        return _reject(
            name, "Attention head dimension exceeds 256"
        )

    return _candidate(name)


def default_registry() -> OptimizationRegistry:
    registry = OptimizationRegistry()
    registry.register("sdpa", check_sdpa)
    registry.register("flash_attention_2", check_flash_attention_2)
    return registry
