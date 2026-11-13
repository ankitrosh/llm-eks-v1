"""Extensible architecture-specific model inspection."""

from dataclasses import dataclass
from typing import Callable

from .model_layout import generate_layout as inspect_qwen3


class UnsupportedArchitectureError(ValueError):
    pass


class InspectorContractError(ValueError):
    pass


@dataclass(frozen=True)
class InspectorSpec:
    architecture: str
    inspect: Callable

    def __post_init__(self):
        if (
            not isinstance(self.architecture, str)
            or not self.architecture.strip()
            or not callable(self.inspect)
        ):
            raise ValueError("Invalid inspector specification")


class ModelInspectorRegistry:
    def __init__(self):
        self._inspectors = {}

    def register(self, spec):
        if not isinstance(spec, InspectorSpec):
            raise TypeError("Expected InspectorSpec")

        if spec.architecture in self._inspectors:
            raise ValueError(
                f"Duplicate inspector: {spec.architecture}"
            )

        self._inspectors[spec.architecture] = spec

    def supported_architectures(self):
        return tuple(sorted(self._inspectors))

    def inspect(self, config, model_id, weight_index=None):
        if not isinstance(config, dict):
            raise ValueError("Model config must be a JSON object")

        architectures = config.get("architectures")

        if (
            not isinstance(architectures, list)
            or len(architectures) != 1
            or not isinstance(architectures[0], str)
        ):
            raise UnsupportedArchitectureError(
                "Expected exactly one declared model architecture"
            )

        architecture = architectures[0]
        spec = self._inspectors.get(architecture)

        if spec is None:
            raise UnsupportedArchitectureError(
                f"Unsupported architecture: {architecture}. "
                f"Registered: {self.supported_architectures()}"
            )

        result = spec.inspect(
            config, model_id, weight_index
        )

        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not all(isinstance(x, dict) for x in result)
        ):
            raise InspectorContractError(
                "Inspector must return (layout, capabilities)"
            )

        layout, capabilities = result

        expected = (
            (layout, "logical_checkpoint_layout"),
            (capabilities, "model_capabilities"),
        )

        for artifact, kind in expected:
            if (
                artifact.get("schema_version") != 1
                or artifact.get("kind") != kind
                or artifact.get("architecture") != architecture
                or artifact.get("model_id") != model_id
            ):
                raise InspectorContractError(
                    f"Invalid {kind} artifact"
                )

        if (
            layout.get("num_layers")
            != capabilities.get("num_layers")
        ):
            raise InspectorContractError(
                "Inspector returned inconsistent layer counts"
            )

        return layout, capabilities


def default_registry(extra_inspectors=()):
    registry = ModelInspectorRegistry()

    registry.register(
        InspectorSpec(
            architecture="Qwen3ForCausalLM",
            inspect=inspect_qwen3,
        )
    )

    for spec in extra_inspectors:
        registry.register(spec)

    return registry


def generate_layout(config, model_id, weight_index=None):
    """Default architecture-dispatched inspection entry point."""
    return default_registry().inspect(
        config, model_id, weight_index
    )
