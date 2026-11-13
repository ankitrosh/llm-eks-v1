"""Backend-agnostic extension contracts, version 0.1."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

KINDS = frozenset({
    "attention",
    "kv_cache",
    "scheduler",
    "backend",
})


class CompatibilityError(ValueError):
    """An extension cannot run with this configuration."""


@dataclass(frozen=True)
class PluginContext:
    """Model and runtime metadata passed into factories."""

    model: Mapping[str, Any]
    layout: Mapping[str, Any]
    runtime: Mapping[str, Any]
    backend: str


@dataclass(frozen=True)
class PluginSpec:
    kind: str
    name: str
    supported_backends: frozenset[str]
    factory: Callable[[PluginContext], Any]
    compatible: Callable[[PluginContext], bool] = (
        lambda _: True
    )

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(
                f"Unsupported plugin kind: {self.kind}"
            )

        if not self.name or not self.supported_backends:
            raise ValueError(
                "Plugin name and supported_backends are required"
            )

        if not callable(self.factory) or not callable(
            self.compatible
        ):
            raise TypeError(
                "factory and compatible must be callable"
            )
