"""Explicit, fail-closed plugin selection."""

from .contracts import (
    CompatibilityError,
    PluginContext,
    PluginSpec,
)


class PluginRegistry:

    def __init__(self):
        self._specs = {}

    def register(self, spec: PluginSpec):
        key = (spec.kind, spec.name)

        if key in self._specs:
            raise ValueError(
                f"Duplicate plugin registration: {key}"
            )

        self._specs[key] = spec

    def resolve(
        self,
        kind: str,
        name: str,
        context: PluginContext,
    ) -> PluginSpec:

        key = (kind, name)

        if key not in self._specs:
            raise KeyError(
                f"Unknown plugin: {kind}/{name}"
            )

        spec = self._specs[key]

        if context.backend not in spec.supported_backends:
            raise CompatibilityError(
                f"{kind}/{name} does not support "
                f"backend {context.backend}"
            )

        if not spec.compatible(context):
            raise CompatibilityError(
                f"{kind}/{name} rejected model "
                "or runtime capabilities"
            )

        return spec

    def build(
        self,
        kind: str,
        name: str,
        context: PluginContext,
    ):
        return self.resolve(
            kind, name, context
        ).factory(context)
