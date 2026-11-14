"""Minimal custom planning/rendering example, not an LLM engine."""

import json

from llm_runtime.extensions.contracts import PluginSpec


class SampleBackend:

    def __init__(self, context):
        self.context = context

    def plan(self):
        return {
            "schema_version": 1,
            "backend": self.context.backend,
            "model": self.context.model["id"],
            "purpose": "demonstrate generic planning",
        }

    def render(self, plan):
        return {
            "example-configmap.json": {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "custom-example",
                },
                "data": {
                    "plan.json": json.dumps(plan),
                },
            }
        }


def register_extensions(registry):
    registry.register(
        PluginSpec(
            kind="backend",
            name="sample-backend",
            supported_backends=frozenset({
                "sample-runtime",
            }),
            factory=lambda context: SampleBackend(context),
        )
    )
