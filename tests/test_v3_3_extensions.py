"""CPU-only extension contract tests."""

import unittest

from llm_runtime.extensions import (
    CompatibilityError,
    PluginContext,
    PluginRegistry,
    PluginSpec,
)


class TestExtensionContracts(unittest.TestCase):

    def setUp(self):
        self.registry = PluginRegistry()
        self.calls = []

        self.context = PluginContext(
            model={
                "architecture": "Qwen3ForCausalLM"
            },
            layout={
                "attention": {
                    "num_kv_heads": 8
                }
            },
            runtime={
                "dtype": "bfloat16"
            },
            backend="research_pytorch",
        )

    def test_user_override_is_actually_invoked(self):

        def wrong_on_purpose(ctx):
            self.calls.append(
                ctx.layout["attention"]["num_kv_heads"]
            )
            return {"fake_attention": True}

        self.registry.register(
            PluginSpec(
                kind="attention",
                name="instrumented-test-only",
                supported_backends=frozenset({
                    "research_pytorch"
                }),
                factory=wrong_on_purpose,
            )
        )

        result = self.registry.build(
            "attention",
            "instrumented-test-only",
            self.context,
        )

        self.assertEqual(
            result, {"fake_attention": True}
        )
        self.assertEqual(self.calls, [8])

    def test_unsupported_backend_rejected(self):

        self.registry.register(
            PluginSpec(
                kind="attention",
                name="test-only",
                supported_backends=frozenset({
                    "research_pytorch"
                }),
                factory=lambda ctx: self.calls.append(
                    "called"
                ),
            )
        )

        vllm_context = PluginContext(
            self.context.model,
            self.context.layout,
            self.context.runtime,
            "vllm",
        )

        with self.assertRaises(CompatibilityError):
            self.registry.build(
                "attention",
                "test-only",
                vllm_context,
            )

        self.assertEqual(self.calls, [])

    def test_model_capability_check(self):

        self.registry.register(
            PluginSpec(
                kind="kv_cache",
                name="only-gqa8",
                supported_backends=frozenset({
                    "research_pytorch"
                }),
                compatible=lambda ctx: (
                    ctx.layout["attention"]["num_kv_heads"]
                    == 8
                ),
                factory=lambda ctx: object(),
            )
        )

        altered = PluginContext(
            self.context.model,
            {"attention": {"num_kv_heads": 16}},
            self.context.runtime,
            "research_pytorch",
        )

        with self.assertRaises(CompatibilityError):
            self.registry.build(
                "kv_cache",
                "only-gqa8",
                altered,
            )

    def test_duplicate_registration_rejected(self):
        plugin = PluginSpec(
            "scheduler",
            "sample",
            frozenset({"vllm"}),
            lambda ctx: None,
        )

        self.registry.register(plugin)

        with self.assertRaises(ValueError):
            self.registry.register(plugin)


if __name__ == "__main__":
    unittest.main()
