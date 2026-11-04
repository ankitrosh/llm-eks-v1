import unittest
from dataclasses import replace

from llm_runtime.model_metadata import ModelMetadata
from llm_runtime.optimization_registry import (
    CapabilityContext,
    default_registry,
)


class OptimizationRegistryTests(unittest.TestCase):

    def setUp(self):
        self.model = ModelMetadata(
            model_id="example/qwen3",
            model_type="qwen3",
            architectures=("Qwen3ForCausalLM",),
            num_layers=40,
            hidden_size=5120,
            num_attention_heads=40,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
            checkpoint_quantization=None,
        )

        self.context = CapabilityContext(
            device_type="cuda",
            compute_capability=(8, 6),
            model_supported_backends=frozenset({
                "sdpa", "flash_attention_2"
            }),
            runtime_available_backends=frozenset({
                "sdpa", "flash_attention_2"
            }),
        )

        self.registry = default_registry()

    def test_a10g_bf16_both_are_candidates(self):
        results = self.registry.evaluate_all(
            self.model, self.context
        )

        self.assertEqual(
            set(results),
            {"sdpa", "flash_attention_2"},
        )

        self.assertTrue(
            all(r.eligible for r in results.values())
        )

        self.assertTrue(
            all(r.runtime_validation_required for r in results.values())
        )

    def test_old_gpu_rejects_fa2(self):
        context = replace(
            self.context, compute_capability=(7, 5)
        )

        self.assertFalse(
            self.registry.evaluate(
                "flash_attention_2", self.model, context
            ).eligible
        )

        self.assertTrue(
            self.registry.evaluate(
                "sdpa", self.model, context
            ).eligible
        )

    def test_float32_rejects_fa2(self):
        model = replace(self.model, dtype="float32")

        self.assertFalse(
            self.registry.evaluate(
                "flash_attention_2", model, self.context
            ).eligible
        )

    def test_missing_model_backend_rejects_fa2(self):
        context = replace(
            self.context,
            model_supported_backends=frozenset({"sdpa"}),
        )

        self.assertFalse(
            self.registry.evaluate(
                "flash_attention_2", self.model, context
            ).eligible
        )

    def test_missing_runtime_backend_rejects_fa2(self):
        context = replace(
            self.context,
            runtime_available_backends=frozenset({"sdpa"}),
        )

        self.assertFalse(
            self.registry.evaluate(
                "flash_attention_2", self.model, context
            ).eligible
        )

    def test_incompatible_kv_group_rejected(self):
        model = replace(self.model, num_key_value_heads=7)

        self.assertFalse(
            self.registry.evaluate(
                "sdpa", model, self.context
            ).eligible
        )

    def test_unregistered_and_duplicate(self):
        with self.assertRaises(KeyError):
            self.registry.evaluate(
                "unknown", self.model, self.context
            )

        with self.assertRaises(ValueError):
            self.registry.register(
                "sdpa", lambda m, c: None
            )


if __name__ == "__main__":
    unittest.main()
