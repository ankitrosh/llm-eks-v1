import copy
import json
import tempfile
import unittest
from pathlib import Path

from llm_runtime.inspection.registry import (
    InspectorSpec,
    ModelInspectorRegistry,
    InspectorContractError,
    UnsupportedArchitectureError,
    default_registry,
)
from llm_runtime.inspection.model_layout import (
    QWEN3_WEIGHT_SUFFIXES,
    generate_layout as legacy_qwen3,
)
from llm_runtime.extensions import (
    ArtifactValidationError,
    load_plugin_context,
)


def qwen_config():
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 2,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "hidden_size": 5120,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
    }


class TestV34ArchitectureRegistry(unittest.TestCase):

    def test_existing_qwen3_output_unchanged(self):
        config = qwen_config()

        actual = default_registry().inspect(
            config, "test/qwen3"
        )

        expected = legacy_qwen3(
            config, "test/qwen3"
        )

        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]["num_layers"], 2)

    def test_checkpoint_backed_dispatch(self):
        weight_map = {
            "model.embed_tokens.weight": "shard-1",
            "model.norm.weight": "shard-1",
            "lm_head.weight": "shard-1",
        }

        for index in range(2):
            for suffix in QWEN3_WEIGHT_SUFFIXES:
                weight_map[
                    f"model.layers.{index}.{suffix}"
                ] = "shard-1"

        layout, caps = default_registry().inspect(
            qwen_config(),
            "test/qwen3",
            {"weight_map": weight_map},
        )

        self.assertEqual(
            layout["verification"],
            "checkpoint_weight_index",
        )
        self.assertEqual(caps["kv_heads"], 8)

    def test_unknown_architecture_rejected(self):
        config = qwen_config()
        config["architectures"] = [
            "UnsupportedForCausalLM"
        ]

        with self.assertRaises(
            UnsupportedArchitectureError
        ):
            default_registry().inspect(
                config, "test/unknown"
            )

    def test_duplicate_registration_rejected(self):
        registry = default_registry()

        with self.assertRaises(ValueError):
            registry.register(
                InspectorSpec(
                    "Qwen3ForCausalLM",
                    legacy_qwen3,
                )
            )

    def test_custom_architecture_dispatched(self):
        calls = []

        def inspect_custom(config, model_id, index):
            calls.append(model_id)

            return (
                {
                    "schema_version": 1,
                    "kind": "logical_checkpoint_layout",
                    "architecture": "CustomForCausalLM",
                    "model_id": model_id,
                    "num_layers": 1,
                    "transformer_container": "transformer.blocks",
                    "layers": [{
                        "index": 0,
                        "module_path": "transformer.blocks.0",
                    }],
                    "verification": "config_inferred_unverified",
                },
                {
                    "schema_version": 1,
                    "kind": "model_capabilities",
                    "architecture": "CustomForCausalLM",
                    "model_id": model_id,
                    "num_layers": 1,
                },
            )

        registry = default_registry([
            InspectorSpec(
                architecture="CustomForCausalLM",
                inspect=inspect_custom,
            )
        ])

        config = {
            "architectures": ["CustomForCausalLM"]
        }

        layout, caps = registry.inspect(
            config, "test/custom"
        )

        self.assertEqual(calls, ["test/custom"])
        self.assertEqual(
            layout["transformer_container"],
            "transformer.blocks",
        )
        self.assertEqual(caps["num_layers"], 1)

    def test_invalid_inspector_output_rejected(self):
        registry = ModelInspectorRegistry()
        registry.register(
            InspectorSpec(
                "BrokenForCausalLM",
                lambda *args: ({}, {}),
            )
        )

        with self.assertRaises(
            InspectorContractError
        ):
            registry.inspect(
                {
                    "architectures": [
                        "BrokenForCausalLM"
                    ]
                },
                "test/broken",
            )

    def test_custom_layer_paths_in_artifact_context(self):
        source = {
            "revision": "test-revision",
            "config_sha256": "example-config-hash",
            "index_sha256": None,
        }

        model = {
            "schema_version": 1,
            "kind": "model_capabilities",
            "model_id": "test/custom",
            "architecture": "CustomForCausalLM",
            "num_layers": 2,
            "source": copy.deepcopy(source),
        }

        layout = {
            "schema_version": 1,
            "kind": "logical_checkpoint_layout",
            "model_id": "test/custom",
            "architecture": "CustomForCausalLM",
            "num_layers": 2,
            "transformer_container": "transformer.blocks",
            "verification": "config_inferred_unverified",
            "layers": [
                {
                    "index": i,
                    "module_path": f"transformer.blocks.{i}",
                }
                for i in range(2)
            ],
            "source": copy.deepcopy(source),
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            model_path = root / "model.json"
            layout_path = root / "layout.json"

            model_path.write_text(json.dumps(model))
            layout_path.write_text(json.dumps(layout))

            kwargs = {
                "model_capabilities_path": model_path,
                "model_layout_path": layout_path,
                "backend": "research_pytorch",
            }

            context = load_plugin_context(
                **kwargs,
                require_verified=False,
            )

            self.assertEqual(
                context.layout["layers"][1]["module_path"],
                "transformer.blocks.1",
            )

            with self.assertRaises(
                ArtifactValidationError
            ):
                load_plugin_context(
                    **kwargs,
                    require_verified=True,
                )


if __name__ == "__main__":
    unittest.main()
