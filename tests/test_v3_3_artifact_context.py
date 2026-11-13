import json
import tempfile
import unittest
from pathlib import Path

from llm_runtime.extensions import (
    ArtifactValidationError,
    PluginRegistry,
    PluginSpec,
    load_plugin_context,
)
from llm_runtime.backends.vllm import (
    register_vllm_backend,
)


ROOT = (
    Path(__file__).resolve().parents[1]
    / "infrastructure/profiles/v3-3/qwen3-14b"
)


def build_context(**overrides):
    args = {
        "model_capabilities_path":
            ROOT / "model_capabilities.json",
        "model_layout_path":
            ROOT / "model_layout.json",
        "backend": "vllm",
        "runtime": {},
    }

    args.update(overrides)

    return load_plugin_context(**args)


class TestArtifactContext(unittest.TestCase):

    def test_real_qwen3_metadata(self):
        ctx = build_context()

        self.assertEqual(
            ctx.model["architecture"],
            "Qwen3ForCausalLM",
        )
        self.assertEqual(ctx.model["num_layers"], 40)
        self.assertEqual(ctx.model["q_heads"], 40)
        self.assertEqual(ctx.model["kv_heads"], 8)
        self.assertEqual(ctx.model["head_dim"], 128)

        self.assertEqual(len(ctx.layout["layers"]), 40)

        first = ctx.layout["layers"][0]
        last = ctx.layout["layers"][-1]

        self.assertEqual(
            first["attention"]["module_path"],
            "model.layers.0.self_attn",
        )
        self.assertEqual(
            last["module_path"],
            "model.layers.39",
        )

    def test_user_factory_receives_actual_layout(self):
        registry = PluginRegistry()
        calls = []

        def custom_factory(ctx):
            layer = ctx.layout["layers"][0]

            calls.append(layer["module_path"])

            return {
                "kv_heads": ctx.model["kv_heads"],
                "q_path": layer["attention"][
                    "projections"
                ]["q"],
            }

        registry.register(
            PluginSpec(
                kind="attention",
                name="layout-aware-test",
                supported_backends=frozenset({
                    "research_pytorch"
                }),
                factory=custom_factory,
            )
        )

        ctx = build_context(
            backend="research_pytorch"
        )

        result = registry.build(
            "attention",
            "layout-aware-test",
            ctx,
        )

        self.assertEqual(
            result["kv_heads"], 8
        )
        self.assertEqual(
            result["q_path"],
            "model.layers.0.self_attn.q_proj",
        )
        self.assertEqual(
            calls, ["model.layers.0"]
        )

    def test_vllm_backend_uses_generated_context(self):
        registry = PluginRegistry()
        register_vllm_backend(registry)

        ctx = build_context(
            runtime={
                "deployment": {
                    "model_path": "/models/qwen3-14b",
                    "nnodes": 2,
                    "gpus_per_node": 1,
                    "pp": 2,
                    "tp": 1,
                    "master_addr":
                        "vllm-rank0.llm.svc",
                    "max_model_len": 2048,
                }
            }
        )

        backend = registry.build(
            "backend", "vllm", ctx
        )

        plan = backend.plan()

        self.assertEqual(
            plan["pipeline_parallel_size"], 2
        )
        self.assertEqual(
            len(plan["nodes"]), 2
        )

    def test_metadata_is_read_only(self):
        ctx = build_context()

        with self.assertRaises(TypeError):
            ctx.model["num_layers"] = 100

        with self.assertRaises(TypeError):
            ctx.layout["layers"][0]["index"] = 50

    def test_inconsistent_revision_rejected(self):
        layout = json.loads(
            (ROOT / "model_layout.json").read_text()
        )

        layout["source"]["revision"] = "different"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            path.write_text(json.dumps(layout))

            with self.assertRaises(
                ArtifactValidationError
            ):
                build_context(
                    model_layout_path=path
                )

    def test_unverified_layout_rejected(self):
        layout = json.loads(
            (ROOT / "model_layout.json").read_text()
        )

        layout["verification"] = (
            "config_inferred_unverified"
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layout.json"
            path.write_text(json.dumps(layout))

            with self.assertRaises(
                ArtifactValidationError
            ):
                build_context(
                    model_layout_path=path
                )

    def test_invalid_context_limit_rejected(self):
        with self.assertRaises(
            ArtifactValidationError
        ):
            build_context(
                runtime={
                    "deployment": {
                        "max_model_len": 50000
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
