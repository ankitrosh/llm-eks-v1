import unittest

from llm_runtime.extensions import (
    CompatibilityError,
    PluginContext,
    PluginRegistry,
)
from llm_runtime.backends.vllm import (
    VLLMLaunchConfig,
    register_vllm_backend,
)


class TestVLLMBackend(unittest.TestCase):

    def setUp(self):
        self.registry = PluginRegistry()
        register_vllm_backend(self.registry)

        self.context = PluginContext(
            model={
                "model_id": "Qwen/Qwen3-14B",
                "architecture": "Qwen3ForCausalLM",
            },
            layout={
                "transformer": {"num_layers": 40}
            },
            runtime={
                "deployment": {
                    "model_path": "/models/qwen3-14b",
                    "nnodes": 2,
                    "gpus_per_node": 1,
                    "pp": 2,
                    "tp": 1,
                    "master_addr": "vllm-rank0.llm.svc",
                    "max_model_len": 2048,
                }
            },
            backend="vllm",
        )

    def test_two_node_pipeline_commands(self):
        backend = self.registry.build(
            "backend", "vllm", self.context
        )

        plan = backend.plan()
        nodes = plan["nodes"]

        self.assertEqual(len(nodes), 2)
        self.assertEqual(plan["pipeline_parallel_size"], 2)
        self.assertEqual(plan["tensor_parallel_size"], 1)

        rank0 = nodes[0]["command"]
        rank1 = nodes[1]["command"]

        self.assertIn("--nnodes", rank0)
        self.assertIn("--master-addr", rank0)
        self.assertIn("--pipeline-parallel-size", rank0)

        self.assertNotIn("--headless", rank0)
        self.assertIn("--headless", rank1)

        self.assertEqual(
            rank0[rank0.index("--node-rank") + 1], "0"
        )
        self.assertEqual(
            rank1[rank1.index("--node-rank") + 1], "1"
        )

    def test_insufficient_gpu_resources_rejected(self):
        with self.assertRaises(CompatibilityError):
            VLLMLaunchConfig(
                model="test-model",
                nnodes=1,
                gpus_per_node=1,
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                master_addr="localhost",
            )

    def test_invalid_node_rank_rejected(self):
        backend = self.registry.build(
            "backend", "vllm", self.context
        )

        with self.assertRaises(ValueError):
            backend.command_for_rank(2)

    def test_wrong_backend_context_rejected(self):
        context = PluginContext(
            self.context.model,
            self.context.layout,
            self.context.runtime,
            "research_pytorch",
        )

        with self.assertRaises(CompatibilityError):
            self.registry.build(
                "backend", "vllm", context
            )


if __name__ == "__main__":
    unittest.main()
