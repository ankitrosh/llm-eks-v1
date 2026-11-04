import copy
import unittest

from llm_runtime.v3_runtime_plan import (
    RuntimePlanError,
    compile_runtime_plan,
)


CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 40,
    "tie_word_embeddings": False,
}

MANIFEST = {
    "model": {
        "id": "Qwen/Qwen3-14B",
        "num_layers": 40,
        "dtype": "bfloat16",
    },
    "pipeline": {
        "num_ranks": 2,
        "partition": "equal_layers",
    },
    "attention": {
        "default": {
            "implementation": "model_native",
            "backend": "sdpa",
        },
    },
    "runtime": {
        "device": "cuda",
        "distributed_backend": "nccl",
    },
    "scheduler": {
        "mode": "sequential",
        "max_active_requests": 1,
    },
}


class RuntimePlanTests(unittest.TestCase):

    def test_bf16_plan(self):
        plan = compile_runtime_plan(MANIFEST, CONFIG)

        self.assertEqual(
            plan["pipeline"]["split"], 20
        )

        self.assertEqual(
            plan["runtime_environment"]["LLM_STAGE_IMPL"],
            "v3_adapter",
        )

    def test_unequal_split(self):
        manifest = copy.deepcopy(MANIFEST)

        manifest["pipeline"] = {
            "num_ranks": 2,
            "partition": "manual",
            "stages": [
                {
                    "rank": 0,
                    "start_layer": 0,
                    "end_layer": 23,
                },
                {
                    "rank": 1,
                    "start_layer": 23,
                    "end_layer": 40,
                },
            ],
        }

        plan = compile_runtime_plan(manifest, CONFIG)

        self.assertEqual(
            plan["pipeline"]["split"], 23
        )

    def test_prefix_cache_settings(self):
        manifest = copy.deepcopy(MANIFEST)

        manifest["cache"] = {
            "prefix_enabled": True,
            "max_mib_per_rank": 256,
        }

        plan = compile_runtime_plan(manifest, CONFIG)

        self.assertEqual(
            plan["runtime_environment"][
                "LLM_PREFIX_CACHE_ENABLED"
            ],
            "1",
        )

    def test_three_ranks_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["pipeline"]["num_ranks"] = 3

        with self.assertRaises(RuntimePlanError):
            compile_runtime_plan(manifest, CONFIG)

    def test_custom_attention_rejected(self):
        manifest = copy.deepcopy(MANIFEST)

        manifest["attention"]["overrides"] = [{
            "layers": [2],
            "implementation": "custom",
            "plugin": "experiments.my_attention:MyAttention",
        }]

        with self.assertRaises(RuntimePlanError):
            compile_runtime_plan(manifest, CONFIG)

    def test_concurrent_scheduler_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["scheduler"]["mode"] = "pipelined"

        with self.assertRaises(RuntimePlanError):
            compile_runtime_plan(manifest, CONFIG)

    def test_layer_mismatch_rejected(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["model"]["num_layers"] = 64

        with self.assertRaises(RuntimePlanError):
            compile_runtime_plan(manifest, CONFIG)

    def test_unsupported_quantization_rejected(self):
        config = copy.deepcopy(CONFIG)

        config["quantization_config"] = {
            "quant_method": "gptq",
            "bits": 4,
        }

        with self.assertRaises(RuntimePlanError):
            compile_runtime_plan(MANIFEST, config)


if __name__ == "__main__":
    unittest.main()
