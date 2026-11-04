import unittest

from llm_runtime.v3_manifest import ManifestError
from llm_runtime.v3_attention_plan import (
    resolve_attention_plan,
)


class AttentionPlanTests(unittest.TestCase):

    def test_qwen_defaults_two_ranks(self):
        plan = resolve_attention_plan({
            "model": {"num_layers": 40},
            "pipeline": {"num_ranks": 2},
        })

        self.assertEqual(len(plan["stages"]), 2)

        self.assertEqual(
            [
                (s["start_layer"], s["end_layer"])
                for s in plan["stages"]
            ],
            [(0, 20), (20, 40)],
        )

        self.assertEqual(
            len(plan["stages"][0]["attention"]), 1
        )

    def test_gpt2_custom_layer_override(self):
        manifest = {
            "model": {"num_layers": 6},
            "pipeline": {"num_ranks": 3},
            "attention": {
                "default": {
                    "implementation": "model_native",
                    "backend": "auto",
                },
                "overrides": [{
                    "layers": [2, 3],
                    "implementation": "custom",
                    "plugin": (
                        "experiments.my_attention:MyAttention"
                    ),
                }],
            },
        }

        plan = resolve_attention_plan(manifest)

        stages = plan["stages"]

        self.assertEqual(len(stages), 3)

        custom = stages[1]["attention"][0]

        self.assertEqual(custom["start_layer"], 2)
        self.assertEqual(custom["end_layer"], 4)
        self.assertEqual(
            custom["execution"],
            "DECLARED_NOT_IMPORTED",
        )

        self.assertEqual(
            stages[0]["attention"][0]["implementation"],
            "model_native",
        )

    def test_plugin_is_not_imported(self):
        plan = resolve_attention_plan({
            "model": {"num_layers": 1},
            "pipeline": {"num_ranks": 1},
            "attention": {
                "overrides": [{
                    "layers": [0],
                    "implementation": "custom",
                    "plugin": (
                        "nonexistent_package.mod:Attention"
                    ),
                }],
            },
        })

        self.assertEqual(
            plan["stages"][0]["attention"][0]["execution"],
            "DECLARED_NOT_IMPORTED",
        )

    def test_partial_override_inherits_default(self):
        plan = resolve_attention_plan({
            "model": {"num_layers": 3},
            "pipeline": {"num_ranks": 1},
            "attention": {
                "default": {"backend": "sdpa"},
                "overrides": [{
                    "layers": [1],
                    "backend": "eager",
                }],
            },
        })

        segments = plan["stages"][0]["attention"]

        self.assertEqual(
            [
                (s["start_layer"], s["end_layer"], s["backend"])
                for s in segments
            ],
            [
                (0, 1, "sdpa"),
                (1, 2, "eager"),
                (2, 3, "sdpa"),
            ],
        )

    def test_unknown_layers_fail(self):
        with self.assertRaises(ManifestError):
            resolve_attention_plan({
                "model": {"id": "custom/model"},
            })

    def test_conflicting_overrides_fail(self):
        with self.assertRaises(ManifestError):
            resolve_attention_plan({
                "model": {"num_layers": 4},
                "attention": {
                    "overrides": [
                        {"layers": [1], "backend": "sdpa"},
                        {"layers": [1], "backend": "eager"},
                    ],
                },
            })

    def test_bad_plugin_options_fail(self):
        with self.assertRaises(ManifestError):
            resolve_attention_plan({
                "model": {"num_layers": 2},
                "attention": {
                    "default": {
                        "options": [1, 2]
                    },
                },
            })


if __name__ == "__main__":
    unittest.main()
