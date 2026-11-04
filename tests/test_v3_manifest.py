import unittest

from llm_runtime.v3_manifest import (
    ManifestError,
    validate,
)


class ManifestTests(unittest.TestCase):

    def test_equal_layer_plan(self):
        r = validate({
            "model": {"num_layers": 40},
            "pipeline": {"num_ranks": 3}
        })

        ranges = [
            (s["start_layer"], s["end_layer"])
            for s in r["stages"]
        ]

        self.assertEqual(
            ranges,
            [(0, 14), (14, 27), (27, 40)]
        )

    def test_manual_partition(self):
        r = validate({
            "model": {"num_layers": 6},
            "pipeline": {
                "num_ranks": 3,
                "partition": "manual",
                "stages": [
                    {
                        "rank": 0,
                        "start_layer": 0,
                        "end_layer": 2
                    },
                    {
                        "rank": 1,
                        "start_layer": 2,
                        "end_layer": 4
                    },
                    {
                        "rank": 2,
                        "start_layer": 4,
                        "end_layer": 6
                    }
                ]
            }
        })

        self.assertEqual(len(r["stages"]), 3)

    def test_gap_rejected(self):
        with self.assertRaises(ManifestError):
            validate({
                "model": {"num_layers": 6},
                "pipeline": {
                    "num_ranks": 2,
                    "partition": "manual",
                    "stages": [
                        {
                            "rank": 0,
                            "start_layer": 0,
                            "end_layer": 2
                        },
                        {
                            "rank": 1,
                            "start_layer": 3,
                            "end_layer": 6
                        }
                    ]
                }
            })

    def test_custom_plugin_rule(self):
        with self.assertRaises(ManifestError):
            validate({
                "attention": {
                    "overrides": [{
                        "layers": [2],
                        "implementation": "custom"
                    }]
                }
            })

    def test_duplicate_override_rejected(self):
        with self.assertRaises(ManifestError):
            validate({
                "model": {"num_layers": 6},
                "attention": {
                    "overrides": [
                        {"layers": [2, 3], "backend": "sdpa"},
                        {"layers": [3], "backend": "eager"}
                    ]
                }
            })

    def test_cpu_nccl_rejected(self):
        with self.assertRaises(ManifestError):
            validate({
                "runtime": {
                    "device": "cpu",
                    "distributed_backend": "nccl"
                }
            })

    def test_unknown_layers_not_assumed(self):
        r = validate({
            "model": {"id": "custom/model"},
            "pipeline": {"num_ranks": 3}
        })

        self.assertEqual(r["stages"], [])
        self.assertIsNone(r["num_layers"])

    def test_per_layer_override_declared(self):
        r = validate({
            "model": {"num_layers": 6},
            "attention": {
                "overrides": [{
                    "layers": [1, 2],
                    "implementation": "custom",
                    "plugin": (
                        "experiments.my_attention:MyAttention"
                    )
                }]
            }
        })

        self.assertTrue(
            any(
                "overrides parsed only" in note
                for note in r["not_run"]
            )
        )


if __name__ == "__main__":
    unittest.main()
