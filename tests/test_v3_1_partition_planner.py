import unittest

from llm_runtime.v3_1_partition_planner import (
    plan_contiguous_layers,
)


class PartitionPlannerTests(unittest.TestCase):

    def plan(self, gpus=(24.0, 24.0), **changes):
        options = {
            "layer_weights_gib": [0.5] * 40,
            "gpu_memory_gib": gpus,
            "gpu_reserve_gib": 2.0,
            "kv_bytes_per_layer_per_token": 4096,
            "max_context_tokens": 8192,
            "max_active_requests": 4,
            "first_stage_gib": 0.4,
            "last_stage_gib": 0.4,
        }

        options.update(changes)
        return plan_contiguous_layers(**options)

    def test_balanced_two_ranks(self):
        result = self.plan()

        self.assertEqual(
            [(p.start, p.end) for p in result],
            [(0, 20), (20, 40)],
        )

    def test_three_rank_contiguity(self):
        result = self.plan(gpus=(24.0,) * 3)

        self.assertEqual(result[0].start, 0)
        self.assertEqual(result[-1].end, 40)

        self.assertEqual(
            [p.end for p in result[:-1]],
            [p.start for p in result[1:]],
        )

    def test_four_rank_partition(self):
        result = self.plan(gpus=(24.0,) * 4)

        self.assertEqual(len(result), 4)

        self.assertEqual(
            sum(p.layer_count for p in result),
            40,
        )

    def test_heterogeneous_devices(self):
        result = self.plan(gpus=(16.0, 24.0))

        self.assertGreater(
            result[1].layer_count,
            result[0].layer_count,
        )

    def test_manual_partition(self):
        result = self.plan(
            manual_layer_counts=[18, 22]
        )

        self.assertEqual(
            [(p.start, p.end) for p in result],
            [(0, 18), (18, 40)],
        )

    def test_unsafe_manual_partition(self):
        with self.assertRaises(ValueError):
            self.plan(
                gpus=(12.0, 24.0),
                manual_layer_counts=[30, 10],
            )

    def test_no_feasible_partition(self):
        with self.assertRaises(ValueError):
            self.plan(gpus=(4.0, 4.0))

    def test_missing_layers(self):
        with self.assertRaises(ValueError):
            self.plan(
                manual_layer_counts=[18, 21]
            )

    def test_too_many_ranks(self):
        with self.assertRaises(ValueError):
            self.plan(gpus=(24.0,) * 41)

    def test_invalid_gpu_memory(self):
        with self.assertRaises(ValueError):
            self.plan(
                gpus=(float("nan"), 24.0)
            )

    def test_kv_memory_increases_with_context(self):
        small = self.plan(
            max_context_tokens=2048
        )

        large = self.plan(
            max_context_tokens=8192
        )

        self.assertGreater(
            sum(p.estimated_gib for p in large),
            sum(p.estimated_gib for p in small),
        )


if __name__ == "__main__":
    unittest.main()
