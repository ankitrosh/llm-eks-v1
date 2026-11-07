import unittest

from llm_runtime.v3_execution_factory import GenericStageFactory
from llm_runtime.v3_pipeline_scheduler import StageResult, StageTask
from llm_runtime.v3_partition_utils import (
    contiguous_stage_ranges,
    stage_bounds,
)


FACTORY_PATH = "examples.v3_toy_provider:ToyStageFactory"


class BYOMContractTests(unittest.TestCase):

    def factory(self):
        return GenericStageFactory(
            factory_path=FACTORY_PATH,
            kwargs={"offsets": [2, 10]},
        )

    def test_custom_provider_resolves_both_ranks(self):
        factory = self.factory()

        for rank, expected in ((0, 7), (1, 15)):
            stage = factory(rank)

            result = stage(
                StageTask(101, 0, 0, 5),
                None,
            )

            self.assertIsInstance(result, StageResult)
            self.assertEqual(result.payload, expected)
            self.assertEqual(result.cache_state["calls"], 1)
            self.assertEqual(stage.forward_calls, 1)

    def test_request_caches_remain_independent(self):
        stage = self.factory()(0)
        caches = {}

        sequence = [
            (101, 0, 5),
            (202, 0, 9),
            (101, 1, 6),
        ]

        for request_id, step, value in sequence:
            result = stage(
                StageTask(request_id, step, step, value),
                caches.get(request_id),
            )
            caches[request_id] = result.cache_state

        self.assertEqual(caches[101]["calls"], 2)
        self.assertEqual(caches[202]["calls"], 1)
        self.assertEqual(stage.forward_calls, 3)

    def test_wrong_cache_is_rejected(self):
        stage = self.factory()(0)

        first = stage(
            StageTask(101, 0, 0, 5),
            None,
        )

        with self.assertRaises(ValueError):
            stage(
                StageTask(202, 0, 0, 9),
                first.cache_state,
            )

    def test_invalid_rank_rejected(self):
        with self.assertRaises(ValueError):
            self.factory()(2)

    def test_balanced_partitions(self):
        self.assertEqual(
            contiguous_stage_ranges(40, [20, 20]),
            ((0, 20), (20, 40)),
        )

    def test_uneven_partitions(self):
        self.assertEqual(
            stage_bounds(40, [18, 22], 1),
            (18, 40),
        )

    def test_missing_layers_rejected(self):
        with self.assertRaises(ValueError):
            contiguous_stage_ranges(40, [18, 20])

    def test_invalid_layer_counts_rejected(self):
        with self.assertRaises(ValueError):
            contiguous_stage_ranges(40, [True, 39])


if __name__ == "__main__":
    unittest.main()
