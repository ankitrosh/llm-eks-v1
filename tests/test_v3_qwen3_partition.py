import unittest

from llm_runtime.providers.v3_qwen3_partition import stage_bounds


class PartitionTests(unittest.TestCase):

    def test_balanced_default(self):
        self.assertEqual(stage_bounds(4, 0), (0, 2))
        self.assertEqual(stage_bounds(4, 1), (2, 4))

    def test_uneven_front_light(self):
        self.assertEqual(stage_bounds(4, 0, 1), (0, 1))
        self.assertEqual(stage_bounds(4, 1, 1), (1, 4))

    def test_uneven_back_light(self):
        self.assertEqual(stage_bounds(4, 0, 3), (0, 3))
        self.assertEqual(stage_bounds(4, 1, 3), (3, 4))

    def test_odd_layer_count(self):
        self.assertEqual(stage_bounds(5, 0, 2), (0, 2))
        self.assertEqual(stage_bounds(5, 1, 2), (2, 5))

    def test_invalid_boundaries(self):
        for value in (0, 4, -1, 1.5, True):
            with self.subTest(split_at=value):
                with self.assertRaises(ValueError):
                    stage_bounds(4, 0, value)

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            stage_bounds(4, 2, 2)


if __name__ == "__main__":
    unittest.main()
