import unittest
from dataclasses import dataclass

from llm_runtime.providers.v3_1_qwen3_layout import (
    Qwen3NRankLayout,
)


class LayoutTests(unittest.TestCase):

    def test_uneven_three_rank(self):
        layout = Qwen3NRankLayout(
            28, (0, 10, 19, 28)
        )

        self.assertEqual(layout.world_size, 3)
        self.assertEqual(
            layout.rank_bounds(1), (10, 19)
        )
        self.assertEqual(
            layout.owned_layers(2),
            tuple(range(19, 28)),
        )

    def test_role_ownership(self):
        layout = Qwen3NRankLayout(
            28, [0, 10, 19, 28]
        )

        self.assertEqual(
            [
                layout.stage_role(r)["embeddings"]
                for r in range(3)
            ],
            [True, False, False],
        )

        self.assertEqual(
            [
                layout.stage_role(r)["norm_and_lm_head"]
                for r in range(3)
            ],
            [False, False, True],
        )

    def test_from_counts(self):
        layout = Qwen3NRankLayout.from_counts(
            [10, 9, 9]
        )

        self.assertEqual(
            layout.boundaries,
            (0, 10, 19, 28),
        )

    def test_two_rank_layout(self):
        layout = Qwen3NRankLayout(
            40, [0, 20, 40]
        )

        self.assertEqual(
            layout.rank_bounds(1),
            (20, 40),
        )

    def test_empty_stage(self):
        with self.assertRaises(ValueError):
            Qwen3NRankLayout(
                28, [0, 10, 10, 28]
            )

    def test_missing_layer(self):
        with self.assertRaises(ValueError):
            Qwen3NRankLayout(
                28, [0, 10, 19, 27]
            )

    def test_boolean_boundary(self):
        with self.assertRaises(ValueError):
            Qwen3NRankLayout(
                28, [0, True, 28]
            )

    def test_invalid_rank(self):
        layout = Qwen3NRankLayout(
            28, [0, 10, 19, 28]
        )

        with self.assertRaises(ValueError):
            layout.rank_bounds(3)

    def test_from_placements(self):
        @dataclass
        class Placement:
            rank: int
            start: int
            end: int

        placements = [
            Placement(0, 0, 10),
            Placement(1, 10, 19),
            Placement(2, 19, 28),
        ]

        layout = Qwen3NRankLayout.from_placements(
            28, placements
        )

        self.assertEqual(
            layout.boundaries,
            (0, 10, 19, 28),
        )

    def test_bad_placements(self):
        @dataclass
        class Placement:
            rank: int
            start: int
            end: int

        with self.assertRaises(ValueError):
            Qwen3NRankLayout.from_placements(
                28,
                [
                    Placement(0, 0, 10),
                    Placement(2, 10, 28),
                ],
            )


if __name__ == "__main__":
    unittest.main()
