"""Qwen3 layer ownership for contiguous N-rank pipelines.

No PyTorch dependency. Used for manifest validation,
memory planning, and stage-only checkpoint loading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3NRankLayout:
    num_layers: int
    boundaries: tuple[int, ...]

    def __post_init__(self):
        if (
            type(self.num_layers) is not int
            or self.num_layers < 2
        ):
            raise ValueError("num_layers must be >= 2")

        try:
            bounds = tuple(self.boundaries)
        except TypeError as exc:
            raise ValueError(
                "boundaries must be a sequence"
            ) from exc

        if (
            len(bounds) < 3
            or len(bounds) - 1 > self.num_layers
        ):
            raise ValueError(
                "Expected 2..num_layers ranks"
            )

        if any(type(v) is not int for v in bounds):
            raise ValueError(
                "Boundary values must be integers"
            )

        if (
            bounds[0] != 0
            or bounds[-1] != self.num_layers
        ):
            raise ValueError(
                "Boundaries must cover every layer"
            )

        if any(
            a >= b
            for a, b in zip(bounds, bounds[1:])
        ):
            raise ValueError(
                "Every rank must own a positive layer range"
            )

        object.__setattr__(
            self, "boundaries", bounds
        )

    @property
    def world_size(self):
        return len(self.boundaries) - 1

    def rank_bounds(self, rank):
        if (
            type(rank) is not int
            or not 0 <= rank < self.world_size
        ):
            raise ValueError(
                "Rank outside partition"
            )

        return (
            self.boundaries[rank],
            self.boundaries[rank + 1],
        )

    def owned_layers(self, rank):
        start, end = self.rank_bounds(rank)
        return tuple(range(start, end))

    def stage_role(self, rank):
        self.rank_bounds(rank)

        return {
            "embeddings": rank == 0,
            "norm_and_lm_head": (
                rank == self.world_size - 1
            ),
        }

    @classmethod
    def from_counts(cls, counts):
        try:
            counts = tuple(counts)
        except TypeError as exc:
            raise ValueError(
                "counts must be a sequence"
            ) from exc

        if (
            len(counts) < 2
            or any(
                type(x) is not int or x < 1
                for x in counts
            )
        ):
            raise ValueError(
                "Need at least two positive "
                "integer layer counts"
            )

        boundaries = [0]

        for count in counts:
            boundaries.append(
                boundaries[-1] + count
            )

        return cls(
            boundaries[-1],
            tuple(boundaries),
        )

    @classmethod
    def from_placements(cls, num_layers, placements):
        try:
            placements = tuple(placements)
        except TypeError as exc:
            raise ValueError(
                "placements must be a sequence"
            ) from exc

        if len(placements) < 2:
            raise ValueError(
                "Expected at least two stage placements"
            )

        boundaries = [0]

        for rank, placement in enumerate(placements):
            if (
                type(placement.rank) is not int
                or placement.rank != rank
                or type(placement.start) is not int
                or type(placement.end) is not int
                or placement.start != boundaries[-1]
                or placement.end <= placement.start
            ):
                raise ValueError(
                    "Noncontiguous or unordered placements"
                )

            boundaries.append(placement.end)

        return cls(
            num_layers,
            tuple(boundaries),
        )
