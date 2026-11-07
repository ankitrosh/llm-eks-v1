"""Optional utilities for user-defined model partitioning.

All ranges are [start, end) and use zero-based layer indices.
"""

from __future__ import annotations


def contiguous_stage_ranges(total_layers, layer_counts):
    if type(total_layers) is not int or total_layers < 1:
        raise ValueError("total_layers must be positive")

    if (
        not isinstance(layer_counts, (list, tuple))
        or not layer_counts
        or any(
            type(count) is not int or count < 1
            for count in layer_counts
        )
    ):
        raise ValueError(
            "layer_counts must contain positive integers"
        )

    if sum(layer_counts) != total_layers:
        raise ValueError(
            "Partition must cover every layer exactly once"
        )

    ranges = []
    start = 0

    for count in layer_counts:
        end = start + count
        ranges.append((start, end))
        start = end

    return tuple(ranges)


def stage_bounds(total_layers, layer_counts, rank):
    ranges = contiguous_stage_ranges(
        total_layers,
        layer_counts,
    )

    if type(rank) is not int or not 0 <= rank < len(ranges):
        raise ValueError("Invalid stage rank")

    return ranges[rank]
