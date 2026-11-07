"""V3.1 CPU-only, memory-aware contiguous layer partition planner."""

from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 1024 ** 3


@dataclass(frozen=True)
class StagePlacement:
    rank: int
    start: int
    end: int
    gpu_memory_gib: float
    budget_gib: float
    estimated_gib: float

    @property
    def layer_count(self):
        return self.end - self.start

    @property
    def budget_fraction(self):
        return self.estimated_gib / self.budget_gib


def _numbers(values, name, positive=True):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc

    if not values:
        raise ValueError(f"{name} cannot be empty")

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(f"Invalid {name}: {value}")

    return tuple(float(value) for value in values)


def plan_contiguous_layers(
    layer_weights_gib,
    gpu_memory_gib,
    *,
    gpu_reserve_gib=2.0,
    kv_bytes_per_layer_per_token=0,
    max_context_tokens=1,
    max_active_requests=1,
    first_stage_gib=0.0,
    last_stage_gib=0.0,
    manual_layer_counts=None,
):
    """Return one contiguous layer placement per GPU rank.

    Inputs are memory estimates, not measured CUDA allocations.
    No guarantee of actual GPU fit or throughput is implied.
    """

    weights = _numbers(layer_weights_gib, "layer_weights_gib")
    gpu = _numbers(gpu_memory_gib, "gpu_memory_gib")

    n = len(weights)
    ranks = len(gpu)

    if ranks > n:
        raise ValueError("More ranks than transformer layers")

    for name, value in (
        ("max_context_tokens", max_context_tokens),
        ("max_active_requests", max_active_requests),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive")

    if (
        type(kv_bytes_per_layer_per_token) is not int
        or kv_bytes_per_layer_per_token < 0
    ):
        raise ValueError("Invalid KV bytes per token")

    reserve = _numbers(
        [gpu_reserve_gib], "gpu_reserve_gib", False
    )[0]
    first = _numbers(
        [first_stage_gib], "first_stage_gib", False
    )[0]
    last = _numbers(
        [last_stage_gib], "last_stage_gib", False
    )[0]

    budgets = [memory - reserve for memory in gpu]

    if any(budget <= 0 for budget in budgets):
        raise ValueError("Reserve exceeds available memory")

    kv_per_layer_gib = (
        kv_bytes_per_layer_per_token
        * max_context_tokens
        * max_active_requests
        / GIB
    )

    prefix = [0.0]

    for weight in weights:
        prefix.append(
            prefix[-1] + weight + kv_per_layer_gib
        )

    def used(rank, start, end):
        return (
            prefix[end] - prefix[start]
            + (first if rank == 0 else 0.0)
            + (last if rank == ranks - 1 else 0.0)
        )

    if manual_layer_counts is not None:
        counts = tuple(manual_layer_counts)

        if (
            len(counts) != ranks
            or any(type(c) is not int or c < 1 for c in counts)
            or sum(counts) != n
        ):
            raise ValueError("Invalid manual layer counts")

        boundaries = [0]
        for count in counts:
            boundaries.append(boundaries[-1] + count)

    else:
        # Minimax dynamic programming over contiguous partitions.
        # dp[r][i] is the best assignment of i layers to r ranks.
        dp = [
            [None] * (n + 1)
            for _ in range(ranks + 1)
        ]
        dp[0][0] = ((0.0, 0), -1)

        for rank in range(ranks):
            for start in range(n + 1):
                previous = dp[rank][start]

                if previous is None:
                    continue

                remaining = ranks - rank - 1
                max_end = n - remaining

                for end in range(start + 1, max_end + 1):
                    required = used(rank, start, end)

                    if required > budgets[rank] + 1e-9:
                        continue

                    score = (
                        max(
                            previous[0][0],
                            required / budgets[rank],
                        ),
                        previous[0][1] + (end - start) ** 2,
                    )

                    current = dp[rank + 1][end]

                    if current is None or score < current[0]:
                        dp[rank + 1][end] = (score, start)

        if dp[ranks][n] is None:
            raise ValueError(
                "No estimated memory-feasible partition"
            )

        boundaries = [n]
        end = n

        for rank in range(ranks, 0, -1):
            start = dp[rank][end][1]
            boundaries.append(start)
            end = start

        boundaries.reverse()

    placements = []

    for rank in range(ranks):
        start = boundaries[rank]
        end = boundaries[rank + 1]
        estimate = used(rank, start, end)

        if estimate > budgets[rank] + 1e-9:
            raise ValueError(
                f"Rank {rank} exceeds memory budget"
            )

        placements.append(
            StagePlacement(
                rank=rank,
                start=start,
                end=end,
                gpu_memory_gib=gpu[rank],
                budget_gib=budgets[rank],
                estimated_gib=estimate,
            )
        )

    return tuple(placements)
