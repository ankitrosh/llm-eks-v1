"""Two-stage contiguous Qwen3 partition selection.

split_at is the first transformer layer assigned to rank 1.
"""

def stage_bounds(num_layers, rank, split_at=None):
    if type(num_layers) is not int or num_layers < 2:
        raise ValueError("num_layers must be at least 2")

    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")

    if split_at is None:
        split_at = num_layers // 2

    if (
        type(split_at) is not int
        or not 1 <= split_at < num_layers
    ):
        raise ValueError(
            "split_at must be between 1 and num_layers - 1"
        )

    return (
        (0, split_at)
        if rank == 0
        else (split_at, num_layers)
    )
