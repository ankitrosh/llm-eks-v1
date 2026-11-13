"""Real asynchronous CPU/Gloo microbatch tests."""

from datetime import timedelta
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm_runtime.v3_2_async_gloo import (
    AsyncGlooBackpressure,
    AsyncGlooCoordinator,
    run_async_gloo_stage,
)
from llm_runtime.v3_2_gloo_wire import WireMicrobatch


def worker(
    rank,
    world_size,
    rendezvous,
    rank1_first,
    rank0_second,
    rank2_first,
    rank1_second,
):
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=25),
    )

    try:
        if rank == 0:
            with AsyncGlooCoordinator(
                world_size=world_size,
                max_pending=2,
            ) as coordinator:

                first = WireMicrobatch(
                    1,
                    "prefill",
                    (1, 2),
                    (0, 0),
                    (2, 3),
                    torch.arange(
                        24, dtype=torch.float32
                    ).reshape(2, 3, 4),
                )

                coordinator.submit(first)

                assert rank1_first.wait(10), (
                    "Rank 1 did not receive first batch"
                )

                # Rank 0 progresses while Rank 1 is busy.
                rank0_second.set()

                second = WireMicrobatch(
                    2,
                    "decode",
                    (3,),
                    (5,),
                    (1,),
                    torch.ones(1, 1, 4),
                )

                coordinator.submit(second)

                # Both slots are occupied.
                with pytest.raises(
                    AsyncGlooBackpressure
                ):
                    coordinator.submit(
                        WireMicrobatch(
                            3,
                            "decode",
                            (4,),
                            (5,),
                            (1,),
                            torch.ones(1, 1, 4),
                        )
                    )

                # Results can be acknowledged in any order.
                result2 = coordinator.wait(2, timeout=15)

                # Request 1 is still reserved because its
                # result has not been acknowledged.
                with pytest.raises(ValueError):
                    coordinator.submit(
                        WireMicrobatch(
                            3,
                            "decode",
                            (1,),
                            (5,),
                            (1,),
                            torch.ones(1, 1, 4),
                        )
                    )

                result1 = coordinator.wait(1, timeout=15)

                for original, result in (
                    (first, result1),
                    (second, result2),
                ):
                    expected = torch.stack([
                        original.payload[i, length - 1]
                        for i, length
                        in enumerate(original.lengths)
                    ])

                    expected += sum(range(1, world_size))

                    torch.testing.assert_close(
                        result.payload,
                        expected,
                    )

                # Once acknowledged, request 1 can be
                # scheduled again with a new position.
                third = WireMicrobatch(
                    3,
                    "decode",
                    (1,),
                    (7,),
                    (1,),
                    torch.ones(1, 1, 4) * 3,
                )

                coordinator.submit(third)

                result3 = coordinator.wait(3, timeout=15)

                torch.testing.assert_close(
                    result3.payload,
                    torch.full(
                        (1, 4),
                        3 + sum(range(1, world_size)),
                        dtype=torch.float32,
                    ),
                )

                state = coordinator.snapshot()

                assert state["received"] == 3
                assert state["sent"] == 3
                assert state["pending_sequences"] == ()
                assert state["reserved_request_ids"] == ()
                assert not state["failed"]

        else:
            def execute(packet):
                if rank == 1 and packet.sequence == 1:
                    rank1_first.set()

                    assert rank0_second.wait(10), (
                        "Rank 0 did not progress while "
                        "Rank 1 was executing"
                    )

                if rank == 2 and packet.sequence == 1:
                    rank2_first.set()

                    assert rank1_second.wait(10), (
                        "Rank 1 did not overlap Rank 2"
                    )

                if (
                    rank == 1
                    and packet.sequence == 2
                    and world_size >= 3
                ):
                    assert rank2_first.wait(10), (
                        "Rank 2 did not start first batch"
                    )

                    rank1_second.set()

                if rank + 1 < world_size:
                    return packet.payload + rank

                return torch.stack([
                    packet.payload[i, length - 1]
                    for i, length
                    in enumerate(packet.lengths)
                ]) + rank

            run_async_gloo_stage(
                rank,
                world_size,
                execute,
            )

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_real_overlap_and_acknowledgement(tmp_path, ranks):
    context = get_context("spawn")

    rank1_first = context.Event()
    rank0_second = context.Event()
    rank2_first = context.Event()
    rank1_second = context.Event()

    mp.spawn(
        worker,
        args=(
            ranks,
            str(tmp_path / "gloo.init"),
            rank1_first,
            rank0_second,
            rank2_first,
            rank1_second,
        ),
        nprocs=ranks,
        join=True,
    )

    assert rank1_first.is_set()
    assert rank0_second.is_set()

    if ranks >= 3:
        assert rank2_first.is_set()
        assert rank1_second.is_set()
