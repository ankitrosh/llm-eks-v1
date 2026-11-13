"""Real CPU/Gloo distributed KV-release lifecycle tests."""

from datetime import timedelta
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm_runtime.v3_2_async_gloo import (
    AsyncGlooCoordinator,
    run_async_gloo_stage,
)
from llm_runtime.v3_2_gloo_wire import (
    WireMicrobatch,
)


def release_worker(
    rank,
    world_size,
    rendezvous,
    release_entered,
    release_allowed,
):
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
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
                    (1,),
                    (0,),
                    (3,),
                    torch.arange(
                        12, dtype=torch.float32
                    ).reshape(1, 3, 4),
                )

                coordinator.submit(first)

                # In-flight inference prevents KV release.
                with pytest.raises(ValueError):
                    coordinator.submit_release((1,))

                response = coordinator.wait(1)

                expected = (
                    first.payload[:, -1, :]
                    + sum(range(1, world_size))
                )

                torch.testing.assert_close(
                    response.payload,
                    expected,
                )

                sequence = coordinator.submit_release(
                    (1,)
                )

                assert release_entered.wait(10)

                # Final rank is blocked inside its release.
                # An ACK must not be returned yet.
                with pytest.raises(TimeoutError):
                    coordinator.wait_release(
                        sequence,
                        timeout=0.1,
                    )

                # Release ownership remains reserved.
                with pytest.raises(
                    ValueError, match="release"
                ):
                    coordinator.submit(
                        WireMicrobatch(
                            2,
                            "decode",
                            (1,),
                            (3,),
                            (1,),
                            torch.ones(1, 1, 4),
                        )
                    )

                release_allowed.set()

                assert coordinator.wait_release(
                    sequence
                ) == (1,)

                # The same request ID is now eligible
                # for another independent KV lifecycle.
                coordinator.submit(
                    WireMicrobatch(
                        2,
                        "prefill",
                        (1,),
                        (0,),
                        (2,),
                        torch.ones(1, 2, 4),
                    )
                )

                coordinator.wait(2)

                second_release = (
                    coordinator.submit_release((1,))
                )

                assert coordinator.wait_release(
                    second_release
                ) == (1,)

                state = coordinator.snapshot()

                assert state["pending_sequences"] == ()
                assert state[
                    "pending_release_sequences"
                ] == ()
                assert state[
                    "releasing_request_ids"
                ] == ()
                assert not state["failed"]

        else:
            resident = set()
            released = []

            def execute(packet):
                for rid in packet.request_ids:
                    assert rid not in resident
                    resident.add(rid)

                transformed = packet.payload + rank

                if rank + 1 < world_size:
                    return transformed

                return torch.stack([
                    transformed[i, n - 1]
                    for i, n in enumerate(
                        packet.lengths
                    )
                ])

            def release(ids):
                assert set(ids).issubset(resident)

                if (
                    rank == world_size - 1
                    and not released
                ):
                    release_entered.set()
                    assert release_allowed.wait(10)

                resident.difference_update(ids)
                released.append(tuple(ids))

            run_async_gloo_stage(
                rank,
                world_size,
                execute,
                release=release,
            )

            assert released == [(1,), (1,)]
            assert not resident

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_distributed_release_ack_and_reuse(
    tmp_path, ranks
):
    context = get_context("spawn")

    release_entered = context.Event()
    release_allowed = context.Event()

    mp.spawn(
        release_worker,
        args=(
            ranks,
            str(tmp_path / "gloo.init"),
            release_entered,
            release_allowed,
        ),
        nprocs=ranks,
        join=True,
    )

    assert release_entered.is_set()
    assert release_allowed.is_set()
