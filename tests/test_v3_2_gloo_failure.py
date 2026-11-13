"""Fail-fast Gloo stage execution and KV release."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm_runtime.v3_2_async_gloo import (
    AsyncGlooCoordinator,
    AsyncGlooFailure,
    run_async_gloo_stage,
)
from llm_runtime.v3_2_gloo_wire import WireMicrobatch


def worker(rank, world_size, rendezvous, mode):
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=35),
    )

    try:
        if rank == 0:
            with AsyncGlooCoordinator(
                world_size=world_size,
                max_pending=2,
            ) as coordinator:

                coordinator.submit(WireMicrobatch(
                    1, "prefill", (1,), (0,), (2,),
                    torch.ones(1, 2, 4),
                ))

                if mode == "execution":
                    coordinator.submit(WireMicrobatch(
                        2, "prefill", (2,), (0,), (2,),
                        torch.ones(1, 2, 4),
                    ))

                    for sequence in (1, 2):
                        with pytest.raises(
                            AsyncGlooFailure
                        ):
                            coordinator.wait(
                                sequence, timeout=12
                            )

                else:
                    result = coordinator.wait(
                        1, timeout=12
                    )
                    assert result.sequence == 1

                    sequence = (
                        coordinator.submit_release((1,))
                    )

                    with pytest.raises(
                        AsyncGlooFailure
                    ):
                        coordinator.wait_release(
                            sequence, timeout=12
                        )

                assert coordinator.snapshot()["failed"]

                with pytest.raises(AsyncGlooFailure):
                    coordinator.submit(WireMicrobatch(
                        3 if mode == "execution" else 2,
                        "prefill", (5,), (0,), (1,),
                        torch.ones(1, 1, 4),
                    ))

        else:
            resident = set()

            def execute(packet):
                if (
                    mode == "execution"
                    and rank == 1
                    and packet.sequence == 1
                ):
                    raise RuntimeError(
                        "Injected stage execution failure"
                    )

                resident.update(packet.request_ids)

                output = packet.payload + rank

                if rank + 1 < world_size:
                    return output

                return output[:, -1, :].contiguous()

            def release(ids):
                if (
                    mode == "release"
                    and rank == world_size - 1
                ):
                    raise RuntimeError(
                        "Injected KV release failure"
                    )

                for rid in ids:
                    resident.remove(rid)

            run_async_gloo_stage(
                rank,
                world_size,
                execute,
                release=release,
            )

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3, 4])
@pytest.mark.parametrize(
    "mode", ["execution", "release"]
)
def test_remote_failure_propagation(
    tmp_path, ranks, mode
):
    mp.spawn(
        worker,
        args=(
            ranks,
            str(tmp_path / "gloo.init"),
            mode,
        ),
        nprocs=ranks,
        join=True,
    )
