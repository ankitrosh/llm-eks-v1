"""GPU NCCL validation of the ordered two-rank transport."""

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from llm_runtime.v3_ordered_rank_transport import (
    OrderedRank0Transport,
    TransportFailure,
    open_control,
    run_rank1,
)


def main():
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2

    assert torch.cuda.is_available(), "CUDA required"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=120),
    )

    conn = open_control(
        rank,
        os.environ.get(
            "MASTER_ADDR",
            "127.0.0.1",
        ),
        int(os.environ.get(
            "V3_CTRL_PORT",
            "29619",
        )),
    )

    if rank == 1:

        def execute(rid, step, position, hidden, cache):
            if step == 0:
                assert cache is None
            else:
                assert cache == step

            time.sleep(0.04)

            return hidden + 1, step + 1

        stats = run_rank1(
            conn,
            device=device,
            execute=execute,
        )

        assert stats["runs"] == 12
        assert stats["releases"] == 4

        print(
            "PASS: Rank 1 processed 12 stage calls",
            flush=True,
        )

        print(
            "PASS: Rank 1 released four request caches",
            flush=True,
        )

    else:
        from llm_runtime.v3_pipeline_scheduler import (
            FatalStageError,
            PipelineScheduler,
            StageResult,
        )

        link = OrderedRank0Transport(
            conn,
            device=device,
            max_pending=4,
        )

        def first(task, cache):
            x = (
                task.payload[-1]
                if isinstance(task.payload, tuple)
                else task.payload
            )

            time.sleep(0.04)

            tensor = torch.tensor(
                [[[2.0 * x]]],
                dtype=torch.float32,
                device=device,
            )

            try:
                ticket = link.publish(
                    task.request_id,
                    task.step,
                    task.position,
                    tensor,
                )
            except TransportFailure as exc:
                raise FatalStageError(str(exc)) from exc

            return StageResult(
                ticket,
                (cache or 0) + 1,
            )

        def second(task, cache):
            ticket = task.payload

            assert ticket.request_id == task.request_id
            assert ticket.step == task.step

            try:
                output = link.await_result(ticket)
            except TransportFailure as exc:
                raise FatalStageError(str(exc)) from exc

            return StageResult(
                output,
                (cache or 0) + 1,
            )

        try:
            with PipelineScheduler(
                [first, second],
                lambda tensor: int(
                    tensor[0, 0, 0].item()
                ),
                max_active_requests=4,
                on_request_end=link.release_request,
            ) as scheduler:

                futures = [
                    scheduler.submit(
                        (1, n),
                        prompt_length=2,
                        max_new_tokens=3,
                    )
                    for n in (2, 4, 6, 8)
                ]

                results = [
                    f.result(timeout=30).tokens
                    for f in futures
                ]

                assert results == [
                    (5, 11, 23),
                    (9, 19, 39),
                    (13, 27, 55),
                    (17, 35, 71),
                ], results

                print(
                    "PASS: Four concurrent requests "
                    "match reference arithmetic",
                    flush=True,
                )

                events = scheduler.snapshot()["events"]

                starts = {
                    (e.stage, e.request_id, e.step): e.at
                    for e in events
                    if e.event == "start"
                }

                finishes = {
                    (e.stage, e.request_id, e.step): e.at
                    for e in events
                    if e.event == "finish"
                }

                assert (
                    starts[(0, 2, 0)]
                    < finishes[(1, 1, 0)]
                )

                assert (
                    starts[(1, 1, 0)]
                    < finishes[(0, 2, 0)]
                )

                print(
                    "PASS: Scheduler stage overlap",
                    flush=True,
                )

                assert link.stats()["sent"] == 12
                assert link.stats()["received"] == 12

                print(
                    "PASS: 12 ordered tensor transfers",
                    flush=True,
                )

        finally:
            link.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
