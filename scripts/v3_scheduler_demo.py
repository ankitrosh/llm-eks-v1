"""Demonstrate forward-only request pipelining."""

from time import sleep

from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    StageResult,
)


def stage0(task, cache):
    sleep(0.04)

    x = (
        task.payload[-1]
        if isinstance(task.payload, tuple)
        else task.payload
    )

    return StageResult(
        2 * x,
        (cache or 0) + 1,
    )


def stage1(task, cache):
    sleep(0.04)

    return StageResult(
        task.payload + 1,
        (cache or 0) + 1,
    )


with PipelineScheduler(
    [stage0, stage1],
    lambda x: x,
    max_active_requests=3,
) as pipeline:

    handles = [
        pipeline.submit(
            (1, n),
            prompt_length=2,
            max_new_tokens=3,
        )
        for n in (2, 4, 6)
    ]

    for handle in handles:
        result = handle.result(timeout=10)

        print(
            f"Request {result.request_id}: "
            f"{result.tokens}"
        )

    events = pipeline.snapshot()["events"]


start = events[0].at

print("\nStage timeline (seconds from start):")

for event in events:
    if event.event == "start":
        print(
            f"{event.at - start:6.3f} "
            f"Stage {event.stage} "
            f"Request {event.request_id} "
            f"forward {event.step}"
        )
