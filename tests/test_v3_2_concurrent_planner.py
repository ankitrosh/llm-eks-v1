import pytest

from llm_runtime.v3_2_concurrent_planner import (
    ConcurrentBatchPlanner,
)


def config():
    return {
        "pipeline": {
            "memory": {
                "max_active_requests": 3,
                "max_total_tokens_per_request": 32,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "policy": "fcfs",
            "max_requests_per_batch": 2,
            "max_batch_tokens": 32,
            "max_queue_requests": 4,
        },
    }


def test_out_of_order_completion_and_no_double_scheduling():
    planner = ConcurrentBatchPlanner(config())

    planner.submit(1, (2, 3), max_new_tokens=2)
    planner.submit(2, (4, 5), max_new_tokens=2)

    first = planner.next_concurrent_batch(
        phase="prefill", max_requests=1
    )
    second = planner.next_concurrent_batch(
        phase="prefill", max_requests=1
    )

    assert {first.plan.items[0].request_id,
            second.plan.items[0].request_id} == {1, 2}

    assert planner.next_concurrent_batch() is None

    # Complete the second batch first.
    update = planner.complete_concurrent_batch(
        second.batch_id, {2: 9}
    )
    assert update.emitted == {2: 9}

    # Request 1 is still reserved. Request 2 can decode.
    third = planner.next_concurrent_batch(
        phase="decode"
    )
    assert tuple(
        x.request_id for x in third.plan.items
    ) == (2,)

    planner.complete_concurrent_batch(
        first.batch_id, {1: 8}
    )

    planner.complete_concurrent_batch(
        third.batch_id,
        {2: 10},
        stop_ids=(2,),
    )

    assert 2 in planner.drain_releases()

    fourth = planner.next_concurrent_batch(
        phase="decode"
    )
    assert fourth.plan.items[0].request_id == 1

    update = planner.complete_concurrent_batch(
        fourth.batch_id, {1: 11}
    )

    assert update.finished == (1,)
    assert planner.outstanding_batch_ids == ()


def test_inflight_cancel_waits_for_completion():
    planner = ConcurrentBatchPlanner(config())
    planner.submit(1, (2, 3), max_new_tokens=4)

    batch = planner.next_concurrent_batch(
        phase="prefill"
    )

    assert planner.cancel(1)
    assert planner.drain_releases() == ()

    update = planner.complete_concurrent_batch(
        batch.batch_id, {1: None}
    )

    assert update.cancelled == (1,)
    assert planner.drain_releases() == (1,)


def test_legacy_methods_are_disabled():
    planner = ConcurrentBatchPlanner(config())

    with pytest.raises(RuntimeError):
        planner.next_batch()

    with pytest.raises(RuntimeError):
        planner.complete_batch({})


def test_invalid_batch_completion_poison():
    planner = ConcurrentBatchPlanner(config())
    planner.submit(1, (2, 3), max_new_tokens=2)

    batch = planner.next_concurrent_batch(
        phase="prefill"
    )

    with pytest.raises(ValueError):
        planner.complete_concurrent_batch(
            batch.batch_id, {999: 10}
        )

    with pytest.raises(RuntimeError):
        planner.next_concurrent_batch()

    with pytest.raises(RuntimeError):
        planner.complete_concurrent_batch(
            batch.batch_id, {1: 10}
        )
