import pytest

from llm_runtime.v3_2_batch_planner import (
    AdmissionBusy,
    ContinuousBatchPlanner,
)


def config(active=2, batch=None, tokens=4,
           queue=2, policy="fcfs"):
    if batch is None:
        batch = min(active, 2)

    return {
        "pipeline": {
            "memory": {
                "max_active_requests": active,
                "max_total_tokens_per_request": 16,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "policy": policy,
            "max_requests_per_batch": batch,
            "max_batch_tokens": tokens,
            "max_queue_requests": queue,
            "max_wait_ms": 0,
        },
    }


def test_arrival_during_decoding():
    planner = ContinuousBatchPlanner(config())

    planner.submit(1, [1, 2], 3)
    assert planner.next_batch().phase == "prefill"
    planner.complete_batch({1: 10})

    planner.submit(2, [3, 4], 2)

    batch = planner.next_batch()
    assert batch.phase == "decode"
    assert batch.items[0].request_id == 1
    planner.complete_batch({1: 11})

    # The late arrival gets prefill before A finishes.
    batch = planner.next_batch()
    assert batch.phase == "prefill"
    assert batch.items[0].request_id == 2
    planner.complete_batch({2: 20})

    batch = planner.next_batch()
    assert batch.phase == "decode"
    assert {x.request_id for x in batch.items} == {1, 2}

    update = planner.complete_batch({1: 12, 2: 21})
    assert set(update.finished) == {1, 2}
    assert set(planner.drain_releases()) == {1, 2}


def test_round_robin_fairness():
    planner = ContinuousBatchPlanner(
        config(batch=1, policy="round_robin")
    )

    planner.submit(1, [1], 3)
    planner.submit(2, [2], 3)

    order = []

    for _ in range(4):
        batch = planner.next_batch()
        rid = batch.items[0].request_id
        order.append(rid)
        planner.complete_batch({rid: rid + 100})

    assert order == [1, 2, 1, 2]


def test_chunked_prefill():
    planner = ContinuousBatchPlanner(
        config(active=1, tokens=3)
    )

    planner.submit(1, [1, 2, 3, 4, 5, 6, 7], 2)

    for position in (0, 3, 6):
        batch = planner.next_batch()
        item = batch.items[0]

        assert item.position == position
        assert batch.token_count <= 3

        planner.complete_batch({
            1: 12 if item.emits_token else None
        })

    batch = planner.next_batch()
    assert batch.phase == "decode"
    assert batch.items[0].step == 3
    assert batch.items[0].position == 7

    outcome = planner.complete_batch({1: 13})

    assert outcome.finished == (1,)
    assert planner.drain_releases() == (1,)
    assert planner.snapshot()["reserved_kv_tokens"] == 0


def test_waiting_queue_and_admission():
    planner = ContinuousBatchPlanner(
        config(active=1, queue=1)
    )

    planner.submit(1, [1], 1)
    planner.submit(2, [2], 1)

    with pytest.raises(AdmissionBusy):
        planner.submit(3, [3], 1)

    assert planner.snapshot()["waiting"] == (2,)

    planner.next_batch()
    update = planner.complete_batch({1: 7})

    assert update.admitted == (2,)
    assert planner.snapshot()["running"] == (2,)


def test_cancel_waiting_request():
    planner = ContinuousBatchPlanner(
        config(active=1, queue=1)
    )

    planner.submit(1, [1], 2)
    planner.submit(2, [2], 2)

    assert planner.cancel(2)
    assert planner.drain_releases() == (2,)
    assert planner.snapshot()["waiting"] == ()


def test_cancel_inflight_defers_cache_release():
    planner = ContinuousBatchPlanner(
        config(active=1)
    )

    planner.submit(1, [1, 2], 1)
    planner.next_batch()

    assert planner.cancel(1)
    assert planner.drain_releases() == ()

    with pytest.raises(RuntimeError):
        planner.next_batch()

    outcome = planner.complete_batch({1: None})

    assert outcome.cancelled == (1,)
    assert planner.drain_releases() == (1,)
    assert planner.snapshot()["reserved_kv_tokens"] == 0


def test_kv_capacity_prevents_over_admission():
    planner = ContinuousBatchPlanner(
        config(active=3),
        kv_token_capacity=5,
    )

    planner.submit(1, [1, 2], 2)
    planner.submit(2, [1], 2)

    assert planner.snapshot()["running"] == (1,)
    assert planner.snapshot()["waiting"] == (2,)

    planner.cancel(1)

    assert planner.snapshot()["running"] == (2,)
    assert planner.snapshot()["reserved_kv_tokens"] == 3


def test_invalid_batch_completion_is_atomic():
    planner = ContinuousBatchPlanner(
        config(active=1)
    )

    planner.submit(1, [1], 2)
    planner.next_batch()

    with pytest.raises(ValueError):
        planner.complete_batch({1: None})

    assert planner.snapshot()["in_flight"] == (1,)

    assert planner.complete_batch({1: 5}).emitted == {1: 5}


def test_token_budget_and_request_ids():
    planner = ContinuousBatchPlanner(
        config(active=1),
        kv_token_capacity=4,
    )

    with pytest.raises(ValueError):
        planner.submit(1, [1, 2, 3, 4], 1)

    planner.submit(1, [1], 1)

    with pytest.raises(ValueError):
        planner.submit(1, [2], 1)


def test_partial_prefill_has_no_sample():
    planner = ContinuousBatchPlanner(
        config(active=1, tokens=2)
    )

    planner.submit(1, [1, 2, 3], 1)
    planner.next_batch()

    with pytest.raises(ValueError):
        planner.complete_batch({1: 42})

    assert planner.complete_batch({1: None}).emitted == {}

    assert planner.next_batch().items[0].position == 2


def test_eos_releases_kv():
    planner = ContinuousBatchPlanner(
        config(active=1)
    )

    planner.submit(1, [1], 5)
    planner.next_batch()

    update = planner.complete_batch(
        {1: 3}, stop_ids={1}
    )

    assert update.finished == (1,)
    assert planner.drain_releases() == (1,)
