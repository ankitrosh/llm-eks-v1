import pytest

from llm_runtime.v3_2_batch_planner import (
    ContinuousBatchPlanner,
)
from llm_runtime.v3_2_policy_api import (
    BatchDecision,
    load_batch_policy,
)


def config():
    return {
        "pipeline": {
            "memory": {
                "max_active_requests": 2,
                "max_total_tokens_per_request": 16,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "max_requests_per_batch": 1,
            "max_batch_tokens": 4,
            "max_queue_requests": 2,
        },
    }


def test_import_user_policy():
    policy = load_batch_policy(
        "examples.v3_2_custom_batch_policy:ShortPrefillFirst"
    )
    assert callable(policy.select_batch)


def test_custom_policy_reorders_requests():
    class NewestFirst:
        def select_batch(self, context):
            candidate = context.candidates[-1]
            return BatchDecision(
                candidate.phase,
                (candidate.request_id,),
            )

    planner = ContinuousBatchPlanner(
        config(), batch_policy=NewestFirst()
    )

    planner.submit(1, [1], 1)
    planner.submit(2, [2], 1)

    plan = planner.next_batch()
    assert plan.items[0].request_id == 2

    update = planner.complete_batch({2: 5})
    assert update.finished == (2,)

    plan = planner.next_batch()
    assert plan.items[0].request_id == 1


@pytest.mark.parametrize("ids", [
    (),
    (999,),
    (1, 1),
    (1, 2),
    ("1",),
])
def test_unsafe_policy_rejected(ids):
    class UnsafePolicy:
        def select_batch(self, context):
            return BatchDecision("prefill", ids)

    planner = ContinuousBatchPlanner(
        config(), batch_policy=UnsafePolicy()
    )
    planner.submit(1, [1], 1)

    with pytest.raises((ValueError, TypeError)):
        planner.next_batch()

    assert planner.snapshot()["in_flight"] == ()
    assert planner.snapshot()["running"] == (1,)


def test_wrong_phase_rejected():
    class WrongPhase:
        def select_batch(self, context):
            return BatchDecision("decode", (1,))

    planner = ContinuousBatchPlanner(
        config(), batch_policy=WrongPhase()
    )
    planner.submit(1, [1], 1)

    with pytest.raises(ValueError):
        planner.next_batch()


def test_configured_import_path_used():
    cfg = config()
    cfg["scheduler"]["batch_policy_adapter"] = (
        "examples.v3_2_custom_batch_policy:ShortPrefillFirst"
    )

    planner = ContinuousBatchPlanner(cfg)

    planner.submit(1, [1, 2, 3], 1)
    planner.submit(2, [4], 1)

    plan = planner.next_batch()

    assert plan.phase == "prefill"
    assert plan.items[0].request_id == 2


def test_invalid_import_path_rejected():
    with pytest.raises(ValueError):
        load_batch_policy("invalid-path")
