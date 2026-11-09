import pytest

from llm_runtime.v3_2_pipeline_policy import (
    PipelineDispatchController,
    PipelineDecision,
    PipelineBusy,
    load_pipeline_policy,
)


def settings(policy="downstream_first", inflight=2, queued=3):
    return {"pipeline": {"scheduling": {
        "policy": policy,
        "max_inflight_microbatches": inflight,
        "max_queued_microbatches": queued,
    }}}


def test_pipeline_overlap_and_forward_only():
    ctl = PipelineDispatchController(settings(), stages=3)
    ctl.enqueue(1, (7, 8), 4)
    ctl.enqueue(2, (9,), 2)

    assert ctl.dispatch().microbatch_id == 1
    assert ctl.dispatch() is None

    assert ctl.complete(0, 1) is False
    assert ctl.dispatch().microbatch_id == 1
    assert ctl.dispatch().microbatch_id == 2

    assert ctl.snapshot()["busy_by_stage"] == (2, 1, None)

    assert ctl.complete(1, 1) is False
    assert ctl.dispatch().stage == 2
    assert ctl.complete(2, 1) is True

    assert ctl.complete(0, 2) is False
    assert ctl.dispatch().microbatch_id == 2
    assert ctl.complete(1, 2) is False
    assert ctl.dispatch().microbatch_id == 2
    assert ctl.complete(2, 2) is True

    assert ctl.snapshot()["completed"] == (1, 2)


def test_inflight_limit():
    ctl = PipelineDispatchController(
        settings(inflight=1), stages=2
    )
    ctl.enqueue(1, (1,), 1)
    ctl.enqueue(2, (2,), 1)

    ctl.dispatch()
    ctl.complete(0, 1)

    assert ctl.dispatch().microbatch_id == 1
    assert ctl.dispatch() is None

    ctl.complete(1, 1)
    assert ctl.dispatch().microbatch_id == 2


def test_queue_capacity():
    ctl = PipelineDispatchController(
        settings(queued=1), stages=2
    )
    ctl.enqueue(1, (1,), 1)

    with pytest.raises(PipelineBusy):
        ctl.enqueue(2, (2,), 1)

    assert ctl.snapshot()["ready_by_stage"][0] == (1,)


def test_custom_policy():
    class StageZeroFirst:
        def select_pipeline(self, context):
            c = min(context.candidates, key=lambda x: x.stage)
            return PipelineDecision(c.microbatch_id, c.stage)

    ctl = PipelineDispatchController(
        settings(), stages=2, policy=StageZeroFirst()
    )
    ctl.enqueue(1, (1,), 1)
    ctl.enqueue(2, (2,), 1)

    ctl.dispatch()
    ctl.complete(0, 1)

    assert ctl.dispatch().microbatch_id == 2


def test_invalid_policy_decision_is_atomic():
    class Wrong:
        def select_pipeline(self, context):
            return PipelineDecision(999, 1)

    ctl = PipelineDispatchController(
        settings(), stages=2, policy=Wrong()
    )
    ctl.enqueue(1, (1,), 1)

    with pytest.raises(ValueError):
        ctl.dispatch()

    assert ctl.snapshot()["ready_by_stage"] == ((1,), ())
    assert ctl.snapshot()["busy_by_stage"] == (None, None)


def test_invalid_completion():
    ctl = PipelineDispatchController(settings(), stages=2)
    ctl.enqueue(1, (1,), 2)

    with pytest.raises(ValueError):
        ctl.enqueue(1, (1,), 2)

    ctl.dispatch()

    with pytest.raises(ValueError):
        ctl.complete(0, 2)

    with pytest.raises(ValueError):
        ctl.complete(1, 1)

    assert ctl.snapshot()["busy_by_stage"] == (1, None)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        PipelineDispatchController(
            settings(inflight=0), stages=2
        )

    with pytest.raises(ValueError):
        PipelineDispatchController(settings(), stages=0)


def test_invalid_adapter():
    with pytest.raises(ValueError):
        load_pipeline_policy("my-file.py:Class")

    with pytest.raises(TypeError):
        load_pipeline_policy(object())


def test_request_cannot_overlap_unfinished_batches():
    ctl = PipelineDispatchController(settings(), stages=2)
    ctl.enqueue(1, (1,), 1)

    with pytest.raises(ValueError):
        ctl.enqueue(2, (1,), 1)

    ctl.dispatch()
    ctl.complete(0, 1)
    ctl.dispatch()
    ctl.complete(1, 1)

    ctl.enqueue(2, (1,), 1)


def test_rank_count_mismatch():
    cfg = settings()
    cfg["pipeline"]["num_ranks"] = 3

    with pytest.raises(ValueError):
        PipelineDispatchController(cfg, stages=2)


def test_import_user_adapter():
    policy = load_pipeline_policy(
        "examples.v3_2_custom_pipeline_policy:PreferStageZero"
    )
    assert callable(policy.select_pipeline)
