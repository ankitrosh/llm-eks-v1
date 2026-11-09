from threading import Event, Lock

import pytest

from llm_runtime.v3_2_microbatch_transport import (
    CPUMicrobatchTransport,
    MicrobatchBackpressure,
    MicrobatchPipelineError,
)


def test_ordered_pipeline_has_real_stage_overlap():
    stage1_entered = Event()
    stage0_second_ran = Event()

    trace = []
    guard = Lock()

    def stage0(packet):
        with guard:
            trace.append((0, packet.sequence))

        if packet.sequence == 2:
            assert stage1_entered.wait(2), (
                "Stage 1 never started the first batch"
            )
            stage0_second_ran.set()

        return packet.payload + 1

    def stage1(packet):
        with guard:
            trace.append((1, packet.sequence))

        if packet.sequence == 1:
            stage1_entered.set()

            assert stage0_second_ran.wait(2), (
                "No pipeline-stage overlap"
            )

        return packet.payload * 2

    with CPUMicrobatchTransport(
        (stage0, stage1),
        max_pending=2,
    ) as pipeline:
        first = pipeline.submit(
            (1, 2), (3, 5), 10
        )
        second = pipeline.submit(
            (3,), (7,), 20
        )

        assert pipeline.wait(first) == 22
        assert pipeline.wait(second) == 42

        state = pipeline.snapshot()

        assert state["stage_calls"] == (2, 2)
        assert state["reserved_request_ids"] == ()

    assert stage1_entered.is_set()

    assert trace.index((0, 1)) < trace.index((0, 2))
    assert trace.index((1, 1)) < trace.index((1, 2))


def test_backpressure_and_reuse():
    gate = Event()

    def slow(packet):
        assert gate.wait(2)
        return packet.payload

    with CPUMicrobatchTransport(
        (slow, lambda packet: packet.payload),
        max_pending=1,
    ) as pipeline:
        first = pipeline.submit(
            (1,), (3,), "ok"
        )

        with pytest.raises(MicrobatchBackpressure):
            pipeline.submit(
                (2,), (3,), "blocked"
            )

        with pytest.raises(TimeoutError):
            pipeline.wait(first, timeout=0.001)

        gate.set()
        assert pipeline.wait(first) == "ok"

        second = pipeline.submit(
            (1,), (4,), "again"
        )
        assert pipeline.wait(second) == "again"


def test_duplicate_active_request():
    gate = Event()

    def slow(packet):
        gate.wait(2)
        return packet.payload

    with CPUMicrobatchTransport(
        (slow, lambda packet: packet.payload),
        max_pending=2,
    ) as pipeline:
        first = pipeline.submit(
            (1, 2), (3, 4), 10
        )

        with pytest.raises(ValueError, match="already"):
            pipeline.submit(
                (2, 3), (4, 5), 20
            )

        gate.set()
        assert pipeline.wait(first) == 10


@pytest.mark.parametrize(
    "ids, positions",
    [
        ((1, 1), (3, 3)),
        ((1,), (0,)),
        ((1,), (1, 2)),
        ((True,), (1,)),
        ((), ()),
    ],
)
def test_invalid_envelopes(ids, positions):
    with CPUMicrobatchTransport(
        (
            lambda packet: packet.payload,
            lambda packet: packet.payload,
        )
    ) as pipeline:
        with pytest.raises(ValueError):
            pipeline.submit(ids, positions, 0)


def test_stage_failure_poison_and_unblock():
    def failing_stage(packet):
        if packet.sequence == 1:
            raise ValueError(
                "Stage KV position mismatch"
            )
        return packet.payload

    with CPUMicrobatchTransport(
        (
            failing_stage,
            lambda packet: packet.payload,
        ),
        max_pending=2,
    ) as pipeline:
        first = pipeline.submit(
            (1,), (3,), 1
        )
        second = pipeline.submit(
            (2,), (5,), 2
        )

        with pytest.raises(MicrobatchPipelineError):
            pipeline.wait(first)

        with pytest.raises(MicrobatchPipelineError):
            pipeline.wait(second)

        assert pipeline.snapshot()["failed"]

        with pytest.raises(MicrobatchPipelineError):
            pipeline.submit(
                (3,), (1,), 3
            )


def test_close_unblocks_waiter():
    gate = Event()

    def slow(packet):
        gate.wait(2)
        return packet.payload

    pipeline = CPUMicrobatchTransport(
        (
            slow,
            lambda packet: packet.payload,
        )
    )

    sequence = pipeline.submit(
        (1,), (1,), 0
    )

    pipeline.close()
    gate.set()

    with pytest.raises(MicrobatchPipelineError):
        pipeline.wait(sequence)
