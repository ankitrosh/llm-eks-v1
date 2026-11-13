from threading import Event

import pytest

from llm_runtime.v3_2_policy_transport import (
    PolicyCPUMicrobatchTransport,
    PolicyTransportBackpressure,
    PolicyTransportError,
)


def test_transport_has_stage_overlap():
    stage1_started = Event()
    stage0_second = Event()

    def stage0(packet):
        if packet.sequence == 2:
            assert stage1_started.wait(2)
            stage0_second.set()
        return packet.payload + 1

    def stage1(packet):
        if packet.sequence == 1:
            stage1_started.set()
            assert stage0_second.wait(2)
        return packet.payload * 2

    with PolicyCPUMicrobatchTransport(
        (stage0, stage1),
        max_pending=2,
    ) as tr:
        a = tr.submit((1, 2), (3, 5), 10)
        b = tr.submit((3,), (4,), 20)

        assert tr.wait(a) == 22
        assert tr.wait(b) == 42

        assert tr.snapshot()["stage_calls"] == (2, 2)


def test_custom_policy_controls_dispatch_priority():
    def scenario(policy):
        entered = Event()
        release = Event()

        def stage0(packet):
            if packet.sequence == 1:
                entered.set()
                assert release.wait(2)
            return packet.payload + 1

        with PolicyCPUMicrobatchTransport(
            (stage0, lambda p: p.payload),
            max_pending=2,
            policy=policy,
        ) as tr:
            a = tr.submit((1,), (3,), 1)
            assert entered.wait(2)

            b = tr.submit((2,), (4,), 2)
            release.set()

            assert tr.wait(a) == 2
            assert tr.wait(b) == 3

            return tr.snapshot()["dispatch_trace"]

    fill = scenario("fill_pipeline")
    downstream = scenario("downstream_first")

    assert fill[:3] == (
        (0, 1),
        (0, 2),
        (1, 1),
    )

    assert downstream[:3] == (
        (0, 1),
        (1, 1),
        (0, 2),
    )


def test_max_inflight_blocks_next_stage_zero():
    downstream_entered = Event()
    downstream_release = Event()
    second_started = Event()

    def stage0(packet):
        if packet.sequence == 2:
            second_started.set()
        return packet.payload

    def stage1(packet):
        if packet.sequence == 1:
            downstream_entered.set()
            assert downstream_release.wait(2)
        return packet.payload

    cfg = {
        "pipeline": {
            "num_ranks": 2,
            "scheduling": {
                "policy": "fill_pipeline",
                "max_inflight_microbatches": 1,
                "max_queued_microbatches": 2,
            },
        }
    }

    with PolicyCPUMicrobatchTransport(
        (stage0, stage1),
        max_pending=2,
        scheduling_config=cfg,
    ) as tr:
        a = tr.submit((1,), (1,), 1)
        assert downstream_entered.wait(2)

        b = tr.submit((2,), (1,), 2)

        assert not second_started.wait(0.02)

        downstream_release.set()

        assert tr.wait(a) == 1
        assert tr.wait(b) == 2
        assert second_started.is_set()


def test_backpressure_and_request_acknowledgement():
    gate = Event()

    def slow(packet):
        assert gate.wait(2)
        return packet.payload

    with PolicyCPUMicrobatchTransport(
        (slow, lambda p: p.payload),
        max_pending=1,
    ) as tr:
        a = tr.submit((1,), (3,), "ok")

        with pytest.raises(PolicyTransportBackpressure):
            tr.submit((2,), (3,), "too many")

        with pytest.raises(TimeoutError):
            tr.wait(a, timeout=0.01)

        gate.set()

        # The computation finishes, but the caller has
        # not acknowledged its result yet.
        with tr._cv:
            ticket = tr._tickets[a]

        assert ticket.ready.wait(2)

        with pytest.raises(PolicyTransportBackpressure):
            tr.submit((1,), (4,), "still reserved")

        assert tr.wait(a) == "ok"

        b = tr.submit((1,), (4,), "again")
        assert tr.wait(b) == "again"


def test_failed_stage_poison_all_tickets():
    def bad(packet):
        raise ValueError("KV position mismatch")

    with PolicyCPUMicrobatchTransport(
        (bad, lambda p: p.payload),
        max_pending=2,
    ) as tr:
        a = tr.submit((1,), (3,), 1)

        with pytest.raises(
            PolicyTransportError,
            match="Pipeline failed",
        ):
            tr.wait(a)

        assert tr.snapshot()["failed"]

        with pytest.raises(PolicyTransportError):
            tr.submit((2,), (1,), 2)


def test_bad_policy_rejects_ineligible_work():
    from llm_runtime.v3_2_pipeline_policy import (
        PipelineDecision,
    )

    class Bad:
        def select_pipeline(self, context):
            return PipelineDecision(999, 0)

    with PolicyCPUMicrobatchTransport(
        (
            lambda p: p.payload,
            lambda p: p.payload,
        ),
        policy=Bad(),
    ) as tr:
        sequence = tr.submit((1,), (3,), 2)

        with pytest.raises(PolicyTransportError):
            tr.wait(sequence)

        assert tr.snapshot()["failed"]


@pytest.mark.parametrize(
    "ids,positions",
    [
        ((1, 1), (2, 2)),
        ((1,), (0,)),
        ((1,), (1, 2)),
        ((), ()),
        ((True,), (1,)),
    ],
)
def test_invalid_envelopes(ids, positions):
    with PolicyCPUMicrobatchTransport(
        (
            lambda p: p.payload,
            lambda p: p.payload,
        )
    ) as tr:
        with pytest.raises(ValueError):
            tr.submit(ids, positions, 1)
