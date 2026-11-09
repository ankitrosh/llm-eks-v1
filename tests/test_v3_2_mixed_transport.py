"""Prefill/decode metadata and overlap in CPU policy transport."""

from threading import Event

import pytest

from llm_runtime.v3_2_policy_transport import (
    PolicyCPUMicrobatchTransport,
)


def test_prefill_decode_can_overlap_across_stages():
    prefill_on_stage1 = Event()
    decode_on_stage0 = Event()
    observed = []

    def stage0(packet):
        observed.append((
            0, packet.sequence, packet.phase,
            packet.positions,
        ))

        if packet.phase == "decode":
            assert prefill_on_stage1.wait(5)
            decode_on_stage0.set()

        return packet.payload

    def stage1(packet):
        observed.append((
            1, packet.sequence, packet.phase,
            packet.positions,
        ))

        if packet.phase == "prefill":
            prefill_on_stage1.set()
            assert decode_on_stage0.wait(5)

        return (packet.phase, packet.payload)

    with PolicyCPUMicrobatchTransport(
        (stage0, stage1),
        max_pending=2,
        policy="downstream_first",
    ) as transport:

        prefill = transport.submit(
            (1, 2),
            (0, 0),
            ((2, 5, 8), (7, 4)),
            phase="prefill",
        )

        decode = transport.submit(
            (3,),
            (5,),
            (9,),
            phase="decode",
        )

        assert transport.wait(prefill) == (
            "prefill",
            ((2, 5, 8), (7, 4)),
        )
        assert transport.wait(decode) == (
            "decode", (9,)
        )

        state = transport.snapshot()
        assert state["stage_calls"] == (2, 2)
        assert not state["failed"]

    assert prefill_on_stage1.is_set()
    assert decode_on_stage0.is_set()
    assert (0, prefill, "prefill", (0, 0)) in observed
    assert (1, decode, "decode", (5,)) in observed


def test_chunked_prefill_allows_nonzero_positions():
    seen = []

    def stage(packet):
        seen.append((
            packet.phase,
            packet.positions,
        ))
        return packet.payload

    with PolicyCPUMicrobatchTransport(
        (stage, stage),
        max_pending=2,
    ) as transport:

        first = transport.submit(
            (1,),
            (0,),
            ((2, 5),),
            phase="prefill",
        )
        assert transport.wait(first) == ((2, 5),)

        second = transport.submit(
            (1,),
            (2,),
            ((8, 11),),
            phase="prefill",
        )
        assert transport.wait(second) == ((8, 11),)

    assert seen == [
        ("prefill", (0,)),
        ("prefill", (0,)),
        ("prefill", (2,)),
        ("prefill", (2,)),
    ]


def test_request_ownership_across_phases():
    gate = Event()

    def first_stage(packet):
        assert gate.wait(5)
        return packet.payload

    with PolicyCPUMicrobatchTransport(
        (first_stage, lambda p: p.payload),
        max_pending=2,
    ) as transport:

        sequence = transport.submit(
            (1,),
            (0,),
            ((2, 5),),
            phase="prefill",
        )

        with pytest.raises(ValueError, match="unacknowledged"):
            transport.submit(
                (1,),
                (2,),
                (8,),
                phase="decode",
            )

        gate.set()
        transport.wait(sequence)

        # After acknowledgement, the same request may
        # proceed to the next phase.
        sequence = transport.submit(
            (1,),
            (2,),
            (8,),
            phase="decode",
        )

        assert transport.wait(sequence) == (8,)


@pytest.mark.parametrize(
    "phase,ids,positions,payload,count",
    [
        ("invalid", (1,), (0,), ((2,),), None),
        ("decode", (1,), (0,), (5,), None),
        ("prefill", (1,), (-1,), ((2,),), None),
        ("prefill", (1,), (0,), ((),), None),
        ("prefill", (1,), (0,), ((True,),), None),
        ("prefill", (1,), (0,), ((2, 3),), 1),
        ("decode", (1,), (3,), (5,), 2),
    ],
)
def test_reject_invalid_phase_metadata(
    phase, ids, positions, payload, count
):
    with PolicyCPUMicrobatchTransport(
        (
            lambda packet: packet.payload,
            lambda packet: packet.payload,
        )
    ) as transport:

        with pytest.raises(ValueError):
            transport.submit(
                ids, positions, payload,
                phase=phase,
                token_count=count,
            )


def test_decode_only_callers_remain_compatible():
    with PolicyCPUMicrobatchTransport(
        (
            lambda packet: packet.payload + 1,
            lambda packet: packet.payload * 2,
        )
    ) as transport:

        sequence = transport.submit(
            (1,), (3,), 10
        )
        assert transport.wait(sequence) == 22
