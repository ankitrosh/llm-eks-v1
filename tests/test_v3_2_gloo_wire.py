"""Real multi-process CPU/Gloo microbatch forwarding."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llm_runtime.v3_2_gloo_wire import (
    WireMicrobatch,
    validate_wire_microbatch,
    send_wire,
    recv_wire,
)


def _worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )

    try:
        if rank == 0:
            packets = [
                WireMicrobatch(
                    1,
                    "prefill",
                    (3, 8),
                    (0, 0),
                    (2, 4),
                    torch.arange(
                        24, dtype=torch.float32
                    ).reshape(2, 4, 3),
                ),
                WireMicrobatch(
                    2,
                    "decode",
                    (3, 8),
                    (2, 4),
                    (1, 1),
                    torch.arange(
                        6, dtype=torch.float32
                    ).reshape(2, 1, 3),
                ),
            ]

            for packet in packets:
                send_wire(
                    1, kind="run", packet=packet
                )

                kind, result = recv_wire(
                    world_size - 1
                )

                assert kind == "result"
                assert result.sequence == packet.sequence
                assert result.phase == packet.phase
                assert result.request_ids == packet.request_ids
                assert result.positions == packet.positions
                assert result.lengths == packet.lengths

                expected = torch.stack([
                    packet.payload[i, size - 1, :]
                    for i, size in enumerate(
                        packet.lengths
                    )
                ])

                # Each remote stage adds its rank.
                expected += sum(range(1, world_size))

                torch.testing.assert_close(
                    result.payload, expected
                )

            send_wire(1, kind="stop")

        else:
            last_sequence = 0

            while True:
                kind, frame = recv_wire(rank - 1)

                if kind == "stop":
                    if rank + 1 < world_size:
                        send_wire(
                            rank + 1, kind="stop"
                        )
                    return

                assert kind == "run"
                assert frame.sequence == last_sequence + 1

                last_sequence = frame.sequence

                transformed = frame.payload + rank

                if rank + 1 < world_size:
                    send_wire(
                        rank + 1,
                        kind="run",
                        packet=WireMicrobatch(
                            frame.sequence,
                            frame.phase,
                            frame.request_ids,
                            frame.positions,
                            frame.lengths,
                            transformed,
                        ),
                    )

                else:
                    result = torch.stack([
                        transformed[i, size - 1, :]
                        for i, size in enumerate(
                            frame.lengths
                        )
                    ])

                    send_wire(
                        0,
                        kind="result",
                        packet=WireMicrobatch(
                            frame.sequence,
                            frame.phase,
                            frame.request_ids,
                            frame.positions,
                            frame.lengths,
                            result,
                        ),
                    )

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_cpu_gloo_nrank_forwarding(
    tmp_path, ranks
):
    mp.spawn(
        _worker,
        args=(
            ranks,
            str(tmp_path / "gloo.init"),
        ),
        nprocs=ranks,
        join=True,
    )


def test_invalid_wire_packet():
    with pytest.raises(ValueError):
        validate_wire_microbatch(
            WireMicrobatch(
                1, "decode",
                (1, 1),
                (2, 2),
                (1, 1),
                torch.zeros(2, 1, 4),
            )
        )

    with pytest.raises(ValueError):
        validate_wire_microbatch(
            WireMicrobatch(
                1, "prefill",
                (1,),
                (0,),
                (2,),
                torch.zeros(1, 1, 4),
            )
        )
