"""CPU/Gloo smoke test of real N-rank P2P tensor transfers.

Run with:
  python -m torch.distributed.run --standalone \
    --nproc-per-node=3 scripts/v3_1_gloo_nrank_smoke.py
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from llm_runtime.v3_1_packet_runtime import (
    PacketBackend,
    PacketNRankCoordinator,
    TorchDistributedByteTransport,
    run_packet_nrank_worker,
)
from llm_runtime.v3_1_torch_codecs import ZlibTensorCodec
from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)


def execute(rank, rid, step, position, tensor, cache):
    """Model-independent toy stage with persistent cache."""

    expected_cache = None if step == 0 else step

    if cache != expected_cache:
        raise AssertionError(
            f"Rank {rank}, request {rid}: "
            f"cache {cache} != {expected_cache}"
        )

    assert position == step

    output = tensor + rank * 10.0

    return output, step + 1


def run_coordinator(world_size, backend):
    peers = open_nrank_control(
        rank=0,
        world_size=world_size,
        host="127.0.0.1",
        port=int(os.environ.get("V31_CTRL_PORT", "29678")),
        timeout_s=90,
    )

    link = PacketNRankCoordinator(
        peers,
        backend,
        max_pending=8,
        timeout_s=90,
    )

    try:
        for step in range(3):
            tickets = []

            for rid in range(1, 4):
                # Simulated rank-0 stage.
                tensor = torch.tensor(
                    [[float(rid + 1)]],
                    dtype=torch.float32,
                )

                ticket = link.publish(
                    request_id=rid,
                    step=step,
                    position=step,
                    tensor=tensor,
                )

                tickets.append((rid, ticket))

            for rid, ticket in tickets:
                actual = link.await_result(ticket)

                expected = (
                    rid + 1
                    + 10 * sum(range(1, world_size))
                )

                torch.testing.assert_close(
                    actual,
                    torch.tensor(
                        [[float(expected)]],
                        dtype=torch.float32,
                    ),
                )

        for rid in range(1, 4):
            link.release_request(rid)

    finally:
        link.close()

    stats = link.stats()

    assert stats["sent"] == 9, stats
    assert stats["received"] == 9, stats
    assert stats["releases"] == 3, stats
    assert stats["hops"] == 9 * (world_size - 2), stats

    print(
        f"PASS: Rank 0 completed {world_size}-rank "
        f"Gloo pipeline; stats={stats}",
        flush=True,
    )


def run_remote(rank, world_size, backend):
    conn = open_nrank_control(
        rank=rank,
        world_size=world_size,
        host=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        port=int(os.environ.get("V31_CTRL_PORT", "29678")),
        timeout_s=90,
    )

    stats = run_packet_nrank_worker(
        rank,
        world_size,
        conn,
        backend,
        execute,
    )

    assert stats["runs"] == 9, stats
    assert stats["releases"] == 3, stats
    assert stats["peak_caches"] == 3, stats

    for rid in range(1, 4):
        steps = [
            step for request_id, step in stats["trace"]
            if request_id == rid
        ]
        assert steps == [0, 1, 2], (rank, rid, steps)

    print(
        f"PASS: Rank {rank} completed 9 forwards, "
        f"3 cache releases",
        flush=True,
    )


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size >= 2
    torch.set_num_threads(1)

    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=180),
    )

    try:
        backend = PacketBackend(codec=ZlibTensorCodec(), transport=TorchDistributedByteTransport(), device="cpu")

        if rank == 0:
            run_coordinator(world_size, backend)
        else:
            run_remote(rank, world_size, backend)

        dist.barrier()

        if rank == 0:
            print(
                f"PASS: {world_size}-rank real "
                "PyTorch/Gloo smoke test",
                flush=True,
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
