"""N-rank CUDA/Gloo-staged transport smoke test.

One process per GPU node.

Required environment:
    RANK
    WORLD_SIZE
    MASTER_ADDR
    MASTER_PORT

Optional:
    V31_CTRL_PORT (default 29679)
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from llm_runtime.v3_1_transport_backends import TorchGlooStagedBackend
from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)


REQUESTS = 3
STEPS = 3


def execute(rank, rid, step, position, tensor, cache):
    expected_cache = None if step == 0 else step

    assert cache == expected_cache
    assert position == step
    assert tensor.is_cuda

    return (
        tensor + rank * 10.0,
        step + 1,
    )


def run_rank0(world_size, backend, port):
    peers = open_nrank_control(
        0,
        world_size,
        os.environ["MASTER_ADDR"],
        port,
        timeout_s=90,
    )

    link = NRankCoordinator(
        peers,
        backend,
        max_pending=8,
        timeout_s=90,
    )

    try:
        for step in range(STEPS):
            tickets = []

            for rid in range(1, REQUESTS + 1):
                tensor = torch.tensor(
                    [[float(rid)]],
                    device="cuda:0",
                    dtype=torch.float32,
                )

                ticket = link.publish(
                    rid,
                    step,
                    step,
                    tensor,
                )
                tickets.append((rid, ticket))

            for rid, ticket in tickets:
                actual = link.await_result(ticket)

                expected_value = (
                    rid + 10 * sum(range(1, world_size))
                )

                expected = torch.tensor(
                    [[float(expected_value)]],
                    device="cuda:0",
                )

                torch.testing.assert_close(
                    actual,
                    expected,
                )

        for rid in range(1, REQUESTS + 1):
            link.release_request(rid)

    finally:
        link.close()

    stats = link.stats()
    forwards = REQUESTS * STEPS

    assert stats["sent"] == forwards, stats
    assert stats["received"] == forwards, stats
    assert stats["releases"] == REQUESTS, stats
    assert stats["hops"] == (
        forwards * (world_size - 2)
    ), stats

    print(
        f"PASS: Rank 0 Gloo-staged coordinator: {stats}",
        flush=True,
    )


def run_remote(rank, world_size, backend, port):
    conn = open_nrank_control(
        rank,
        world_size,
        os.environ["MASTER_ADDR"],
        port,
        timeout_s=90,
    )

    stats = run_nrank_worker(
        rank,
        world_size,
        conn,
        backend,
        execute,
    )

    assert stats["runs"] == REQUESTS * STEPS, stats
    assert stats["releases"] == REQUESTS, stats

    for rid in range(1, REQUESTS + 1):
        steps = [
            step
            for request, step in stats["trace"]
            if request == rid
        ]
        assert steps == list(range(STEPS)), steps

    print(
        f"PASS: Rank {rank} Gloo-staged worker: "
        f"{stats['runs']} forwards, "
        f"{stats['releases']} releases",
        flush=True,
    )


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    port = int(os.environ.get(
        "V31_CTRL_PORT", "29679"
    ))

    assert world_size >= 2
    assert 0 <= rank < world_size
    assert torch.cuda.is_available(), (
        "CUDA device unavailable"
    )

    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    free, total = torch.cuda.mem_get_info(0)

    print(
        f"Rank {rank}/{world_size}: "
        f"GPU={torch.cuda.get_device_name(0)}, "
        f"free={free / 2**30:.2f} GiB, "
        f"total={total / 2**30:.2f} GiB",
        flush=True,
    )

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=180),
    )

    try:
        backend = TorchGlooStagedBackend("cuda:0")

        if rank == 0:
            run_rank0(world_size, backend, port)
        else:
            run_remote(rank, world_size, backend, port)

        dist.barrier()

        if rank == 0:
            print(
                f"PASS: {world_size}-rank real "
                "CUDA/Gloo-staged transport smoke test",
                flush=True,
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
