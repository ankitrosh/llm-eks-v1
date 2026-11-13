"""Gate C.4b: Abrupt NCCL worker termination.

Rank 1 exits with code 77 without sending FATAL.
Rank 0 must detect the lost peer within 30 seconds.
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    NRankTransportError,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)

from v3_1_qwen3_distributed_marlin import (
    PROMPTS,
    load_rank_stage,
)


EXIT_CODE = 77
RESULT_TIMEOUT_S = 25
DETECTION_LIMIT_S = 30


def run_rank0(stage, backend):
    peers = open_nrank_control(
        0,
        2,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=45,
    )

    link = NRankCoordinator(
        peers,
        backend,
        max_pending=2,
        timeout_s=RESULT_TIMEOUT_S,
    )

    try:
        task = SimpleNamespace(
            request_id=1,
            step=0,
            position=0,
            payload=torch.tensor(
                [PROMPTS[0]],
                dtype=torch.long,
            ),
        )

        with torch.inference_mode():
            result = stage(task, None)

            start = time.monotonic()

            ticket = link.publish(
                1, 0, 0, result.payload
            )

            try:
                link.await_result(ticket)

            except NRankTransportError as exc:
                elapsed = time.monotonic() - start
                message = str(exc)

                print(
                    f"Rank 0 peer-loss error: {message}",
                    flush=True,
                )

                assert elapsed <= DETECTION_LIMIT_S, (
                    f"Detection exceeded limit: {elapsed:.3f}s"
                )

                assert "V31_INJECTED_EXECUTION_ERROR" not in message

                expected_signals = (
                    "closed",
                    "connection",
                    "timed out",
                    "timeout",
                    "broken pipe",
                    "reset",
                    "eof",
                )

                assert any(
                    signal in message.lower()
                    for signal in expected_signals
                ), message

                print(
                    "PASS: Rank 0 detected abrupt peer loss; "
                    f"detection_seconds={elapsed:.3f}",
                    flush=True,
                )

            else:
                raise AssertionError(
                    "Unexpected successful inference "
                    "after rank 1 termination"
                )

    finally:
        link.close()


def run_rank1(backend):
    connection = open_nrank_control(
        1,
        2,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=90,
    )

    def execute(rank, request_id, step,
                position, tensor, cache):
        assert rank == 1
        assert request_id == 1
        assert step == 0
        assert tensor.is_cuda

        print(
            "INJECTING: Rank 1 abrupt exit "
            "without FATAL; exit_code=77",
            flush=True,
        )

        os._exit(EXIT_CODE)

    run_nrank_worker(
        1,
        2,
        connection,
        backend,
        execute,
    )

    raise AssertionError(
        "Rank 1 unexpectedly returned normally"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert world_size == 2
    assert rank in (0, 1)

    checkpoint = Path(
        args.checkpoint_dir
    ).resolve()

    assert checkpoint.is_dir()
    assert torch.cuda.is_available()

    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    print(
        f"Gate C.4b rank {rank}: "
        f"{torch.cuda.get_device_name(0)}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=180),
    )

    try:
        stage = load_rank_stage(checkpoint, rank)
        backend = TorchP2PBackend("cuda:0")

        if rank == 0:
            run_rank0(stage, backend)

            print(
                "PASS: Gate C.4b rank 0 completed",
                flush=True,
            )
        else:
            run_rank1(backend)

    finally:
        if rank == 0 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
