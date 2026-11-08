"""Gate C.4a: Real NCCL rank execution failure propagation.

Expected:
  rank 1 sends FATAL after injected stage exception
  rank 0 receives the precise failure within 45 seconds
  both ranks exit successfully after verifying the fault

This test does not modify the production transport.
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


INJECTED_ERROR = "V31_INJECTED_EXECUTION_ERROR"
RESULT_TIMEOUT_S = 45


def rank0(stage, backend):
    peers = open_nrank_control(
        0,
        2,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=90,
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
                task.request_id,
                task.step,
                task.position,
                result.payload,
            )

            try:
                link.await_result(ticket)

            except NRankTransportError as exc:
                elapsed = time.monotonic() - start
                message = str(exc)

                print(
                    f"Rank 0 detected error: {message}",
                    flush=True,
                )

                assert INJECTED_ERROR in message, (
                    "Wrong failure reason: " + message
                )

                assert elapsed <= RESULT_TIMEOUT_S, (
                    f"Failure detection took {elapsed:.3f}s"
                )

                print(
                    "PASS: Rank 0 received remote FATAL; "
                    f"detection_seconds={elapsed:.3f}",
                    flush=True,
                )

            else:
                raise AssertionError(
                    "Rank 0 unexpectedly received logits "
                    "instead of the injected failure"
                )

    finally:
        link.close()


def rank1(stage, backend):
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
            "INJECTING: Rank 1 execution exception",
            flush=True,
        )

        raise RuntimeError(INJECTED_ERROR)

    try:
        run_nrank_worker(
            1,
            2,
            connection,
            backend,
            execute,
        )

    except RuntimeError as exc:
        assert str(exc) == INJECTED_ERROR, str(exc)

        print(
            "PASS: Rank 1 raised the injected error "
            "through the transport FATAL path",
            flush=True,
        )

    else:
        raise AssertionError(
            "Worker unexpectedly returned normally"
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
        f"Gate C.4a rank {rank}: "
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
            rank0(stage, backend)
        else:
            rank1(stage, backend)

        # No final barrier: this test intentionally
        # exercises a failed remote execution.
        print(
            f"PASS: Gate C.4a rank {rank} completed",
            flush=True,
        )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
