"""V3.1 configuration-driven N-rank Qwen3 NCCL execution.

Runs one autoregressive generation request for deployment
validation. Uses the public launch compiler and stage factory.

The rank-0 prompt is supplied as token IDs. This is a
one-shot execution entrypoint, not an HTTP serving API.
"""

import argparse
import json
import os

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from llm_runtime.v3_1_launch_spec import (
    compile_v31_launch_spec,
)

from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)

from llm_runtime.providers.v3_1_configured_stage_factory import (
    create_v31_qwen3_stage_factory,
)


def load_config(path):
    config = json.loads(Path(path).read_text())

    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")

    checkpoint = config.get("model", {}).get(
        "checkpoint_dir"
    )

    if not checkpoint or not Path(checkpoint).is_dir():
        raise ValueError(
            "Model checkpoint directory is missing"
        )

    return config


def generate(stage, backend, world_size, prompt, count,
             timeout):
    peers = open_nrank_control(
        0,
        world_size,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=timeout,
    )

    coordinator = NRankCoordinator(
        peers,
        backend,
        max_pending=4,
        timeout_s=timeout,
    )

    cache = None
    generated = []

    try:
        with torch.inference_mode():
            for step in range(count):
                position = (
                    0 if step == 0
                    else len(prompt) + step - 1
                )

                payload = (
                    torch.tensor(
                        [prompt],
                        dtype=torch.long,
                    )
                    if step == 0
                    else generated[-1]
                )

                task = SimpleNamespace(
                    request_id=1,
                    step=step,
                    position=position,
                    payload=payload,
                )

                result = stage(task, cache)
                cache = result.cache_state

                ticket = coordinator.publish(
                    1,
                    step,
                    position,
                    result.payload,
                )

                logits = coordinator.await_result(ticket)

                token = int(
                    logits[0, -1, :].argmax().item()
                )

                generated.append(token)

            coordinator.release_request(1)

    finally:
        coordinator.close()

    print(
        "PASS: V3.1 configured NCCL generation",
        flush=True,
    )

    print(json.dumps({
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "world_size": world_size,
        "transport_stats": coordinator.stats(),
    }), flush=True)


def run_worker(rank, stage, backend, world_size, timeout):
    connection = open_nrank_control(
        rank,
        world_size,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=timeout,
    )

    def execute(worker_rank, request_id, step,
                position, tensor, cache):
        assert worker_rank == rank
        assert tensor.is_cuda

        task = SimpleNamespace(
            request_id=request_id,
            step=step,
            position=position,
            payload=tensor,
        )

        result = stage(task, cache)

        return result.payload, result.cache_state

    stats = run_nrank_worker(
        rank,
        world_size,
        connection,
        backend,
        execute,
    )

    print(
        f"PASS: V3.1 worker rank {rank}: {stats}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prompt-token-ids",
        default="2,5,8,11",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    config = load_config(args.config)

    communication = config.get("communication", {})

    if not isinstance(communication, dict):
        raise ValueError("communication must be an object")

    backend_name = communication.get("backend", "auto")
    codec_name = communication.get("codec", "identity")

    if backend_name not in ("nccl", "auto"):
        raise ValueError(
            "This entrypoint requires NCCL or auto transport"
        )

    if codec_name != "identity":
        raise ValueError(
            "NCCL execution requires the identity codec"
        )

    requested_ranks = config["pipeline"]["num_ranks"]

    if requested_ranks != world_size:
        raise ValueError(
            "Kubernetes/Docker WORLD_SIZE differs "
            "from configured num_ranks"
        )

    if not 0 <= rank < world_size:
        raise ValueError("Invalid rank")

    prompt = [
        int(value.strip())
        for value in args.prompt_token_ids.split(",")
    ]

    if not prompt or any(x < 0 for x in prompt):
        raise ValueError("Invalid prompt token IDs")

    if args.max_new_tokens < 1:
        raise ValueError(
            "max-new-tokens must be positive"
        )

    if args.timeout < 1:
        raise ValueError(
            "timeout must be positive"
        )

    memory = config["pipeline"].get("memory", {})
    token_limit = memory.get(
        "max_total_tokens_per_request"
    )

    if (
        token_limit is not None
        and len(prompt) + args.max_new_tokens > token_limit
    ):
        raise ValueError(
            "Generation exceeds configured token limit"
        )

    launch = compile_v31_launch_spec(config)

    if launch.world_size != world_size:
        raise ValueError(
            "Compiled launch rank count mismatch"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    print(
        f"Rank {rank}/{world_size}; "
        f"plan_sha256={launch.plan_sha256}; "
        f"layers={launch.layout.rank_bounds(rank)}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=args.timeout),
    )

    try:
        digests = [None] * world_size

        dist.all_gather_object(
            digests,
            launch.plan_sha256,
        )

        if len(set(digests)) != 1:
            raise RuntimeError(
                "Ranks compiled different launch plans"
            )

        factory = create_v31_qwen3_stage_factory(
            config,
            launch_spec=launch,
            device="cuda:0",
            dtype=args.activation_dtype,
        )

        stage = factory(rank)
        backend = TorchP2PBackend("cuda:0")

        if rank == 0:
            generate(
                stage,
                backend,
                world_size,
                prompt,
                args.max_new_tokens,
                args.timeout,
            )
        else:
            run_worker(
                rank,
                stage,
                backend,
                world_size,
                args.timeout,
            )

        dist.barrier(device_ids=[0])

        print(
            f"PASS: V3.1 configured rank {rank} complete",
            flush=True,
        )

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
