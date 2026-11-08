"""Two-node Qwen3 GPTQ INT8 Marlin / NCCL correctness test.

Rank 0:
  - Compute independent TritonV2 reference results
  - Release reference model
  - Load embeddings and layers 0..13
  - Coordinate NCCL transfers and compare returned logits

Rank 1:
  - Load layers 14..27, final norm and LM head
  - Maintain its own KV cache
  - Execute ordered requests through the V3.1 worker

The framework inference stages load only their assigned
partitions. The temporary independent reference is used
for validation, not distributed inference.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import Qwen3Config

from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)
from llm_runtime.providers.v3_1_configured_stage_factory import (
    create_v31_qwen3_stage_factory,
)

from v3_1_qwen3_independent_reference import (
    PROMPTS,
    DECODE_TOKENS,
    ATOL,
    RTOL,
    reference_logits,
)


def make_config(checkpoint):
    """The same public config is used by both ranks."""

    return {
        "model": {
            "checkpoint_dir": str(checkpoint),
            "quantization": {
                "provider": "gptq",
                "options": {"bits": 8},
            },
        },
        "pipeline": {
            "num_ranks": 2,
            "partition": "manual",
            "layer_counts": [14, 14],
            "memory": {
                "gpu_memory_gib": [20.0, 20.0],
                "reserve_gib": 1.0,
                "max_active_requests": 1,
                "max_total_tokens_per_request": 128,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "identity",
        },
    }


def load_rank_stage(checkpoint, rank):
    factory = create_v31_qwen3_stage_factory(
        make_config(checkpoint),
        device="cuda:0",
        dtype="float16",
    )

    assert tuple(factory.layer_boundaries) == (
        0, 14, 28
    )

    stage = factory(rank)

    projections = stage.model.quantized_linear_count

    assert projections == 98, (
        rank, projections
    )

    print(
        f"PASS: Rank {rank} loaded "
        f"14 layers and {projections} Marlin projections",
        flush=True,
    )

    return stage


def verify_logits(actual, expected, prompt, step):
    actual = actual.detach().float().cpu()

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()

    difference = (actual - expected).abs()
    max_abs = difference.max().item()

    top_match = torch.equal(
        actual.argmax(-1),
        expected.argmax(-1),
    )

    print(
        json.dumps({
            "prompt": prompt,
            "step": step,
            "max_abs": round(max_abs, 6),
            "top1_agrees": top_match,
        }),
        flush=True,
    )

    assert top_match, (
        f"Top-token mismatch: prompt={prompt}, step={step}"
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    return max_abs


def run_rank0(stage, backend, expected):
    peers = open_nrank_control(
        0,
        2,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=600,
    )

    coordinator = NRankCoordinator(
        peers,
        backend,
        max_pending=4,
        timeout_s=300,
    )

    maximum_error = 0.0

    try:
        with torch.inference_mode():
            for index, prompt in enumerate(PROMPTS):
                request_id = index + 1
                cache = None

                inputs = [
                    (
                        0,
                        0,
                        torch.tensor(
                            [prompt],
                            dtype=torch.long,
                        ),
                    )
                ]

                inputs.extend(
                    (
                        step,
                        len(prompt) + step - 1,
                        token,
                    )
                    for step, token in enumerate(
                        DECODE_TOKENS, 1
                    )
                )

                for step, position, payload in inputs:
                    task = SimpleNamespace(
                        request_id=request_id,
                        step=step,
                        position=position,
                        payload=payload,
                    )

                    result = stage(task, cache)
                    cache = result.cache_state

                    ticket = coordinator.publish(
                        request_id,
                        step,
                        position,
                        result.payload,
                    )

                    actual = coordinator.await_result(
                        ticket
                    )

                    error = verify_logits(
                        actual,
                        expected[(index, step)],
                        index,
                        step,
                    )

                    maximum_error = max(
                        maximum_error, error
                    )

                coordinator.release_request(request_id)

                print(
                    f"PASS: Distributed request "
                    f"{request_id}, 4 inference steps",
                    flush=True,
                )

    finally:
        coordinator.close()

    stats = coordinator.stats()

    assert stats["sent"] == 12, stats
    assert stats["received"] == 12, stats
    assert stats["releases"] == 3, stats

    print(
        "PASS: Distributed NCCL rank 0; "
        f"max_abs={maximum_error:.6f}; "
        f"stats={stats}",
        flush=True,
    )


def run_rank1(stage, backend):
    connection = open_nrank_control(
        1,
        2,
        os.environ["MASTER_ADDR"],
        int(os.environ.get("V31_CTRL_PORT", "29679")),
        timeout_s=600,
    )

    def execute(rank, request_id, step, position,
                tensor, cache):
        assert rank == 1
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
        1,
        2,
        connection,
        backend,
        execute,
    )

    assert stats["runs"] == 12, stats
    assert stats["releases"] == 3, stats

    print(
        f"PASS: Distributed NCCL rank 1; "
        f"stats={stats}",
        flush=True,
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

    assert torch.cuda.get_device_capability(0) == (
        8, 6
    )

    model_config = Qwen3Config.from_pretrained(
        str(checkpoint)
    )

    assert model_config.num_hidden_layers == 28
    assert model_config.tie_word_embeddings

    print(
        f"Rank {rank}: "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=600),
    )

    try:
        expected = None

        if rank == 0:
            print(
                "========== INDEPENDENT REFERENCE ==========",
                flush=True,
            )
            expected = reference_logits(checkpoint)
            assert len(expected) == 12

        stage = load_rank_stage(checkpoint, rank)
        backend = TorchP2PBackend("cuda:0")

        if rank == 0:
            run_rank0(stage, backend, expected)
        else:
            run_rank1(stage, backend)

        dist.barrier(device_ids=[0])

        if rank == 0:
            print(
                "PASS: TWO REAL A10G GPUs; "
                "Qwen3 GPTQ INT8 Marlin; "
                "NCCL Socket; "
                "12 independent logit comparisons",
                flush=True,
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
