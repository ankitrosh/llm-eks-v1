"""V3.1: Real three-rank Qwen3 inference using CPU/Gloo.

Validates:
- selective stage-only checkpoint loading
- real rank-to-rank tensor transfers
- concurrent request scheduling
- independent per-rank KV caches
- reference-model generation parity
- remote cache release

No GPU, pretrained download, or AWS provisioning required.
"""

import os
import shutil
import tempfile

from datetime import timedelta

import torch
import torch.distributed as dist

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_execution_factory import GenericStageFactory
from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError,
    PipelineScheduler,
    StageResult,
    StageTask,
)
import json as _packet_json
from pathlib import Path as _packet_Path
import torch.distributed as _packet_dist
from llm_runtime.v3_1_packet_factory import create_packet_backend
from llm_runtime.v3_1_launch_spec import compile_v31_launch_spec
from llm_runtime.v3_1_packet_runtime import (
    PacketBackend,
    PacketNRankCoordinator,
    TorchDistributedByteTransport,
    run_packet_nrank_worker,
)
from llm_runtime.v3_1_torch_codecs import ZlibTensorCodec
from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    NRankTransportError,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)


BOUNDARIES = None  # Assigned from the resolved launch plan
PROMPTS = (
    (2, 5, 8, 11),
    (7, 4, 12, 9),
    (3, 13, 18),
)
OUTPUT_TOKENS = 3


def create_checkpoint():
    torch.manual_seed(42)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 6,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )

    config._attn_implementation = "eager"

    model = Qwen3ForCausalLM(config).eval()

    checkpoint = tempfile.mkdtemp(
        prefix="v31-qwen3-distributed-",
        dir="/tmp",
    )

    model.save_pretrained(
        checkpoint,
        safe_serialization=True,
    )

    return checkpoint, model


def reference_generate(model, prompt):
    history = list(prompt)
    generated = []

    with torch.inference_mode():
        for _ in range(OUTPUT_TOKENS):
            ids = torch.tensor(
                [history],
                dtype=torch.long,
            )

            logits = model(
                ids,
                use_cache=False,
            ).logits

            token = int(
                logits[0, -1, :].argmax().item()
            )

            generated.append(token)
            history.append(token)

    return tuple(generated)


def run_rank0(stage, backend, expected, world_size):
    peers = open_nrank_control(
        rank=0,
        world_size=world_size,
        host=os.environ["MASTER_ADDR"],
        port=int(
            os.environ.get("V31_CTRL_PORT", "29679")
        ),
        timeout_s=90,
    )

    link = PacketNRankCoordinator(
        peers,
        backend,
        max_pending=4,
        timeout_s=90,
    )

    def first(task, cache):
        result = stage(task, cache)

        try:
            ticket = link.publish(
                task.request_id,
                task.step,
                task.position,
                result.payload,
            )
        except NRankTransportError as exc:
            raise FatalStageError(str(exc)) from exc

        return StageResult(
            ticket,
            result.cache_state,
        )

    def last(task, cache):
        try:
            logits = link.await_result(
                task.payload
            )
        except NRankTransportError as exc:
            raise FatalStageError(str(exc)) from exc

        return StageResult(logits, cache)

    try:
        with PipelineScheduler(
            [first, last],
            lambda logits: int(
                logits[0, -1, :].argmax().item()
            ),
            max_active_requests=len(PROMPTS),
            on_request_end=link.release_request,
        ) as scheduler:

            futures = [
                scheduler.submit(
                    torch.tensor(
                        [prompt],
                        dtype=torch.long,
                    ),
                    prompt_length=len(prompt),
                    max_new_tokens=OUTPUT_TOKENS,
                    timeout_s=60,
                )
                for prompt in PROMPTS
            ]

            for index, future in enumerate(futures):
                result = future.result(timeout=75)

                actual = tuple(result.tokens)

                assert actual == expected[index], (
                    index,
                    actual,
                    expected[index],
                )

                print(
                    f"PASS: Request {index + 1} "
                    f"matches reference: {actual}",
                    flush=True,
                )

            metrics = scheduler.metrics()

            assert metrics["healthy"], metrics
            assert metrics["failed_requests"] == 0
            assert metrics["completed_requests"] == len(PROMPTS)

            assert metrics["generated_tokens"] == (
                len(PROMPTS) * OUTPUT_TOKENS
            ), metrics

    finally:
        link.close()

    stats = link.stats()
    total = len(PROMPTS) * OUTPUT_TOKENS

    assert stats["sent"] == total, stats
    assert stats["received"] == total, stats
    assert stats["releases"] == len(PROMPTS), stats
    assert stats["hops"] == total * (
        world_size - 2
    ), stats

    print(
        f"PASS: Rank 0 completed distributed "
        f"Qwen3 inference; transport={stats}",
        flush=True,
    )


def run_remote(rank, world_size, stage, backend):
    conn = open_nrank_control(
        rank=rank,
        world_size=world_size,
        host=os.environ["MASTER_ADDR"],
        port=int(
            os.environ.get("V31_CTRL_PORT", "29679")
        ),
        timeout_s=90,
    )

    def execute(r, rid, step, position, hidden, cache):
        result = stage(
            StageTask(rid, step, position, hidden),
            cache,
        )

        return result.payload, result.cache_state

    stats = run_packet_nrank_worker(
        rank,
        world_size,
        conn,
        backend,
        execute,
    )

    total = len(PROMPTS) * OUTPUT_TOKENS

    assert stats["runs"] == total, stats
    assert stats["releases"] == len(PROMPTS), stats

    print(
        f"PASS: Rank {rank} executed {total} "
        f"Qwen3 forwards and released "
        f"{stats['releases']} caches",
        flush=True,
    )


def main():
    global BOUNDARIES
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    user_config = _packet_json.loads(
        _packet_Path(
            "configs/v3_1/qwen3_generated_auto_n3_zlib.json"
        ).read_text()
    )

    if world_size != user_config["pipeline"]["num_ranks"]:
        raise ValueError(
            "WORLD_SIZE disagrees with pipeline.num_ranks"
        )

    torch.set_num_threads(1)

    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=180),
    )

    checkpoint = None

    try:
        reference = None

        if rank == 0:
            checkpoint, reference = create_checkpoint()

        paths = [checkpoint]
        dist.broadcast_object_list(paths, src=0)

        # The generated checkpoint is accessible to all three
        # processes because this CPU test runs on one machine.
        user_config["model"]["checkpoint_dir"] = paths[0]

        launch_spec = compile_v31_launch_spec(user_config)

        BOUNDARIES = tuple(
            launch_spec.layout.boundaries
        )

        # All ranks independently compute the launch plan.
        # Detect configuration/placement disagreements before
        # constructing stage executors.
        digests = [None] * world_size

        dist.all_gather_object(
            digests,
            launch_spec.plan_sha256,
        )

        if len(set(digests)) != 1:
            raise RuntimeError(
                "Ranks disagree on the launch plan"
            )

        if rank == 0:
            print(
                "PASS: Automatic checkpoint-aware "
                f"partition boundaries={BOUNDARIES}",
                flush=True,
            )

            print(
                "PASS: All ranks agree on plan "
                f"digest={launch_spec.plan_sha256}",
                flush=True,
            )

        stage_plan = launch_spec.for_rank(rank)

        assert (
            stage_plan["start_layer"],
            stage_plan["end_layer"],
        ) == (
            BOUNDARIES[rank],
            BOUNDARIES[rank + 1],
        )
        checkpoint = paths[0]

        factory = GenericStageFactory(
            factory_path=(
                "llm_runtime.providers."
                "v3_1_qwen3_nrank_factory:"
                "Qwen3NRankStageOnlyFactory"
            ),
            kwargs={
                "checkpoint_dir": checkpoint,
                "layer_boundaries": list(BOUNDARIES),
                "device": "cpu",
                "dtype": "float32",
            },
        )

        stage = factory(rank)

        assert stage._stage.model.owned_layers == tuple(
            range(
                BOUNDARIES[rank],
                BOUNDARIES[rank + 1],
            )
        )

        print(
            f"PASS: Rank {rank} selectively loaded "
            f"layers {BOUNDARIES[rank]}:"
            f"{BOUNDARIES[rank + 1]}",
            flush=True,
        )

        expected = None

        if rank == 0:
            expected = [
                reference_generate(reference, prompt)
                for prompt in PROMPTS
            ]

            del reference

        dist.barrier()

        backend = create_packet_backend(user_config, rank=_packet_dist.get_rank(), world_size=_packet_dist.get_world_size(), device="cpu")

        if rank == 0:
            run_rank0(
                stage,
                backend,
                expected,
                world_size,
            )

        else:
            run_remote(
                rank,
                world_size,
                stage,
                backend,
            )

        dist.barrier()

        if rank == 0:
            print(
                "PASS: Three-rank distributed Qwen3 "
                "reference parity with real Gloo P2P",
                flush=True,
            )

    finally:
        dist.destroy_process_group()

        if rank == 0 and checkpoint:
            shutil.rmtree(
                checkpoint,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
