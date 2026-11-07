"""Two-rank V3 Qwen3-14B 20/20 NCCL concurrency benchmark.

Loads only owned BF16 transformer layers, never a full reference model.
Generates a fixed number of tokens with EOS stopping disabled.
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3Config

from llm_runtime.v3_execution_factory import GenericStageFactory
from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError,
    PipelineScheduler,
    StageResult,
    StageTask,
)
from llm_runtime.v3_ordered_rank_transport import (
    OrderedRank0Transport,
    TransportFailure,
    open_control,
    run_rank1,
)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/qwen3-14b"))
SPLIT_AT = 20
REQUESTS = 4
PROMPT_TOKENS = 128
NEW_TOKENS = 64
WARMUP_TOKENS = 16

PROMPTS = (
    "Explain the role of memory bandwidth in transformer inference. ",
    "Discuss why pipeline parallelism can improve aggregate serving throughput. ",
    "Compare key value caching with recomputing attention for each token. ",
    "Describe tradeoffs in distributed GPU communication and scheduling. ",
)


def check_checkpoint():
    """Fail locally without downloading anything."""
    config_path = MODEL_DIR / "config.json"
    index_path = MODEL_DIR / "model.safetensors.index.json"

    if not config_path.is_file():
        raise RuntimeError(f"Missing BF16 config: {config_path}")

    if not index_path.is_file():
        raise RuntimeError(f"Missing sharded BF16 index: {index_path}")

    config = Qwen3Config.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    if config.num_hidden_layers != 40:
        raise RuntimeError(
            f"Expected 40 layers, got {config.num_hidden_layers}"
        )

    if getattr(config, "quantization_config", None):
        raise RuntimeError(
            "Checkpoint is quantized; BF16 stage loader required"
        )

    data = json.loads(index_path.read_text())
    shards = sorted(set(data["weight_map"].values()))

    missing = [
        name for name in shards
        if not (MODEL_DIR / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Missing BF16 checkpoint shards: {missing}"
        )

    required = (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.39.self_attn.q_proj.weight",
        "lm_head.weight",
    )

    for name in required:
        if name not in data["weight_map"]:
            if name == "lm_head.weight" and config.tie_word_embeddings:
                continue
            raise RuntimeError(f"Missing tensor in index: {name}")

    revision_file = MODEL_DIR / ".model_revision"
    revision = (
        revision_file.read_text().strip()
        if revision_file.is_file()
        else "unknown"
    )

    print(
        f"CHECKPOINT_OK rank={os.environ['RANK']} "
        f"shards={len(shards)} layers=40 "
        f"revision={revision} dir={MODEL_DIR}",
        flush=True,
    )


def make_prompts(tokenizer):
    """Four deterministic 128-token prompts."""
    result = []

    for text in PROMPTS:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if not ids:
            raise RuntimeError("Tokenizer produced an empty prompt")

        result.append(
            (ids * ((PROMPT_TOKENS + len(ids) - 1) // len(ids)))
            [:PROMPT_TOKENS]
        )

    return result


def main():
    rank = int(os.environ["RANK"])

    if rank not in (0, 1) or int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("Exactly two ranks are required")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not visible")

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")

    check_checkpoint()

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=600),
    )

    try:
        loaded_at = time.perf_counter()

        stage = GenericStageFactory(
            factory_path=(
                "llm_runtime.providers.v3_qwen3_stage_only_factory:"
                "Qwen3StageOnlyFactory"
            ),
            kwargs={
                "checkpoint_dir": str(MODEL_DIR),
                "device": "cuda:0",
                "dtype": "bfloat16",
                "split_at": SPLIT_AT,
            },
        )(rank)

        torch.cuda.synchronize(device)

        owned_layers = getattr(
            stage._stage.model,
            "owned_layers",
            (),
        )

        expected_layers = tuple(
            range(rank * 20, (rank + 1) * 20)
        )

        if tuple(owned_layers) != expected_layers:
            raise RuntimeError(
                f"Wrong stage ownership: {owned_layers}"
            )

        owned_params = sum(
            p.numel()
            for p in stage._stage.model.parameters()
        )

        print(
            f"STAGE_LOADED rank={rank} "
            f"layers={owned_layers[0]}..{owned_layers[-1]} "
            f"parameters={owned_params} "
            f"load_s={time.perf_counter()-loaded_at:.3f} "
            f"allocated_gib="
            f"{torch.cuda.memory_allocated(device)/2**30:.3f}",
            flush=True,
        )

        dist.barrier(device_ids=[0])

        conn = open_control(
            rank,
            os.environ["MASTER_ADDR"],
            int(os.getenv("V3_CTRL_PORT", "29659")),
            timeout_s=180,
        )

        if rank == 1:

            def execute(rid, step, position, hidden, cache):
                result = stage(
                    StageTask(rid, step, position, hidden),
                    cache,
                )
                return result.payload, result.cache_state

            stats = run_rank1(
                conn,
                device=device,
                execute=execute,
            )

            expected_forwards = (
                REQUESTS * NEW_TOKENS + WARMUP_TOKENS
            )

            assert stats["runs"] == expected_forwards, stats
            assert stats["releases"] == REQUESTS + 1, stats

            print(
                "RANK1_METRICS " + json.dumps({
                    "forwards": stats["runs"],
                    "releases": stats["releases"],
                    "peak_caches": stats["peak_caches"],
                    "peak_gpu_gib": round(
                        torch.cuda.max_memory_allocated(device)
                        / 2**30, 3
                    ),
                }),
                flush=True,
            )

            print(
                "PASS: Rank 1 full-model inference",
                flush=True,
            )
            return

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        prompts = make_prompts(tokenizer)

        link = OrderedRank0Transport(
            conn,
            device=device,
            max_pending=4,
            timeout_s=240,
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
            except TransportFailure as exc:
                raise FatalStageError(str(exc)) from exc

            return StageResult(
                ticket,
                result.cache_state,
            )

        def second(task, cache):
            ticket = task.payload

            if (
                ticket.request_id,
                ticket.step,
                ticket.position,
            ) != (
                task.request_id,
                task.step,
                task.position,
            ):
                raise RuntimeError("Ticket/task mismatch")

            try:
                logits = link.await_result(ticket)
            except TransportFailure as exc:
                raise FatalStageError(str(exc)) from exc

            return StageResult(logits, cache)

        try:
            # Warm-up excluded from benchmark metrics.
            before_releases = link.stats()["releases"]

            with PipelineScheduler(
                [first, second],
                lambda logits: int(
                    logits[0, -1, :].argmax().item()
                ),
                max_active_requests=REQUESTS,
                on_request_end=link.release_request,
            ) as warmup_scheduler:

                warmup = warmup_scheduler.submit(
                    torch.tensor(
                        [prompts[0]],
                        device=device,
                        dtype=torch.long,
                    ),
                    prompt_length=PROMPT_TOKENS,
                    max_new_tokens=WARMUP_TOKENS,
                    timeout_s=600,
                ).result(timeout=660)

                assert len(warmup.tokens) == WARMUP_TOKENS

            # Wait until rank 1 acknowledges cache release.
            deadline = time.monotonic() + 30

            while link.stats()["releases"] < before_releases + 1:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Warm-up cache release not acknowledged"
                    )
                time.sleep(0.05)

            torch.cuda.synchronize(device)

            print(
                "PASS: Qwen3-14B warm-up completed",
                flush=True,
            )

            torch.cuda.reset_peak_memory_stats(device)

            with PipelineScheduler(
                [first, second],
                lambda logits: int(
                    logits[0, -1, :].argmax().item()
                ),
                max_active_requests=REQUESTS,
                on_request_end=link.release_request,
            ) as scheduler:

                t0 = time.perf_counter()

                futures = [
                    scheduler.submit(
                        torch.tensor(
                            [prompt],
                            device=device,
                            dtype=torch.long,
                        ),
                        prompt_length=len(prompt),
                        max_new_tokens=NEW_TOKENS,
                        timeout_s=600,
                    )
                    for prompt in prompts
                ]

                results = [
                    future.result(timeout=660)
                    for future in futures
                ]

                elapsed = time.perf_counter() - t0

                counts = [
                    len(result.tokens)
                    for result in results
                ]

                assert counts == [NEW_TOKENS] * REQUESTS, counts

                metrics = scheduler.metrics()

                assert metrics["healthy"], metrics
                assert metrics["completed_requests"] == REQUESTS, metrics
                assert metrics["generated_tokens"] == REQUESTS * NEW_TOKENS, metrics

                expected_forwards = (
                    REQUESTS * NEW_TOKENS + WARMUP_TOKENS
                )

                assert stage.forward_calls == expected_forwards

                transfer = link.stats()

                assert transfer["sent"] == expected_forwards, transfer
                assert transfer["received"] == expected_forwards, transfer

                torch.cuda.synchronize(device)

                print(
                    "BENCHMARK_METRICS " + json.dumps({
                        "model": "Qwen3-14B",
                        "dtype": "bfloat16",
                        "split": "20/20",
                        "concurrency": REQUESTS,
                        "input_tokens_per_request": PROMPT_TOKENS,
                        "output_tokens_per_request": NEW_TOKENS,
                        "generated_tokens": sum(counts),
                        "wall_seconds": round(elapsed, 6),
                        "aggregate_tps": round(
                            sum(counts) / elapsed, 6
                        ),
                        "avg_ttft_ms": metrics["avg_ttft_ms"],
                        "avg_latency_ms": metrics["avg_latency_ms"],
                        "stages": metrics["stages"],
                        "transport": transfer,
                        "peak_gpu_gib": round(
                            torch.cuda.max_memory_allocated(device)
                            / 2**30, 3
                        ),
                    }),
                    flush=True,
                )

                for i, result in enumerate(results, 1):
                    print(
                        f"REQUEST_{i}_TOKENS={tuple(result.tokens)}",
                        flush=True,
                    )
                    print(
                        f"REQUEST_{i}_TEXT="
                        f"{tokenizer.decode(result.tokens, skip_special_tokens=True)[:300]!r}",
                        flush=True,
                    )

                print(
                    "PASS: Qwen3-14B 20/20 four-request NCCL benchmark",
                    flush=True,
                )

        finally:
            link.close()

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
