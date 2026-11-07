#!/usr/bin/env python3
"""Config-driven two-rank V3 NCCL inference runner.

Supports decoder-only providers implementing StageTask -> StageResult.
Uses fixed-length greedy generation; EOS stopping is disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_config(path):
    """Validate configuration without importing PyTorch."""
    from scripts.v3_provider_preflight import validate

    data = json.loads(Path(path).read_text())
    validate(data)

    generation = data.get("generation", {})
    has_text = "prompts" in generation
    has_ids = "token_ids" in generation

    if has_text == has_ids:
        raise ValueError(
            "Specify exactly one of "
            "generation.prompts or generation.token_ids"
        )

    if has_ids:
        inputs = generation["token_ids"]

        if (
            not isinstance(inputs, list)
            or not inputs
            or any(
                not isinstance(ids, list)
                or not ids
                or any(
                    type(token) is not int or token < 0
                    for token in ids
                )
                for ids in inputs
            )
        ):
            raise ValueError(
                "generation.token_ids must contain "
                "nonempty lists of nonnegative integers"
            )

    elif (
        not isinstance(data.get("tokenizer_path"), str)
        or not data["tokenizer_path"]
    ):
        raise ValueError(
            "tokenizer_path is required for text prompts"
        )

    for key, default in (
        ("max_new_tokens", 64),
        ("warmup_new_tokens", 8),
    ):
        value = generation.get(key, default)

        if type(value) is not int or value < 1:
            raise ValueError(
                f"generation.{key} must be a positive integer"
            )

    selector = data.get("token_selector")
    if selector is not None and (
        not isinstance(selector, str)
        or ":" not in selector
    ):
        raise ValueError(
            "token_selector must be module:callable"
        )

    return data


def prompt_count(data):
    generation = data["generation"]

    if "prompts" in generation:
        return len(generation["prompts"])

    return len(generation["token_ids"])


def prepare_prompts(data):
    """Prepare inputs on rank 0 only."""
    generation = data["generation"]

    if "token_ids" in generation:
        return generation["token_ids"], None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        data["tokenizer_path"],
        local_files_only=True,
    )

    prompts = [
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        for text in generation["prompts"]
    ]

    if any(not ids for ids in prompts):
        raise ValueError(
            "Tokenizer generated an empty prompt"
        )

    return prompts, tokenizer


def greedy_selector(logits):
    """Default output: [1, sequence, vocabulary]."""
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(
            "Default selector requires logits [1, seq, vocab]"
        )

    return int(
        logits[0, -1, :].argmax().item()
    )


def run(data):
    import torch
    import torch.distributed as dist

    from llm_runtime.v3_execution_factory import (
        GenericStageFactory,
        resolve_factory,
    )
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

    rank = int(os.environ["RANK"])

    if rank not in (0, 1):
        raise RuntimeError("Expected rank 0 or 1")

    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError(
            "Generic NCCL runner currently supports two ranks"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use --validate-only in CloudShell"
        )

    torch.cuda.set_device(0)
    torch.set_num_threads(2)

    device = torch.device("cuda:0")

    pipeline = data.get("pipeline", {})
    generation = data["generation"]

    max_active = pipeline.get("max_active_requests", 4)
    max_pending = pipeline.get("max_pending", 4)
    new_tokens = generation.get("max_new_tokens", 64)
    warmup_tokens = generation.get("warmup_new_tokens", 8)

    requests = prompt_count(data)
    expected_forwards = (
        warmup_tokens + requests * new_tokens
    )

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=600),
    )

    try:
        start_load = time.perf_counter()

        stage = GenericStageFactory(
            **data["provider"]
        )(rank)

        torch.cuda.synchronize(device)

        print(
            "STAGE_LOADED " + json.dumps({
                "rank": rank,
                "provider": data["provider"]["factory_path"],
                "load_seconds": round(
                    time.perf_counter() - start_load, 3
                ),
                "allocated_gpu_gib": round(
                    torch.cuda.memory_allocated(device) / 2**30,
                    3,
                ),
            }),
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

            def execute(
                request_id, step, position, hidden, cache
            ):
                result = stage(
                    StageTask(
                        request_id,
                        step,
                        position,
                        hidden,
                    ),
                    cache,
                )
                return result.payload, result.cache_state

            stats = run_rank1(
                conn,
                device=device,
                execute=execute,
            )

            if (
                stats["runs"] != expected_forwards
                or stats["releases"] != requests + 1
            ):
                raise RuntimeError(
                    f"Unexpected rank 1 statistics: {stats}"
                )

            print(
                "RANK1_METRICS " + json.dumps(stats),
                flush=True,
            )
            print("PASS: Generic rank 1", flush=True)
            return

        prompts, tokenizer = prepare_prompts(data)

        selector = (
            resolve_factory(data["token_selector"])
            if "token_selector" in data
            else greedy_selector
        )

        link = OrderedRank0Transport(
            conn,
            device=device,
            max_pending=max_pending,
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
                raise RuntimeError(
                    "Transport ticket/task mismatch"
                )

            try:
                logits = link.await_result(ticket)
            except TransportFailure as exc:
                raise FatalStageError(str(exc)) from exc

            return StageResult(logits, cache)

        try:
            before = link.stats()["releases"]

            # Warm up in a separate scheduler.
            with PipelineScheduler(
                [first, second],
                selector,
                max_active_requests=max_active,
                on_request_end=link.release_request,
            ) as scheduler:

                future = scheduler.submit(
                    torch.tensor(
                        [prompts[0]],
                        dtype=torch.long,
                        device=device,
                    ),
                    prompt_length=len(prompts[0]),
                    max_new_tokens=warmup_tokens,
                    timeout_s=600,
                )

                result = future.result(timeout=660)

                if len(result.tokens) != warmup_tokens:
                    raise RuntimeError(
                        "Incorrect warm-up token count"
                    )

            # Ensure rank 1 has released the warm-up KV cache
            # before a new scheduler reuses request IDs.
            deadline = time.monotonic() + 30

            while link.stats()["releases"] < before + 1:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Warm-up cache release not acknowledged"
                    )
                time.sleep(0.05)

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            print("PASS: Generic warm-up", flush=True)

            with PipelineScheduler(
                [first, second],
                selector,
                max_active_requests=max_active,
                on_request_end=link.release_request,
            ) as scheduler:

                started = time.perf_counter()

                futures = [
                    scheduler.submit(
                        torch.tensor(
                            [ids],
                            dtype=torch.long,
                            device=device,
                        ),
                        prompt_length=len(ids),
                        max_new_tokens=new_tokens,
                        timeout_s=600,
                    )
                    for ids in prompts
                ]

                results = [
                    future.result(timeout=660)
                    for future in futures
                ]

                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started

                counts = [
                    len(result.tokens)
                    for result in results
                ]

                if counts != [new_tokens] * requests:
                    raise RuntimeError(
                        f"Unexpected output counts: {counts}"
                    )

                metrics = scheduler.metrics()

                if (
                    not metrics["healthy"]
                    or metrics["completed_requests"] != requests
                ):
                    raise RuntimeError(
                        f"Scheduler failure: {metrics}"
                    )

                if stage.forward_calls != expected_forwards:
                    raise RuntimeError(
                        "Stage forward count mismatch"
                    )

                transfer = link.stats()

                if (
                    transfer["sent"] != expected_forwards
                    or transfer["received"] != expected_forwards
                ):
                    raise RuntimeError(
                        f"Transport count mismatch: {transfer}"
                    )

                report = {
                    "provider": data["provider"]["factory_path"],
                    "concurrency_limit": max_active,
                    "requests": requests,
                    "input_token_lengths": [
                        len(ids) for ids in prompts
                    ],
                    "generated_tokens": sum(counts),
                    "wall_seconds": round(elapsed, 6),
                    "aggregate_tps": round(
                        sum(counts) / elapsed, 6
                    ),
                    "scheduler": metrics,
                    "transport": transfer,
                    "peak_gpu_gib": round(
                        torch.cuda.max_memory_allocated(device)
                        / 2**30,
                        3,
                    ),
                }

                print(
                    "GENERIC_METRICS " + json.dumps(report),
                    flush=True,
                )

                for i, result in enumerate(results, 1):
                    print(
                        f"REQUEST_{i}_TOKENS="
                        f"{tuple(result.tokens)}",
                        flush=True,
                    )

                    if tokenizer is not None:
                        decoded = tokenizer.decode(
                            result.tokens,
                            skip_special_tokens=True,
                        )
                        print(
                            f"REQUEST_{i}_TEXT="
                            f"{decoded[:300]!r}",
                            flush=True,
                        )

                print(
                    "PASS: Generic two-rank NCCL inference",
                    flush=True,
                )

        finally:
            link.close()

    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"FAIL: {exc}\n")

    if args.validate_only:
        print(
            "PASS: Generic NCCL runner config valid (CPU only)"
        )
        print(
            "NOTE: Provider, checkpoint, and CUDA "
            "were not tested"
        )
        return

    run(config)


if __name__ == "__main__":
    main()
