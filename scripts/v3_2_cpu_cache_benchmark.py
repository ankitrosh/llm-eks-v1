"""Compare DynamicCache and block-backed continuous engines."""

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_continuous_engine import ContinuousQwen3Engine
from llm_runtime.v3_2_block_engine import BlockBackedQwen3Engine


PROMPTS = (
    (2, 5, 8, 11),
    (7, 4, 12, 9, 3, 6),
    (3, 7, 9, 12, 15, 18, 21),
    (4, 8, 16, 32, 64, 96),
)

NEW_TOKENS = 12
REPEATS = 5


def make_model():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )
    config._attn_implementation = "eager"

    return Qwen3ForCausalLM(config).eval()


def settings(concurrency):
    return {
        "pipeline": {
            "memory": {
                "max_active_requests": concurrency,
                "max_total_tokens_per_request": 128,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "policy": "fcfs",
            "max_requests_per_batch": concurrency,
            "max_batch_tokens": 64,
            "max_queue_requests": 8,
        },
    }


def run_once(engine_class, model, concurrency):
    engine = engine_class(
        model,
        settings(concurrency),
        stop_token_ids=(),
    )

    prompts = PROMPTS[:concurrency]
    request_ids = []

    start = perf_counter()

    # Start with two requests.
    for prompt in prompts[:2]:
        request_ids.append(
            engine.submit(prompt, NEW_TOKENS)
        )

    engine.advance()  # Batched prefill
    engine.advance()  # Batched decode

    # Additional requests arrive during active decoding.
    for prompt in prompts[2:]:
        request_ids.append(
            engine.submit(prompt, NEW_TOKENS)
        )

    for _ in range(256):
        state = engine.snapshot()

        if not state["running"] and not state["waiting"]:
            break

        engine.advance()
    else:
        raise AssertionError("Engine failed to drain")

    elapsed = perf_counter() - start

    outputs = tuple(
        engine.result(rid) for rid in request_ids
    )

    assert all(
        output is not None and len(output) == NEW_TOKENS
        for output in outputs
    )

    state = engine.snapshot()

    assert state["reserved_kv_tokens"] == 0
    assert state["cache_request_ids"] == ()

    if "kv_blocks" in state:
        assert state["kv_blocks"]["allocated_blocks"] == 0

    generated = sum(len(output) for output in outputs)

    return {
        "seconds": elapsed,
        "output_tokens": generated,
        "model_forwards": state["model_forwards"],
        "batched_forwards": state["batched_forwards"],
        "outputs": outputs,
    }


def main():
    model = make_model()

    implementations = {
        "dynamic_cache": ContinuousQwen3Engine,
        "block_reference": BlockBackedQwen3Engine,
    }

    report = {
        "environment": "CPU",
        "model": "random Qwen3 2-layer correctness model",
        "repeats": REPEATS,
        "new_tokens_per_request": NEW_TOKENS,
        "results": {},
    }

    for concurrency in (2, 4):
        # Warm up both implementations.
        for cls in implementations.values():
            run_once(cls, model, concurrency)

        measurements = {
            name: []
            for name in implementations
        }

        reference_outputs = None

        for _ in range(REPEATS):
            for name, cls in implementations.items():
                item = run_once(cls, model, concurrency)

                if reference_outputs is None:
                    reference_outputs = item["outputs"]
                else:
                    assert item["outputs"] == reference_outputs, (
                        "Different generated tokens between engines"
                    )

                measurements[name].append(item)

        results = {}

        for name, runs in measurements.items():
            seconds = median(
                r["seconds"] for r in runs
            )

            generated = runs[0]["output_tokens"]

            results[name] = {
                "median_seconds": round(seconds, 6),
                "aggregate_output_tokens_per_second": round(
                    generated / seconds, 3
                ),
                "median_model_forwards": median(
                    r["model_forwards"] for r in runs
                ),
                "median_batched_forwards": median(
                    r["batched_forwards"] for r in runs
                ),
            }

        dynamic = results["dynamic_cache"]["median_seconds"]
        block = results["block_reference"]["median_seconds"]

        results["block_to_dynamic_time_ratio"] = round(
            block / dynamic, 3
        )

        report["results"][str(concurrency)] = results

        print(f"\n========== CONCURRENCY {concurrency} ==========")

        for name in implementations:
            record = results[name]
            print(
                f"{name}: "
                f"{record['aggregate_output_tokens_per_second']}"
                " aggregate tok/s; "
                f"{record['median_seconds']:.4f}s"
            )

        print(
            "Block/Dynamic execution-time ratio:",
            results["block_to_dynamic_time_ratio"],
        )
        print("PASS: Both engines generated identical tokens")

    output = Path(
        "test-results/v3_2/cpu_cache_benchmark.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(f"\nPASS: Benchmark saved to {output}")


if __name__ == "__main__":
    main()
