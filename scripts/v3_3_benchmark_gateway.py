"""Benchmark V3.3 gateway using streaming OpenAI completions."""

import json
import statistics
import time
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier


URL = "http://127.0.0.1:8080/v1/completions"
MODEL = "/models/qwen3-14b"

CONCURRENCY = (1, 2, 4)
REPEATS = 2
OUTPUT_TOKENS = 64


def prompt_for(identifier):
    # Unique prefix reduces accidental prefix-cache reuse.
    text = (
        f"Benchmark request {identifier}. "
        "Explain the following distributed inference concepts: "
    )

    text += (
        "A transformer layer applies attention, "
        "residual updates, and feed-forward projections. "
    ) * 9

    return text


def generate(identifier, barrier=None):
    if barrier is not None:
        barrier.wait(timeout=15)

    payload = {
        "model": MODEL,
        "prompt": prompt_for(identifier),
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token = None
    usage = None
    finish_reason = None
    received_done = False

    with urllib.request.urlopen(
        request, timeout=180
    ) as response:
        for raw_line in response:
            line = raw_line.decode(
                "utf-8", errors="replace"
            ).strip()

            if not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                received_done = True
                break

            if not data:
                continue

            event = json.loads(data)

            if event.get("usage"):
                usage = event["usage"]

            for choice in event.get("choices", []):
                if choice.get("text") and first_token is None:
                    first_token = time.perf_counter()

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    end = time.perf_counter()

    if not received_done or usage is None or first_token is None:
        raise RuntimeError(
            f"Incomplete streaming response: {identifier}"
        )

    tokens = usage["completion_tokens"]
    e2e = end - start
    ttft = first_token - start

    tpot = (
        (e2e - ttft) / (tokens - 1)
        if tokens > 1 else 0.0
    )

    return {
        "prompt_tokens": usage["prompt_tokens"],
        "output_tokens": tokens,
        "e2e_ms": e2e * 1000,
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "finish_reason": finish_reason,
    }


def benchmark(concurrency, repeat):
    barrier = Barrier(concurrency)

    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(
                generate,
                f"b{concurrency}-r{repeat}-q{i}",
                barrier,
            )
            for i in range(concurrency)
        ]

        results = [
            future.result()
            for future in futures
        ]

    elapsed = time.perf_counter() - start

    tokens = sum(
        result["output_tokens"]
        for result in results
    )

    return {
        "batch": concurrency,
        "repeat": repeat,
        "total_tokens": tokens,
        "wall_seconds": elapsed,
        "throughput": tokens / elapsed,
        "prompt_tokens": [
            result["prompt_tokens"]
            for result in results
        ],
        "e2e_ms": statistics.median(
            result["e2e_ms"]
            for result in results
        ),
        "ttft_ms": statistics.median(
            result["ttft_ms"]
            for result in results
        ),
        "tpot_ms": statistics.median(
            result["tpot_ms"]
            for result in results
        ),
        "finish_reasons": [
            result["finish_reason"]
            for result in results
        ],
    }


print("Warming up gateway and vLLM...", flush=True)
warm = generate("warmup")
print(
    f"Warmup: {warm['prompt_tokens']} input tokens, "
    f"{warm['output_tokens']} output tokens",
    flush=True,
)

all_results = []

for batch in CONCURRENCY:
    for repeat in range(1, REPEATS + 1):
        result = benchmark(batch, repeat)
        all_results.append(result)

        print(
            f"B{batch} run={repeat} "
            f"input={result['prompt_tokens']} "
            f"output={result['total_tokens']} "
            f"E2E={result['e2e_ms']:.1f}ms "
            f"TTFT={result['ttft_ms']:.1f}ms "
            f"TPOT={result['tpot_ms']:.1f}ms "
            f"aggregate={result['throughput']:.2f}tok/s",
            flush=True,
        )

print("\n========== V3.3 RESULTS ==========")

for batch in CONCURRENCY:
    runs = [
        result for result in all_results
        if result["batch"] == batch
    ]

    print(
        f"B{batch}: "
        f"aggregate={statistics.median(r['throughput'] for r in runs):.2f} tok/s, "
        f"TTFT={statistics.median(r['ttft_ms'] for r in runs):.1f} ms, "
        f"TPOT={statistics.median(r['tpot_ms'] for r in runs):.1f} ms, "
        f"E2E={statistics.median(r['e2e_ms'] for r in runs):.1f} ms"
    )

print("==================================")
