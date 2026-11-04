import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:18080"
BASELINE = Path("benchmarks/qwen3_14b_int8_smoke_20261103.json")
data = json.loads(BASELINE.read_text())

TARGET_INPUT = int(data["input_tokens"])
TARGET_OUTPUT = int(data["requested_output_tokens"])

assert TARGET_INPUT == 130, TARGET_INPUT
assert TARGET_OUTPUT == 64, TARGET_OUTPUT

WARMUPS = 3
RUNS = 10

output = Path(
    "benchmarks/qwen3_14b_marlin_full_"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + ".json"
)


def request(prompt, output_tokens):
    payload = {
        "prompt": prompt,
        "max_new_tokens": output_tokens,
    }

    req = Request(
        URL + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=240) as response:
        return json.load(response)


def find_prompt():
    candidates = []

    for key in ("benchmark_prompt", "prompt", "input_text"):
        value = data.get(key)
        if isinstance(value, str):
            candidates.append(value)

    for run in data.get("runs", []):
        if isinstance(run, dict):
            value = run.get("prompt")
            if isinstance(value, str):
                candidates.append(value)
                break

    for prompt in candidates:
        result = request(prompt, 1)
        if result["prompt_tokens"] == TARGET_INPUT:
            return prompt, "original baseline"

    # Construct a reproducible prompt with exactly 130
    # tokens, checked using the serving tokenizer.
    prefix = (
        "Write a detailed technical tutorial of at least "
        "700 words explaining transformer inference, "
        "attention, KV caching, quantization, GPU kernels, "
        "batching, memory bandwidth, and NCCL communication. "
        "Explain each topic thoroughly with practical examples. "
        "Additional topics:"
    )

    suffix = "\nBegin the detailed tutorial now."

    initial = request(prefix + suffix, 1)["prompt_tokens"]

    for filler in (" example", " data", " GPU", " memory"):
        guess = TARGET_INPUT - initial

        for offset in range(-5, 6):
            count = guess + offset
            if count < 0:
                continue

            prompt = prefix + filler * count + suffix
            result = request(prompt, 1)

            if result["prompt_tokens"] == TARGET_INPUT:
                return prompt, "generated token-matched prompt"

    raise RuntimeError("Could not construct a 130-token prompt")


def validate(result):
    assert result["prompt_tokens"] == TARGET_INPUT, result
    assert result["generated_tokens"] == TARGET_OUTPUT, (
        "Generation ended before 64 tokens: "
        + str(result["generated_tokens"])
    )
    assert not result.get("error"), result


def median(values):
    return statistics.median(values)


print("========== BENCHMARK CONFIGURATION ==========")
print("Input tokens:", TARGET_INPUT)
print("Output tokens:", TARGET_OUTPUT)
print("Warmups:", WARMUPS)
print("Measured runs:", RUNS)

prompt, prompt_source = find_prompt()
print("Prompt source:", prompt_source)
print("Prompt length:", len(prompt), "characters")

results = {
    "model": "Qwen3-14B-GPTQ-Int8",
    "revision": data.get("revision"),
    "split": data.get("split"),
    "backend": "marlin",
    "input_tokens": TARGET_INPUT,
    "requested_output_tokens": TARGET_OUTPUT,
    "prompt_source": prompt_source,
    "prompt": prompt,
    "warmups": WARMUPS,
    "runs": [],
    "baseline_file": str(BASELINE),
}

print("\n========== WARMUP ==========", flush=True)

for i in range(WARMUPS):
    result = request(prompt, TARGET_OUTPUT)
    validate(result)
    print(
        f"Warmup {i + 1}/{WARMUPS}: "
        f"{result['metrics']['e2e_ms']:.2f} ms",
        flush=True,
    )

print("\n========== MEASURED RUNS ==========", flush=True)

for i in range(RUNS):
    start = time.perf_counter()
    result = request(prompt, TARGET_OUTPUT)
    client_ms = (time.perf_counter() - start) * 1000

    validate(result)

    metrics = result["metrics"]

    entry = {
        "run": i + 1,
        "e2e_ms": metrics["e2e_ms"],
        "ttft_ms": metrics["ttft_ms"],
        "decode_interval_median_ms":
            metrics["decode_interval_median_ms"],
        "client_wall_ms": client_ms,
        "generated_tokens": result["generated_tokens"],
        "finish_reason": result.get("finish_reason"),
        "peak_allocated_mib": metrics.get("peak_allocated_mib"),
    }

    results["runs"].append(entry)
    output.write_text(json.dumps(results, indent=2))

    print(
        f"Run {i + 1:02d}: "
        f"E2E={entry['e2e_ms']:.2f} ms  "
        f"TTFT={entry['ttft_ms']:.2f} ms  "
        f"Decode={entry['decode_interval_median_ms']:.2f} ms",
        flush=True,
    )

results["median_e2e_ms"] = median(
    x["e2e_ms"] for x in results["runs"]
)
results["median_ttft_ms"] = median(
    x["ttft_ms"] for x in results["runs"]
)
results["median_decode_ms"] = median(
    x["decode_interval_median_ms"] for x in results["runs"]
)
results["throughput_tokens_per_second"] = (
    1000 / results["median_decode_ms"]
)

output.write_text(json.dumps(results, indent=2))

print("\n========== FINAL COMPARISON ==========")
print(f"{'Metric':<22} {'TritonV2':>13} {'Marlin':>13} {'Speedup':>11}")

for label, baseline_key, result_key in (
    ("E2E", "median_e2e_ms", "median_e2e_ms"),
    ("TTFT", "median_ttft_ms", "median_ttft_ms"),
    ("Decode/token", "median_decode_ms", "median_decode_ms"),
):
    old = float(data[baseline_key])
    new = float(results[result_key])

    print(
        f"{label:<22} "
        f"{old:>10.2f} ms "
        f"{new:>10.2f} ms "
        f"{old / new:>10.2f}x"
    )

print(
    "\nMarlin throughput:",
    round(results["throughput_tokens_per_second"], 2),
    "tokens/s",
)
print("Results saved:", output)
print("PASS: Full Marlin benchmark completed")
