import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:18081/generate"

output = Path(
    "benchmarks/prefix_correctness_"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + ".json"
)

# Long identical context ensures substantial prefix reuse.
context = (
    "In a distributed transformer inference pipeline, "
    "GPU zero owns the first half of the transformer layers "
    "and GPU one owns the second half. "
    "Each stage maintains its own key and value tensors. "
    "NCCL transfers intermediate hidden activations between stages. "
    "Quantized matrix multiplication reduces GPU memory and latency. "
    "Prefix caching avoids recomputing attention states "
    "for identical previously processed input tokens. "
) * 12

base = (
    "Read the following technical context carefully:\n\n"
    + context
    + "\n\nQuestion: "
)

prompt_a = (
    base
    + "Explain why prefix KV caching reduces prefill computation."
)

prompt_b = (
    base
    + "Explain why NCCL communication is necessary between GPU stages."
)

results = []


def generate(name, prompt):
    request = Request(
        URL,
        data=json.dumps({
            "prompt": prompt,
            "max_new_tokens": 48,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urlopen(request, timeout=240) as response:
        result = json.load(response)

    client_ms = (time.perf_counter() - start) * 1000
    metrics = result["metrics"]

    record = {
        "name": name,
        "prompt_tokens": result["prompt_tokens"],
        "generated_tokens": result["generated_tokens"],
        "generated_ids": result["generated_ids"],
        "text": result["text"],
        "finish_reason": result["finish_reason"],
        "e2e_ms": metrics["e2e_ms"],
        "ttft_ms": metrics["ttft_ms"],
        "prefill_gpu_ms": metrics["prefill_gpu_ms"],
        "decode_ms": metrics["decode_interval_median_ms"],
        "cache_hit": metrics["prefix_cache_hit"],
        "cached_tokens": metrics["prefix_cached_tokens"],
        "cache_stored": metrics["prefix_cache_stored"],
        "cache_entries": metrics["prefix_cache_entries"],
        "cache_used_mib": metrics["prefix_cache_used_mib"],
        "client_ms": round(client_ms, 2),
    }

    results.append(record)

    # Save after each request, even if a later assertion fails.
    output.write_text(json.dumps(results, indent=2))

    print(f"\n========== {name} ==========", flush=True)
    print("Prompt tokens:", record["prompt_tokens"])
    print("Generated tokens:", record["generated_tokens"])
    print("Cache hit:", record["cache_hit"])
    print("Cached tokens:", record["cached_tokens"])
    print("Snapshot stored:", record["cache_stored"])
    print("Cache memory (MiB):", record["cache_used_mib"])
    print("Prefill GPU (ms):", record["prefill_gpu_ms"])
    print("TTFT (ms):", record["ttft_ms"])
    print("E2E (ms):", record["e2e_ms"])
    print("Text:", record["text"][:250], flush=True)

    assert record["prompt_tokens"] > 64
    assert record["generated_tokens"] > 0

    return record


print("Starting distributed prefix-cache correctness test")

cold = generate("cold_A", prompt_a)

assert not cold["cache_hit"], (
    "Expected initial request to be a cache miss"
)
assert cold["cache_stored"], (
    "Initial prompt KV snapshot was not stored"
)

warm = generate("warm_A", prompt_a)

assert warm["cache_hit"], (
    "Repeated prompt did not produce a cache hit"
)

assert warm["cached_tokens"] == warm["prompt_tokens"] - 1, (
    "Expected all but the final prompt token to be reused"
)

assert cold["generated_ids"] == warm["generated_ids"], (
    "FAIL: Cached generation differs from uncached generation"
)

print("\nPASS: Cold and warm generated token IDs match")

partial = generate("partial_B", prompt_b)

assert partial["cache_hit"], (
    "Shared-context request did not reuse the prefix"
)
assert partial["cached_tokens"] >= 64
assert partial["cached_tokens"] < partial["prompt_tokens"]

print("\n========== PERFORMANCE COMPARISON ==========")

for key in ("prefill_gpu_ms", "ttft_ms", "e2e_ms"):
    a = cold[key]
    b = warm[key]

    print(
        f"{key}: cold={a:.2f} ms, "
        f"warm={b:.2f} ms, "
        f"speedup={a / b:.2f}x"
    )

print("\nPASS: Distributed prefix-cache smoke tests")
print("Results:", output)
print(
    "Note: Partial-prefix output still needs comparison "
    "against a separate cold reference."
)
