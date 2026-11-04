import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:18081/generate"

previous_files = sorted(
    Path("benchmarks").glob("prefix_correctness_*.json")
)
assert previous_files, "Previous correctness results missing"

previous = json.loads(previous_files[-1].read_text())
reference = next(
    r for r in previous if r["name"] == "partial_B"
)

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

prompt_b = (
    "Read the following technical context carefully:\n\n"
    + context
    + "\n\nQuestion: "
    + "Explain why NCCL communication is necessary between GPU stages."
)

output = Path(
    "benchmarks/prefix_eviction_"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + ".json"
)

results = []


def generate(name, prompt, max_tokens):
    request = Request(
        URL,
        data=json.dumps({
            "prompt": prompt,
            "max_new_tokens": max_tokens,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=240) as response:
        raw = json.load(response)

    metrics = raw["metrics"]

    result = {
        "name": name,
        "prompt_tokens": raw["prompt_tokens"],
        "generated_ids": raw["generated_ids"],
        "cache_hit": metrics["prefix_cache_hit"],
        "cached_tokens": metrics["prefix_cached_tokens"],
        "cache_stored": metrics["prefix_cache_stored"],
        "entries": metrics["prefix_cache_entries"],
        "used_mib": metrics["prefix_cache_used_mib"],
        "prefill_gpu_ms": metrics["prefill_gpu_ms"],
        "ttft_ms": metrics["ttft_ms"],
        "e2e_ms": metrics["e2e_ms"],
    }

    results.append(result)
    output.write_text(json.dumps(results, indent=2))

    print(
        f"{name}: "
        f"hit={result['cache_hit']} "
        f"reused={result['cached_tokens']} "
        f"stored={result['cache_stored']} "
        f"entries={result['entries']} "
        f"memory={result['used_mib']}MiB "
        f"TTFT={result['ttft_ms']:.2f}ms",
        flush=True,
    )

    assert result["entries"] <= 4
    assert result["used_mib"] <= 256.1

    return result


print("========== EVICT PREVIOUS PREFIXES ==========")

# Four distinct long prompts fill the 4-entry LRU cache.
# They diverge near the beginning and don't share a
# sufficiently long prefix with Question B.
for i in range(4):
    prompt = (
        f"Independent eviction test {i}: "
        + context
        + "\nDescribe the technical context."
    )

    result = generate(f"eviction_{i}", prompt, 1)
    assert result["cache_stored"], (
        f"Eviction prompt {i} was not cached"
    )


print("\n========== COLD QUESTION B ==========")

cold = generate("cold_B", prompt_b, 48)

assert cold["prompt_tokens"] == reference["prompt_tokens"]
assert not cold["cache_hit"], (
    "Question B was not completely evicted"
)


print("\n========== WARM QUESTION B ==========")

warm = generate("warm_B", prompt_b, 48)

assert warm["cache_hit"]
assert warm["cached_tokens"] == warm["prompt_tokens"] - 1


print("\n========== CORRECTNESS VERIFICATION ==========")

assert cold["generated_ids"] == warm["generated_ids"], (
    "FAIL: Cold and warm Question B differ"
)

assert cold["generated_ids"] == reference["generated_ids"], (
    "FAIL: Original partial-prefix answer differs from cold B"
)

print("PASS: Cold, full-cache and partial-cache outputs match")
print("PASS: LRU eviction works")
print("PASS: Cache memory limit respected")

print(
    "TTFT improvement:",
    round(cold["ttft_ms"] / warm["ttft_ms"], 2),
    "x",
)
print("Results:", output)
