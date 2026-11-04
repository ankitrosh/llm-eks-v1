import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["explicit", "lower_right"],
    required=True,
)
parser.add_argument(
    "--url",
    default="http://127.0.0.1:18081/generate",
)
args = parser.parse_args()

output = Path(f"benchmarks/sdpa_e2e_{args.mode}.json")

context = (
    "In distributed language model inference, transformer "
    "layers are partitioned across GPU workers. "
    "Each worker maintains its local attention key and value "
    "tensors. NCCL transfers intermediate activations between "
    "stages. GPTQ quantization reduces the weight memory "
    "required for large models. Prefix caching reuses "
    "previously processed token contexts. "
) * 12

shared = (
    "Consider the following technical documentation:\n\n"
    + context
    + "\n\nQuestion: "
)

prompt_a = (
    shared
    + "Summarize the purpose of pipeline parallelism."
)

# Large suffix: exercises q > 1 with a substantial KV prefix.
prompt_b = (
    shared
    + "Analyze the following operational scenario:\n"
    + (
        "A GPU worker processes a new request while cached "
        "attention states are available from an earlier request. "
        "Explain the implications for latency and correctness. "
    ) * 18
)

results = {}


def generate(name, prompt):
    req = Request(
        args.url,
        data=json.dumps({
            "prompt": prompt,
            "max_new_tokens": 24,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=240) as response:
        result = json.load(response)

    m = result["metrics"]

    record = {
        "prompt_tokens": result["prompt_tokens"],
        "generated_ids": result["generated_ids"],
        "cache_hit": m["prefix_cache_hit"],
        "cached_tokens": m["prefix_cached_tokens"],
        "prefill_gpu_ms": m["prefill_gpu_ms"],
        "ttft_ms": m["ttft_ms"],
        "e2e_ms": m["e2e_ms"],
        "cache_stored": m["prefix_cache_stored"],
        "cache_used_mib": m["prefix_cache_used_mib"],
    }

    results[name] = record
    output.write_text(json.dumps(results, indent=2))

    print(
        f"{name}: prompt={record['prompt_tokens']} "
        f"reused={record['cached_tokens']} "
        f"prefill={record['prefill_gpu_ms']:.2f}ms "
        f"TTFT={record['ttft_ms']:.2f}ms "
        f"E2E={record['e2e_ms']:.2f}ms",
        flush=True,
    )

    return record


print(f"========== MODE: {args.mode} ==========")

seed = generate("seed_A", prompt_a)
partial = generate("partial_B", prompt_b)

assert seed["cache_stored"], "Seed KV snapshot missing"
assert partial["cache_hit"], "Partial-prefix cache miss"

new_tokens = (
    partial["prompt_tokens"] - partial["cached_tokens"]
)

print("New suffix tokens:", new_tokens)

assert partial["cached_tokens"] >= 512, (
    "Shared cached prefix unexpectedly short"
)
assert new_tokens >= 128, (
    "Suffix too short to test large-prefix FlashAttention"
)

if args.mode == "lower_right":
    baseline_path = Path(
        "benchmarks/sdpa_e2e_explicit.json"
    )
    baseline = json.loads(baseline_path.read_text())

    reference = baseline["partial_B"]

    assert (
        partial["prompt_tokens"] == reference["prompt_tokens"]
        and partial["cached_tokens"] == reference["cached_tokens"]
    ), "Attention experiments used different token lengths"

    assert (
        partial["generated_ids"] == reference["generated_ids"]
    ), "Generated token IDs differ between attention modes"

    print("PASS: Attention modes produce identical token IDs")

    for metric in ("prefill_gpu_ms", "ttft_ms", "e2e_ms"):
        old = reference[metric]
        new = partial[metric]
        print(
            f"{metric}: explicit={old:.3f}ms "
            f"lower_right={new:.3f}ms "
            f"speedup={old / new:.3f}x"
        )

print("PASS: Experiment complete")
print("Saved:", output)
