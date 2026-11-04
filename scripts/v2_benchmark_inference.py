"""Small, repeatable HTTP inference benchmark. Standard library only."""

import json
import statistics
import sys
import urllib.request

label = sys.argv[1] if len(sys.argv) > 1 else "unknown"

prompt = (
    "Distributed LLM inference requires correct KV caching, "
    "token positioning, GPU communication and synchronization. "
) * 8

prompt += (
    " Explain 20 detailed steps for implementing a reliable "
    "multi-GPU inference pipeline. Give an example for each step."
)

payload = json.dumps({
    "prompt": prompt,
    "max_new_tokens": 64,
}).encode()

def generate():
    request = urllib.request.Request(
        "http://127.0.0.1:8000/generate",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    assert result.get("generated_tokens", 0) > 0, result
    return result

# Warm up without measuring.
for _ in range(2):
    generate()

samples = []

for i in range(8):
    result = generate()
    metrics = result["metrics"]

    sample = {
        "generated_tokens": result["generated_tokens"],
        "e2e_ms": metrics["e2e_ms"],
        "ttft_ms": metrics["ttft_ms"],
        "decode_interval_median_ms":
            metrics["decode_interval_median_ms"],
    }

    samples.append(sample)

def median(key):
    return round(
        statistics.median(sample[key] for sample in samples),
        3,
    )

report = {
    "label": label,
    "warmups": 2,
    "measured_requests": len(samples),
    "max_new_tokens": 64,
    "prompt_chars": len(prompt),
    "generated_token_counts": [
        s["generated_tokens"] for s in samples
    ],
    "median": {
        "e2e_ms": median("e2e_ms"),
        "ttft_ms": median("ttft_ms"),
        "decode_interval_median_ms":
            median("decode_interval_median_ms"),
    },
    "samples": samples,
}

print(json.dumps(report, indent=2))
