import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000/generate"
WARMUPS = 2
RUNS = 10

payload = json.loads(Path("benchmark-request.json").read_text())
records = []

for i in range(WARMUPS + RUNS):
    req = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urlopen(req, timeout=240) as response:
        result = json.load(response)

    wall_ms = (time.perf_counter() - start) * 1000

    record = {
        "run": i + 1,
        "warmup": i < WARMUPS,
        "prompt_tokens": result["prompt_tokens"],
        "generated_tokens": result["generated_tokens"],
        "finish_reason": result["finish_reason"],
        "wall_ms": round(wall_ms, 3),
        **result["metrics"],
    }

    records.append(record)

    print(
        f"{'WARMUP' if record['warmup'] else 'RUN':6} "
        f"{i+1:02d} | "
        f"prompt={record['prompt_tokens']} "
        f"output={record['generated_tokens']} "
        f"e2e={record['e2e_ms']:.2f}ms "
        f"ttft={record['ttft_ms']:.2f}ms "
        f"decode={record['decode_interval_median_ms']:.2f}ms",
        flush=True,
    )

Path("test-results").mkdir(exist_ok=True)
Path("test-results/eks-serving-benchmark.json").write_text(
    json.dumps(records, indent=2)
)

measured = [r for r in records if not r["warmup"]]

print("\n=== EKS SERVING BENCHMARK ===")
print("Requests:", len(measured))
print("Prompt token counts:", sorted(set(r["prompt_tokens"] for r in measured)))
print("Generated token counts:", sorted(set(r["generated_tokens"] for r in measured)))

for key in (
    "e2e_ms",
    "ttft_ms",
    "decode_interval_median_ms",
    "wall_ms",
):
    values = [r[key] for r in measured]
    print(f"{key}: median={statistics.median(values):.3f} ms")

full = [
    r for r in measured
    if r["prompt_tokens"] == 128 and r["generated_tokens"] == 64
]

print(f"\nExact p128/o64 samples: {len(full)}/{len(measured)}")

if full:
    print(
        "Exact p128/o64 median E2E:",
        round(statistics.median(r["e2e_ms"] for r in full), 3),
        "ms",
    )
else:
    print("No exact p128/o64 samples; do not label these results p128/o64.")
