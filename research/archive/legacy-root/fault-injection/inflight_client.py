import json
import time
import urllib.request
import urllib.error

payload = json.dumps({
    "prompt": (
        "Write 100 numbered, detailed technical observations "
        "about distributed systems, GPU computing, and "
        "transformer inference. Continue until all 100 "
        "observations are complete."
    ),
    "max_new_tokens": 256
}).encode()

request = urllib.request.Request(
    "http://127.0.0.1:8080/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

start = time.perf_counter()
print("REQUEST STARTED", flush=True)

try:
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
        print("REQUEST SUCCEEDED")
        print(json.dumps(result, indent=2))

except urllib.error.HTTPError as exc:
    print("HTTP ERROR:", exc.code)
    print(exc.read(4096).decode(errors="replace"))

except Exception as exc:
    print("REQUEST FAILED:", repr(exc))

finally:
    elapsed = time.perf_counter() - start
    print(f"Client elapsed: {elapsed:.2f} seconds", flush=True)
