import json
import urllib.request

BASE = "http://127.0.0.1:8000"

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return json.load(r)

schema = get("/openapi.json")
definition = schema["components"]["schemas"]["GenerateRequest"]
properties = definition["properties"]

assert "prompt" in properties, (
    f"Unexpected generation schema: {list(properties)}"
)

prompt = (
    "Explain pipeline parallelism, NCCL communication, "
    "and KV caching in a distributed transformer system. "
) * 15

payload = {"prompt": prompt}

for key in ("max_new_tokens", "max_tokens", "output_len"):
    if key in properties:
        payload[key] = 24
        break

request = urllib.request.Request(
    BASE + "/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=180) as response:
    result = json.load(response)

assert isinstance(result.get("generated_ids"), list), result

print(json.dumps(result))
