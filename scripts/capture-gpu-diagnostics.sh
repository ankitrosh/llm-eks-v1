#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="test-results/gpu-candidate-debug/live/$STAMP"

mkdir -p "$OUT"

kubectl get pod qwen3-0 qwen3-1 \
  -n llm \
  --request-timeout=10s \
  -o json > "$OUT/pods.json" 2>"$OUT/pods-error.txt" || true

for rank in 0 1; do
  kubectl logs "qwen3-$rank" \
    -n llm -c runtime \
    --request-timeout=10s \
    --tail=250 \
    > "$OUT/rank${rank}.log" 2>&1 || true

  kubectl logs "qwen3-$rank" \
    -n llm -c runtime \
    --previous \
    --request-timeout=10s \
    --tail=250 \
    > "$OUT/rank${rank}-previous.log" 2>&1 || true
done

kubectl get events -n llm \
  --request-timeout=10s \
  --sort-by=.lastTimestamp \
  > "$OUT/events.txt" 2>&1 || true

python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
path = folder / "pods.json"

if not path.exists() or not path.stat().st_size:
    print("Pod status unavailable")
    raise SystemExit(0)

data = json.loads(path.read_text())

for pod in data.get("items", []):
    name = pod["metadata"]["name"]
    status = pod.get("status", {})

    print(f"\n{name}: {status.get('phase')}")

    for c in status.get("containerStatuses", []):
        print("  Container:", c["name"])
        print("  Ready:", c.get("ready"))
        print("  Restarts:", c.get("restartCount"))
        print("  Current:", c.get("state"))
        print("  Previous:", c.get("lastState"))
PY

echo "Snapshot: $OUT"
