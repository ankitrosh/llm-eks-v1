#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p test-results

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CLIENT_LOG="test-results/inflight-${STAMP}.log"
EVENT_LOG="test-results/inflight-events-${STAMP}.log"

echo "Checking inference workers..."

kubectl wait -n llm \
  --for=condition=Ready \
  pod/qwen3-0 pod/qwen3-1 \
  --timeout=30s

echo "Starting 256-token request..."
echo "Started at $(date -u +%FT%TZ)" | tee "$EVENT_LOG"

kubectl exec -i -n llm \
  deployment/qwen3-coordinator \
  -- python -u - \
  < inflight_client.py \
  > "$CLIENT_LOG" 2>&1 &

CLIENT_PID=$!

INJECTED=false

for i in $(seq 1 30); do
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    echo "Client exited before fault injection" | tee -a "$EVENT_LOG"
    break
  fi

  ACTIVE=$(kubectl exec -n llm qwen3-0 -c runtime \
    -- python -c '
import json
import urllib.request

data = json.load(urllib.request.urlopen(
    "http://127.0.0.1:8000/metrics",
    timeout=2,
))

print(data["active"])
' 2>/dev/null || true)

  if [ "$ACTIVE" = "True" ]; then
    echo "Inference ACTIVE at $(date -u +%FT%TZ)" \
      | tee -a "$EVENT_LOG"

    echo "Crashing Rank 1 model worker..." | tee -a "$EVENT_LOG"

    kubectl exec -i -n llm qwen3-1 -c runtime -- \
      python - < kill_rank1_worker.py | tee -a "$EVENT_LOG"

    INJECTED=true
    break
  fi

  sleep 0.3
done

echo "Fault injected: $INJECTED" | tee -a "$EVENT_LOG"

wait "$CLIENT_PID" || true

echo "Client completed at $(date -u +%FT%TZ)" \
  | tee -a "$EVENT_LOG"

echo
echo "=== CLIENT RESULT ==="
cat "$CLIENT_LOG"

echo
echo "=== EVENT LOG ==="
cat "$EVENT_LOG"

echo
echo "=== RECOVERY LOGS ==="
kubectl logs -n llm \
  deployment/qwen3-recovery \
  --tail=30 | grep -E \
  'Recovery started|Recovery succeeded|Recovery attempt|Recovery exhausted' \
  || true
