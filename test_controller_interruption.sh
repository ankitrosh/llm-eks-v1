#!/usr/bin/env bash
set -euo pipefail

cd ~/llm-eks-v1
mkdir -p test-results

state_field() {
  kubectl get configmap qwen3-recovery-state \
    -n llm -o jsonpath='{.data.state}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"
}

echo "=== Preflight ==="

MODE=$(state_field mode)
PHASE=$(state_field phase)

if [[ "$MODE" != "RUNNING" || "$PHASE" != "SERVING" ]]; then
  echo "Controller not healthy. Aborting."
  exit 1
fi

kubectl wait -n llm --for=condition=Ready \
  pod/qwen3-0 pod/qwen3-1 --timeout=30s

BEFORE_GENERATION=$(state_field generation)
BEFORE_RECOVERIES=$(state_field recoveries)

echo "Generation before: $BEFORE_GENERATION"
echo "Recoveries before: $BEFORE_RECOVERIES"

echo
echo "=== Injecting Rank 1 failure ==="

kubectl delete pod qwen3-1 -n llm --wait=false

echo
echo "=== Waiting for persisted RECOVERING and replicas=0 ==="

FOUND=false

for i in $(seq 1 120); do
  PHASE=$(state_field phase)
  REPLICAS=$(kubectl get statefulset qwen3 \
    -n llm -o jsonpath='{.spec.replicas}')

  if [[ "$PHASE" == "RECOVERING" && "$REPLICAS" == "0" ]]; then
    FOUND=true
    echo "Recovery in progress; GPU replicas=0"
    break
  fi

  sleep 1
done

if [[ "$FOUND" != "true" ]]; then
  echo "Did not observe expected interruption point."
  echo "No controller restart injected."
  exit 1
fi

echo
echo "=== Restarting CPU recovery controller ==="

kubectl rollout restart deployment/qwen3-recovery -n llm

kubectl rollout status deployment/qwen3-recovery \
  -n llm --timeout=180s

echo
echo "=== Waiting for automatic GPU recovery ==="

kubectl wait -n llm --for=condition=Ready \
  pod/qwen3-0 pod/qwen3-1 --timeout=900s

FOUND=false

for i in $(seq 1 60); do
  PHASE=$(state_field phase)

  if [[ "$PHASE" == "SERVING" ]]; then
    FOUND=true
    break
  fi

  sleep 2
done

if [[ "$FOUND" != "true" ]]; then
  echo "Controller did not reach SERVING."
  exit 1
fi

AFTER_GENERATION=$(state_field generation)
AFTER_RECOVERIES=$(state_field recoveries)

echo
echo "=== RESULTS ==="
echo "Generation: $BEFORE_GENERATION -> $AFTER_GENERATION"
echo "Recoveries: $BEFORE_RECOVERIES -> $AFTER_RECOVERIES"
echo "Final phase: $PHASE"

if [[ "$AFTER_GENERATION" -eq $((BEFORE_GENERATION + 1)) &&
      "$AFTER_RECOVERIES" -eq $((BEFORE_RECOVERIES + 1)) ]]; then
  echo "PASS: Recovery survived controller restart"
else
  echo "FAIL: Recovery counters unexpected"
  exit 1
fi
