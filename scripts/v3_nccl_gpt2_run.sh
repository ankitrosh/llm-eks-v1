#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NS=llm
JOB=v3-gpt2-nccl-concurrent
APP=v3-gpt2-nccl-concurrent

MANIFEST=infrastructure/profiles/v3-2-5/manifests/v3-gpt2-nccl-concurrent.yaml

OUT="infrastructure/profiles/v3-2-5/nccl-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

ARMED=0

save_logs() {
  kubectl -n "$NS" get pods \
    -l app="$APP" \
    -o json > "$OUT/pods.json" 2>/dev/null || return 0

  python3 - "$OUT/pods.json" <<'PY' > "$OUT/pod-index.tsv"
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

for pod in data.get("items", []):
    meta = pod["metadata"]
    idx = meta.get("annotations", {}).get(
        "batch.kubernetes.io/job-completion-index"
    )
    if idx is not None:
        print(f"{idx}\t{meta['name']}")
PY

  while IFS=$'\t' read -r INDEX POD; do
    echo "Saving rank $INDEX logs"

    kubectl -n "$NS" logs "pod/$POD" \
      -c runtime --timestamps \
      > "$OUT/rank${INDEX}.log" 2>&1 || true

  done < "$OUT/pod-index.tsv"
}

cleanup() {
  local result=$?

  trap - EXIT INT TERM
  set +e

  if [[ "$ARMED" == 1 ]]; then
    echo "========== SAVE LOGS =========="
    save_logs

    echo "========== AUTOMATIC SHUTDOWN =========="
    bash scripts/v3_nccl_gpt2_cleanup.sh || result=1
  fi

  echo "Results directory: $OUT"
  echo "Final exit code: $result"
  exit "$result"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 scripts/v3_nccl_gpt2_prepare.py

echo "========== PREFLIGHT =========="

test "$(kubectl config current-context)" = \
  "arn:aws:eks:us-east-2:657173565477:cluster/llm-pipeline-v1"

test "$(kubectl -n "$NS" get sts qwen3 \
  -o jsonpath='{.spec.replicas}')" = "0"

for dep in qwen3-coordinator qwen3-recovery \
           qwen3-gpu-observer qwen3-maintenance-supervisor; do
  test "$(kubectl -n "$NS" get deployment "$dep" \
    -o jsonpath='{.spec.replicas}')" = "0"
done

for ng in cpu-system gpu-a10g; do
  test "$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --query 'nodegroup.status' \
    --output text)" = "ACTIVE"

  test "$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)" = "0"
done

if kubectl -n "$NS" get job "$JOB" >/dev/null 2>&1; then
  echo "ERROR: Test Job already exists"
  exit 1
fi

test -f scripts/v3_nccl_gpt2_integration.py

kubectl apply --dry-run=server \
  -f "$MANIFEST" >/dev/null

echo "PASS: NCCL smoke preflight"

ARMED=1

echo "========== CREATE TEST CODE CONFIGMAP =========="

kubectl apply -f /tmp/v3-gpt2-nccl-code-configmap.yaml

start_group() {
  local ng="$1"
  local desired="$2"
  local max="$3"

  echo "========== START $ng =========="

  aws eks update-nodegroup-config \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --scaling-config \
      "minSize=0,maxSize=${max},desiredSize=${desired}" \
    --query 'update.id' \
    --output text
}

wait_nodes() {
  local pool="$1"
  local required="$2"

  for i in $(seq 1 120); do
    COUNT=$(kubectl get nodes \
      -l "pool=$pool" \
      -o json | python3 -c '
import json, sys

data = json.load(sys.stdin)
pool = sys.argv[1]

count = 0
for node in data["items"]:
    conditions = node["status"].get("conditions", [])
    ready = any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in conditions
    )

    gpu = int(
        node["status"]
        .get("allocatable", {})
        .get("nvidia.com/gpu", "0")
    )

    if ready and (pool != "gpu" or gpu >= 1):
        count += 1

print(count)
' "$pool")

    if [[ "$COUNT" -ge "$required" ]]; then
      echo "PASS: $pool nodes ready = $COUNT"
      return 0
    fi

    if (( i % 6 == 1 )); then
      echo "Waiting for $pool nodes: $COUNT/$required"
    fi

    sleep 10
  done

  echo "ERROR: Node readiness timeout"
  return 1
}

start_group cpu-system 1 1
wait_nodes cpu 1

start_group gpu-a10g 2 2
wait_nodes gpu 2

echo "========== VERIFY GPU PLACEMENT =========="
kubectl get nodes -l pool=gpu -o wide

echo "========== START NCCL JOB =========="
kubectl apply -f "$MANIFEST"

RESULT=1

for i in $(seq 1 180); do
  STATUS=$(kubectl -n "$NS" get job "$JOB" \
    -o jsonpath='{.status.succeeded}:{.status.failed}')

  SUCCEEDED=${STATUS%%:*}
  FAILED=${STATUS#*:}

  SUCCEEDED=${SUCCEEDED:-0}
  FAILED=${FAILED:-0}

  if [[ "$SUCCEEDED" == 2 ]]; then
    echo "PASS: Both NCCL ranks completed"
    RESULT=0
    break
  fi

  if [[ "$FAILED" -gt 0 ]]; then
    echo "ERROR: NCCL Job failed"
    break
  fi

  if (( i % 6 == 1 )); then
    echo "NCCL job: succeeded=$SUCCEEDED failed=$FAILED"
    kubectl -n "$NS" get pods -l app="$APP" -o wide
  fi

  sleep 5
done

exit "$RESULT"
