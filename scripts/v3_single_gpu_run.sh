#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.." || exit 1

export AWS_PAGER=""

CLUSTER=llm-pipeline-v1
REGION=us-east-2
NS=llm
JOB=v3-gpt2-single-gpu

mkdir -p infrastructure/profiles/v3-1-6

LOG="infrastructure/profiles/v3-1-6/single-gpu-$(date -u +%Y%m%dT%H%M%SZ).log"

echo "Test log: $LOG"

CLEANUP_ARMED=0

cleanup() {
  local status=$?
  trap - EXIT

  if [ "$CLEANUP_ARMED" = "1" ]; then
    echo "========== AUTOMATIC GPU SHUTDOWN =========="
    if ! bash scripts/v3_single_gpu_cleanup.sh; then
      echo "ERROR: Shutdown not fully verified"
      status=1
    fi
  fi

  echo "Exit status: $status"
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "========== PREFLIGHT =========="

test "$(kubectl config current-context)" = \
  "arn:aws:eks:us-east-2:657173565477:cluster/llm-pipeline-v1"

test "$(kubectl -n "$NS" get sts qwen3 \
  -o jsonpath='{.spec.replicas}')" = "0"

for DEP in qwen3-coordinator qwen3-recovery \
           qwen3-gpu-observer qwen3-maintenance-supervisor; do

  test "$(kubectl -n "$NS" get deployment "$DEP" \
    -o jsonpath='{.spec.replicas}')" = "0"
done

for NG in gpu-a10g cpu-system; do
  test "$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --region "$REGION" \
    --query 'nodegroup.status' \
    --output text)" = "ACTIVE"

  test "$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)" = "0"
done

if kubectl -n "$NS" get job "$JOB" &>/dev/null; then
  echo "ERROR: Test job already exists"
  exit 1
fi

kubectl -n "$NS" get cm \
  v3-gpt2-gpu-test \
  v3-gpt2-framework-src >/dev/null

kubectl apply --dry-run=server \
  -f /tmp/v3-gpt2-single-gpu-job.yaml

echo "PASS: Preflight checks"

CLEANUP_ARMED=1

echo "========== START ONE A10G =========="

aws eks update-nodegroup-config \
  --cluster-name "$CLUSTER" \
  --nodegroup-name gpu-a10g \
  --region "$REGION" \
  --scaling-config minSize=0,maxSize=2,desiredSize=1 \
  --query 'update.id' \
  --output text

echo "========== WAIT FOR GPU DEVICE =========="

READY=0

for i in $(seq 1 90); do
  COUNT=$(kubectl get nodes -l pool=gpu -o json |
    python3 -c '
import json, sys
nodes=json.load(sys.stdin)["items"]
count=0
for n in nodes:
    conditions=n["status"].get("conditions", [])
    ready=any(
        x["type"]=="Ready" and x["status"]=="True"
        for x in conditions
    )
    gpu=int(
        n["status"].get("allocatable", {})
        .get("nvidia.com/gpu", "0")
    )
    if ready and gpu >= 1:
        count += 1
print(count)
')

  if [ "$COUNT" -ge 1 ]; then
    READY=1
    echo "PASS: GPU node Ready with allocatable GPU"
    break
  fi

  if (( i % 6 == 1 )); then
    echo "Waiting for GPU node: check $i"
  fi

  sleep 10
done

test "$READY" = "1" || {
  echo "ERROR: GPU capacity unavailable"
  exit 1
}

kubectl get nodes -l pool=gpu -o wide

echo "========== EXECUTE CUDA TEST =========="

kubectl apply -f /tmp/v3-gpt2-single-gpu-job.yaml

RESULT=1

for i in $(seq 1 85); do
  SUCCEEDED=$(kubectl -n "$NS" get job "$JOB" \
    -o jsonpath='{.status.succeeded}')

  FAILED=$(kubectl -n "$NS" get job "$JOB" \
    -o jsonpath='{.status.failed}')

  if [ "${SUCCEEDED:-0}" = "1" ]; then
    RESULT=0
    break
  fi

  if [ "${FAILED:-0}" != "0" ]; then
    echo "ERROR: Job failed"
    break
  fi

  if (( i % 6 == 1 )); then
    kubectl -n "$NS" get pods -l "job-name=$JOB" -o wide
  fi

  sleep 10
done

echo "========== SAVE TEST LOGS =========="

kubectl -n "$NS" logs "job/$JOB" \
  --timestamps \
  > "$LOG" 2>&1 || true

cat "$LOG"

if [ "$RESULT" != "0" ]; then
  kubectl -n "$NS" describe job "$JOB" || true
  kubectl -n "$NS" describe pods \
    -l "job-name=$JOB" || true
  exit 1
fi

grep -Fq \
  "PASS: V3.1.6 single-GPU extension validation" \
  "$LOG" || {
    echo "ERROR: Final validation marker missing"
    exit 1
  }

echo "========== SINGLE-GPU VALIDATION PASSED =========="
echo "Saved results: $LOG"
echo "Automatic shutdown follows."
