#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.." || exit 1

export AWS_PAGER=""

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NS=llm
JOB=v3-gpt2-dist

OUT="infrastructure/profiles/v3-1-6/distributed-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

echo "Results directory: $OUT"

CLEANUP_ARMED=0

cleanup() {
  local result=$?
  trap - EXIT

  if [ "$CLEANUP_ARMED" = "1" ]; then
    echo "========== AUTOMATIC SHUTDOWN =========="
    bash scripts/v3_distributed_cleanup.sh || result=1
  fi

  echo "Final exit code: $result"
  exit "$result"
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

for NG in cpu-system gpu-a10g; do
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

if kubectl -n "$NS" get job "$JOB" >/dev/null 2>&1; then
  echo "ERROR: Distributed Job already exists"
  exit 1
fi

if kubectl -n "$NS" get service v3-gpt2-dist-headless \
  >/dev/null 2>&1; then
  echo "ERROR: Rendezvous Service already exists"
  exit 1
fi

kubectl -n "$NS" get cm \
  v3-gpt2-dist-test \
  v3-gpt2-gpu-test \
  v3-gpt2-framework-src >/dev/null

kubectl apply --dry-run=server \
  -f /tmp/v3-gpt2-distributed-job.yaml >/dev/null

echo "PASS: Distributed test preflight"

CLEANUP_ARMED=1

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
    --output text | tee "$OUT/${ng}-update-id.txt"
}

wait_nodes() {
  local pool="$1"
  local required="$2"

  for i in $(seq 1 90); do
    COUNT=$(kubectl get nodes -l "pool=$pool" -o json |
      python3 -c '
import json,sys
data=json.load(sys.stdin)
count=0
for node in data["items"]:
    ready=any(
        c["type"]=="Ready" and c["status"]=="True"
        for c in node["status"].get("conditions", [])
    )
    gpu=int(node["status"].get("allocatable", {})
            .get("nvidia.com/gpu", "0"))
    if ready and (sys.argv[1]!="gpu" or gpu>=1):
        count+=1
print(count)
' "$pool")

    if [ "$COUNT" -ge "$required" ]; then
      echo "PASS: $pool Ready nodes = $COUNT"
      return 0
    fi

    if (( i % 6 == 1 )); then
      echo "Waiting for $pool capacity: $COUNT/$required"
    fi

    sleep 10
  done

  echo "ERROR: $pool node readiness timeout"
  return 1
}

# Restore Kubernetes system capacity first.
start_group cpu-system 1 1
wait_nodes cpu 1

# Then start two GPU workers.
start_group gpu-a10g 2 2
wait_nodes gpu 2

echo "========== VERIFY GPU PLACEMENT =========="
kubectl get nodes -l pool=gpu -o wide

echo "========== START DISTRIBUTED JOB =========="
kubectl apply -f /tmp/v3-gpt2-distributed-job.yaml

RESULT=1

for i in $(seq 1 180); do
  STATUS=$(kubectl -n "$NS" get job "$JOB" -o json |
    python3 -c '
import json,sys
s=json.load(sys.stdin).get("status",{})
print(str(s.get("succeeded",0)) + ":" +
      str(s.get("failed",0)))
')

  SUCCEEDED=${STATUS%%:*}
  FAILED=${STATUS#*:}

  if [ "$SUCCEEDED" = "2" ]; then
    RESULT=0
    echo "PASS: Both distributed ranks completed"
    break
  fi

  if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: Distributed Job failed"
    break
  fi

  if (( i % 6 == 1 )); then
    echo "Job progress: succeeded=$SUCCEEDED failed=$FAILED"
    kubectl -n "$NS" get pods -l app=v3-gpt2-dist -o wide
  fi

  sleep 5
done

echo "========== SAVE BOTH RANK LOGS =========="

kubectl -n "$NS" get pods \
  -l app=v3-gpt2-dist \
  -o json > "$OUT/pods.json"

python3 - "$OUT/pods.json" <<'PY' > "$OUT/pod-index.tsv"
import json,sys

data=json.load(open(sys.argv[1]))

for pod in data["items"]:
    meta=pod["metadata"]
    index=meta.get("annotations",{}).get(
        "batch.kubernetes.io/job-completion-index"
    )
    if index is None:
        index=meta.get("labels",{}).get(
            "batch.kubernetes.io/job-completion-index"
        )
    if index is not None:
        print(f"{index}\t{meta['name']}")
PY

while IFS=$'\t' read -r INDEX POD; do
  echo "Collecting rank $INDEX from $POD"

  kubectl -n "$NS" logs "pod/$POD" \
    -c runtime --timestamps \
    > "$OUT/rank-${INDEX}.log" 2>&1 || true
done < "$OUT/pod-index.tsv"

for RANK in 0 1; do
  echo "========== RANK $RANK =========="

  if [ -f "$OUT/rank-${RANK}.log" ]; then
    cat "$OUT/rank-${RANK}.log"
  else
    echo "No log found for rank $RANK"
    RESULT=1
  fi
done

if [ "$RESULT" != "0" ]; then
  kubectl -n "$NS" describe job "$JOB" || true
  kubectl -n "$NS" describe pods \
    -l app=v3-gpt2-dist || true
  exit 1
fi

for RANK in 0 1; do
  grep -Fq \
    "PASS: rank $RANK distributed extension validation" \
    "$OUT/rank-${RANK}.log"
done

grep -Fq \
  "PASS: Distributed logits match unsplit reference" \
  "$OUT/rank-0.log"

grep -Fq \
  "PASS: Distributed cached generation" \
  "$OUT/rank-0.log"

echo "========== DISTRIBUTED VALIDATION PASSED =========="
echo "Results saved: $OUT"
echo "Automatic shutdown follows."
