#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER=llm-pipeline-v1
REGION=us-east-2
NS=llm
NAME=v3-qwen3-14b-benchmark
EXPECTED="arn:aws:eks:us-east-2:657173565477:cluster/llm-pipeline-v1"

MANIFEST=deploy/manifests/v3-qwen3-14b-benchmark.yaml
CONFIGMAP=/tmp/v3-qwen3-14b-benchmark-code-configmap.yaml
OUT="run-logs/v3-14b/$(date -u +%Y%m%dT%H%M%SZ)"

export AWS_PAGER=""
export AWS_DEFAULT_REGION="$REGION"

mkdir -p "$OUT"
ARMED=0

save_logs() {
  kubectl -n "$NS" get jobs "${NAME}-rank0" "${NAME}-rank1" \
    -o yaml > "$OUT/jobs.yaml" 2>"$OUT/jobs-error.txt" || true

  kubectl -n "$NS" get pods -l "app=$NAME" \
    -o wide > "$OUT/pods.txt" 2>&1 || true

  kubectl -n "$NS" describe pods -l "app=$NAME" \
    > "$OUT/pods-describe.txt" 2>&1 || true

  for rank in 0 1; do
    kubectl -n "$NS" logs "job/${NAME}-rank${rank}" \
      --timestamps=true --all-containers=true \
      > "$OUT/rank${rank}.log" \
      2> "$OUT/rank${rank}-error.txt" || true
  done
}

finish() {
  local result=$?
  trap - EXIT INT TERM
  set +e

  if [[ "$ARMED" == "1" ]]; then
    echo "========== SAVE LOGS =========="
    save_logs

    echo "========== AUTOMATIC CLEANUP =========="
    bash scripts/v3_qwen3_14b_cleanup.sh || result=1
  fi

  echo "Log directory: $OUT"
  echo "Final exit code: $result"
  exit "$result"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "========== PREFLIGHT =========="

[[ "$(kubectl config current-context)" == "$EXPECTED" ]] || {
  echo "ERROR: Wrong Kubernetes context"
  exit 1
}

[[ -f "$MANIFEST" && -f "$CONFIGMAP" ]] || {
  echo "ERROR: Missing manifest or ConfigMap"
  exit 1
}

[[ "$(kubectl -n "$NS" get sts qwen3 \
  -o jsonpath='{.spec.replicas}')" == "0" ]] || {
  echo "ERROR: Original StatefulSet must be at zero replicas"
  exit 1
}

for rank in 0 1; do
  [[ "$(kubectl -n "$NS" get pvc "model-files-qwen3-${rank}" \
    -o jsonpath='{.status.phase}')" == "Bound" ]] || {
    echo "ERROR: Model PVC $rank is not Bound"
    exit 1
  }

  if kubectl -n "$NS" get job "${NAME}-rank${rank}" \
    >/dev/null 2>&1; then
    echo "ERROR: Existing benchmark Job for rank $rank"
    exit 1
  fi
done

for ng in cpu-system gpu-a10g; do
  [[ "$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --query 'nodegroup.status' \
    --output text)" == "ACTIVE" ]] || {
    echo "ERROR: Nodegroup $ng is not ACTIVE"
    exit 1
  }

  [[ "$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)" == "0" ]] || {
    echo "ERROR: Nodegroup $ng must start at desired capacity 0"
    exit 1
  }
done

kubectl apply --dry-run=server \
  -f "$CONFIGMAP" \
  -f "$MANIFEST"

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
  local count

  for i in $(seq 1 120); do
    count=$(kubectl get nodes -l "pool=$pool" -o json |
      python3 -c '
import json, sys
pool = sys.argv[1]
nodes = json.load(sys.stdin)["items"]

def available(node):
    status = node.get("status", {})
    ready = any(
        c.get("type") == "Ready" and c.get("status") == "True"
        for c in status.get("conditions", [])
    )
    gpu = int(
        status.get("allocatable", {}).get("nvidia.com/gpu", "0")
    )
    return ready and (pool != "gpu" or gpu >= 1)

print(sum(map(available, nodes)))
' "$pool")

    if (( count >= required )); then
      echo "PASS: $pool nodes ready = $count"
      return 0
    fi

    if (( i % 6 == 1 )); then
      echo "Waiting for $pool nodes: $count/$required"
    fi

    if [[ "$pool" == "gpu" ]] &&
       (( i % 36 == 1 )) &&
       (( i > 1 )); then

      asg=$(aws eks describe-nodegroup \
        --cluster-name "$CLUSTER" \
        --nodegroup-name gpu-a10g \
        --region "$REGION" \
        --query 'nodegroup.resources.autoScalingGroups[0].name' \
        --output text)

      aws autoscaling describe-scaling-activities \
        --auto-scaling-group-name "$asg" \
        --region "$REGION" \
        --max-items 4 \
        --query 'Activities[*].[StatusCode,StatusMessage]' \
        --output text || true
    fi

    sleep 10
  done

  echo "ERROR: $pool node readiness timeout"
  return 1
}

# All preflight checks passed.
# Enable automatic cleanup before starting AWS workers.
ARMED=1

echo "========== APPLY CONFIGMAP =========="
kubectl apply -f "$CONFIGMAP"

start_group cpu-system 1 1
wait_nodes cpu 1

start_group gpu-a10g 2 2
wait_nodes gpu 2

echo "========== CREATE BENCHMARK JOBS =========="
kubectl apply -f "$MANIFEST"

echo "========== WAIT FOR BOTH RANKS =========="

for i in $(seq 1 300); do
  s0=$(kubectl -n "$NS" get job "${NAME}-rank0" \
    -o jsonpath='{.status.succeeded}')
  s1=$(kubectl -n "$NS" get job "${NAME}-rank1" \
    -o jsonpath='{.status.succeeded}')

  f0=$(kubectl -n "$NS" get job "${NAME}-rank0" \
    -o jsonpath='{.status.failed}')
  f1=$(kubectl -n "$NS" get job "${NAME}-rank1" \
    -o jsonpath='{.status.failed}')

  s0=${s0:-0}
  s1=${s1:-0}
  f0=${f0:-0}
  f1=${f1:-0}

  if (( f0 > 0 || f1 > 0 )); then
    echo "ERROR: Rank Job failed: rank0=$f0 rank1=$f1"
    exit 1
  fi

  if (( s0 == 1 && s1 == 1 )); then
    echo "PASS: Both rank Jobs completed successfully"
    exit 0
  fi

  if (( i % 6 == 1 )); then
    echo "Waiting: rank0=$s0 rank1=$s1"

    kubectl -n "$NS" get pods \
      -l "app=$NAME" -o wide || true
  fi

  sleep 10
done

echo "ERROR: Benchmark Jobs timed out"
exit 1
