#!/usr/bin/env bash
set -euo pipefail

cd ~/llm-eks-v1

REGION=us-east-2
CLUSTER=llm-pipeline-v1
IMAGE="657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:7a18a04e571a6d74027c145023299df91ae856207838f141e9f15d0b1c8f6e09"

STARTED=0
MODE=""
PF_PID=""

wait_update() {
  local group="$1"
  local update="$2"
  local state

  for i in $(seq 1 120); do
    state=$(aws eks describe-update \
      --region "$REGION" \
      --name "$CLUSTER" \
      --nodegroup-name "$group" \
      --update-id "$update" \
      --query 'update.status' \
      --output text) || return 1

    if [[ "$state" == "Successful" ]]; then
      echo "PASS: $group update complete"
      return 0
    fi

    if [[ "$state" == "Failed" || "$state" == "Cancelled" ]]; then
      echo "FAIL: $group update $state"
      return 1
    fi

    sleep 10
  done

  echo "FAIL: $group update timeout"
  return 1
}

stop_pods() {
  kubectl -n llm scale sts/qwen3 --replicas=0 || return 1

  for i in $(seq 1 60); do
    local remaining

    remaining=$(kubectl -n llm get pods -o name |
      grep -E '^pod/qwen3-[01]$' || true)

    if [[ -z "$remaining" ]]; then
      echo "PASS: Both inference pods stopped"
      return 0
    fi

    sleep 5
  done

  echo "WARNING: Inference pods still present"
  return 1
}

cleanup() {
  local original_status=$?
  trap - EXIT
  set +e

  if [[ -n "$PF_PID" ]]; then
    kill "$PF_PID" 2>/dev/null
    wait "$PF_PID" 2>/dev/null
  fi

  if [[ "$STARTED" != "1" ]]; then
    exit "$original_status"
  fi

  echo
  echo "========== AUTOMATIC AWS SHUTDOWN =========="

  # Capture diagnostics before deleting the pods.
  if [[ -n "$MODE" ]]; then
    for rank in 0 1; do
      kubectl -n llm logs "qwen3-$rank" -c runtime \
        --tail=150 \
        > "benchmarks/sdpa_${MODE}_rank${rank}.log" \
        2>&1
    done
  fi

  stop_pods
  local pods_stopped=$?

  if [[ "$pods_stopped" != "0" ]]; then
    echo "WARNING: Pods did not finish deleting; inspect shutdown."
  fi

  echo "========== REQUEST NODEGROUP SCALE-DOWN =========="

  local gpu_update cpu_update

  gpu_update=$(aws eks update-nodegroup-config \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --nodegroup-name gpu-a10g \
    --scaling-config minSize=0,maxSize=2,desiredSize=0 \
    --query 'update.id' --output text)

  cpu_update=$(aws eks update-nodegroup-config \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --nodegroup-name cpu-system \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 \
    --query 'update.id' --output text)

  if [[ -n "$gpu_update" ]]; then
    wait_update gpu-a10g "$gpu_update"
  fi

  if [[ -n "$cpu_update" ]]; then
    wait_update cpu-system "$cpu_update"
  fi

  echo "========== TERMINATION LIFECYCLE HOOKS =========="

  local shutdown_pass=0

  for attempt in 1 2 3; do
    if python3 scripts/finalize_eks_shutdown.py |
      tee /tmp/qwen3-shutdown-finalize.log &&
      grep -q '^PASS: All EKS worker instances removed$' \
        /tmp/qwen3-shutdown-finalize.log; then
      shutdown_pass=1
      break
    fi

    echo "Shutdown still finalizing: check $attempt"
    sleep 10
  done

  if [[ "$shutdown_pass" == "0" ]]; then
    echo "WARNING: Shutdown not fully verified."
    echo "Run: python3 scripts/finalize_eks_shutdown.py"
  else
    echo "PASS: AWS worker shutdown verified"
  fi

  if [[ "$original_status" != "0" ]]; then
    exit "$original_status"
  fi

  if [[ "$shutdown_pass" != "1" ]]; then
    exit 1
  fi

  exit 0
}

trap cleanup EXIT

echo "========== PREFLIGHT =========="

test "$(kubectl -n llm get sts qwen3 \
  -o jsonpath='{.spec.replicas}')" = "0"

python3 -m py_compile \
  scripts/benchmark_v2_sdpa_e2e.py

test -f scripts/finalize_eks_shutdown.py

# Preserve any earlier comparison output.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

for mode in explicit lower_right; do
  FILE="benchmarks/sdpa_e2e_${mode}.json"
  if [[ -f "$FILE" ]]; then
    mv "$FILE" "${FILE%.json}_${STAMP}.json"
  fi
done

echo "========== BACK UP STATEFULSET =========="

kubectl -n llm get sts qwen3 -o yaml \
  > infrastructure/backups/qwen3-before-sdpa-e2e.yaml

echo "========== SELECT TEST IMAGE =========="

kubectl -n llm set image \
  sts/qwen3 "runtime=$IMAGE"

kubectl -n llm set env sts/qwen3 \
  -c runtime \
  LLM_GPTQ_BACKEND=marlin \
  LLM_ATTENTION_BACKEND=sdpa \
  LLM_PREFIX_CACHE_ENABLED=1 \
  LLM_PREFIX_CACHE_MAX_MIB=256 \
  LLM_PREFIX_CACHE_MAX_ENTRIES=4 \
  LLM_PREFIX_CACHE_MIN_TOKENS=64 \
  LLM_SDPA_PREFIX_MASK=explicit

echo "========== START AWS WORKERS =========="

STARTED=1

CPU_UPDATE=$(aws eks update-nodegroup-config \
  --region "$REGION" \
  --cluster-name "$CLUSTER" \
  --nodegroup-name cpu-system \
  --scaling-config minSize=0,maxSize=1,desiredSize=1 \
  --query 'update.id' --output text)

GPU_UPDATE=$(aws eks update-nodegroup-config \
  --region "$REGION" \
  --cluster-name "$CLUSTER" \
  --nodegroup-name gpu-a10g \
  --scaling-config minSize=0,maxSize=2,desiredSize=2 \
  --query 'update.id' --output text)

wait_update cpu-system "$CPU_UPDATE"
wait_update gpu-a10g "$GPU_UPDATE"

run_mode() {
  MODE="$1"

  echo
  echo "========== ATTENTION MODE: $MODE =========="

  kubectl -n llm set env sts/qwen3 \
    -c runtime \
    LLM_SDPA_PREFIX_MASK="$MODE"

  kubectl -n llm scale sts/qwen3 --replicas=2

  echo "========== WAIT FOR POD OBJECTS =========="

  local pods_found=0

  for i in $(seq 1 120); do
    if kubectl -n llm get pod qwen3-0 qwen3-1       >/dev/null 2>&1; then
      pods_found=1
      break
    fi
    sleep 2
  done

  if [[ "$pods_found" != "1" ]]; then
    echo "FAIL: Inference pods were not created"
    kubectl -n llm get pods -o wide
    return 1
  fi

  echo "========== WAIT FOR BOTH STAGES READY =========="

  if ! kubectl -n llm wait \
    --for=condition=Ready \
    pod/qwen3-0 pod/qwen3-1 \
    --timeout=1200s; then
    kubectl -n llm get pods -o wide
    for rank in 0 1; do
      kubectl -n llm logs "qwen3-$rank" \
        -c runtime --tail=80 || true
    done
    return 1
  fi

  for rank in 0 1; do
    POD="qwen3-$rank"

    echo "----- $POD -----"

    kubectl -n llm exec "$POD" -c runtime -- \
      printenv LLM_SDPA_PREFIX_MASK

    test "$(kubectl -n llm exec "$POD" -c runtime -- \
      printenv LLM_SDPA_PREFIX_MASK)" = "$MODE"

    test "$(kubectl -n llm exec "$POD" -c runtime -- \
      printenv LLM_ATTENTION_BACKEND)" = "sdpa"

    test "$(kubectl -n llm exec "$POD" -c runtime -- \
      printenv LLM_PREFIX_CACHE_ENABLED)" = "1"
  done

  echo "========== CONNECT TO INFERENCE API =========="

  kubectl -n llm port-forward \
    pod/qwen3-0 18081:8000 \
    > "/tmp/qwen3-sdpa-${MODE}-pf.log" 2>&1 &

  PF_PID=$!

  local ready=0

  for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:18081/readyz \
      >/dev/null 2>&1; then
      ready=1
      break
    fi

    sleep 2
  done

  if [[ "$ready" != "1" ]]; then
    cat "/tmp/qwen3-sdpa-${MODE}-pf.log"
    return 1
  fi

  echo "========== RUN BENCHMARK =========="

  python3 -u scripts/benchmark_v2_sdpa_e2e.py \
    --mode "$MODE"

  echo "========== SAVE RANK LOGS =========="

  for rank in 0 1; do
    kubectl -n llm logs "qwen3-$rank" -c runtime \
      --tail=150 \
      > "benchmarks/sdpa_${MODE}_rank${rank}.log"
  done

  kill "$PF_PID" 2>/dev/null || true
  wait "$PF_PID" 2>/dev/null || true
  PF_PID=""

  stop_pods
}

run_mode explicit
run_mode lower_right

echo
echo "========== COMPARISON COMPLETED =========="
echo "The automatic shutdown procedure will now run."
