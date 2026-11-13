#!/usr/bin/env bash
set -u

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NS=llm

echo "========== DELETE NCCL TEST RESOURCES =========="

kubectl -n "$NS" delete job v3-qwen3-stageonly-nccl \
  --ignore-not-found \
  --wait=true \
  --timeout=120s || true

kubectl -n "$NS" delete service v3-qwen3-stageonly-nccl-headless \
  --ignore-not-found || true

kubectl -n "$NS" delete configmap v3-qwen3-stageonly-nccl-code \
  --ignore-not-found || true

scale_down() {
  local ng="$1"
  local max="$2"

  echo "========== SCALE DOWN $ng =========="

  for i in $(seq 1 60); do
    local status
    local desired

    status=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$ng" \
      --region "$REGION" \
      --query 'nodegroup.status' \
      --output text) || return 1

    desired=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$ng" \
      --region "$REGION" \
      --query 'nodegroup.scalingConfig.desiredSize' \
      --output text) || return 1

    if [[ "$status" == ACTIVE ]]; then
      if [[ "$desired" == 0 ]]; then
        echo "PASS: $ng already desiredSize=0"
        return 0
      fi

      if aws eks update-nodegroup-config \
        --cluster-name "$CLUSTER" \
        --nodegroup-name "$ng" \
        --region "$REGION" \
        --scaling-config \
          "minSize=0,maxSize=${max},desiredSize=0" \
        --query 'update.id' \
        --output text; then
        echo "PASS: Scale-down requested for $ng"
        return 0
      fi
    fi

    sleep 10
  done

  echo "ERROR: Could not scale down $ng"
  return 1
}

FAILED=0

scale_down gpu-a10g 2 || FAILED=1
scale_down cpu-system 1 || FAILED=1

echo "========== FINAL NODEGROUP STATE =========="

for ng in gpu-a10g cpu-system; do
  aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --query 'nodegroup.{Status:status,Scaling:scalingConfig}' \
    --output yaml || FAILED=1
done

exit "$FAILED"
