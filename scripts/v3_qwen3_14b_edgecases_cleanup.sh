#!/usr/bin/env bash
set -Eeuo pipefail

CLUSTER=llm-pipeline-v1
REGION=us-east-2
NS=llm
NAME=v3-qwen3-14b-edgecases
EXPECTED="arn:aws:eks:us-east-2:657173565477:cluster/llm-pipeline-v1"
export AWS_PAGER=""

if [[ "$(kubectl config current-context)" != "$EXPECTED" ]]; then
  echo "ERROR: Wrong Kubernetes context; refusing cleanup"
  exit 1
fi

FAILED=0

echo "========== REMOVE BENCHMARK JOBS =========="
kubectl -n "$NS" delete job "${NAME}-rank0" "${NAME}-rank1" \
  --ignore-not-found --wait=true --timeout=120s || FAILED=1

kubectl -n "$NS" delete service "${NAME}-headless" \
  --ignore-not-found --wait=false || FAILED=1

kubectl -n "$NS" delete configmap "${NAME}-code" \
  --ignore-not-found --wait=false || FAILED=1

scale_down() {
  local ng="$1" max="$2" status desired

  for attempt in $(seq 1 45); do
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

    if [[ "$status" == "ACTIVE" ]]; then
      if [[ "$desired" == "0" ]]; then
        echo "PASS: $ng already at desired capacity 0"
        return 0
      fi

      if aws eks update-nodegroup-config \
        --cluster-name "$CLUSTER" \
        --nodegroup-name "$ng" \
        --region "$REGION" \
        --scaling-config "minSize=0,maxSize=${max},desiredSize=0" \
        --query 'update.id' \
        --output text; then
        echo "PASS: Scale-down requested for $ng"
        return 0
      fi
    fi

    echo "Waiting: $ng status=$status desired=$desired"
    sleep 10
  done

  echo "ERROR: Could not scale down $ng"
  return 1
}

echo "========== SCALE DOWN WORKERS =========="
scale_down gpu-a10g 2 || FAILED=1
scale_down cpu-system 1 || FAILED=1

echo "========== VERIFY ACTUAL ASG CAPACITY =========="

for attempt in $(seq 1 40); do
  REMAINING=0

  for ng in gpu-a10g cpu-system; do
    asg=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$ng" \
      --region "$REGION" \
      --query 'nodegroup.resources.autoScalingGroups[0].name' \
      --output text) || {
        FAILED=1
        REMAINING=1
        continue
      }

    state=$(aws autoscaling describe-auto-scaling-groups \
      --auto-scaling-group-names "$asg" \
      --region "$REGION" \
      --query 'AutoScalingGroups[0].[DesiredCapacity,length(Instances)]' \
      --output text) || {
        FAILED=1
        REMAINING=1
        continue
      }

    read -r capacity count <<< "$state"

    desired=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$ng" \
      --region "$REGION" \
      --query 'nodegroup.scalingConfig.desiredSize' \
      --output text) || {
        FAILED=1
        REMAINING=1
        continue
      }

    echo "$ng: EKS desired=$desired ASG desired=$capacity instances=$count"

    if [[ "$desired" != "0" || "$capacity" != "0" || "$count" != "0" ]]; then
      REMAINING=1
    fi
  done

  if [[ "$REMAINING" == "0" ]]; then
    echo "PASS: Both worker ASGs have zero capacity and instances"
    exit "$FAILED"
  fi

  sleep 15
done

echo "WARNING: Worker instances may still be terminating."
echo "Inspect lifecycle hooks before manually completing any actions."

for ng in gpu-a10g cpu-system; do
  asg=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$ng" \
    --region "$REGION" \
    --query 'nodegroup.resources.autoScalingGroups[0].name' \
    --output text) || continue

  aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names "$asg" \
    --region "$REGION" \
    --query 'AutoScalingGroups[0].Instances[*].[InstanceId,LifecycleState]' \
    --output table || true
done

exit 1
