#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
export AWS_PAGER=""

CLUSTER="llm-pipeline-v1"
NODEGROUP="gpu-l4-small"
MODE="${1:---check}"

if [[ "$MODE" != "--check" && "$MODE" != "--execute" ]]; then
  echo "Usage: $0 [--check|--execute]"
  exit 2
fi

ACCOUNT=$(aws sts get-caller-identity \
  --query Account --output text)

[[ "$ACCOUNT" == "657173565477" ]] || {
  echo "ERROR: Unexpected AWS account"
  exit 1
}

MAX=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.scalingConfig.maxSize' \
  --output text)

DESIRED=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.scalingConfig.desiredSize' \
  --output text)

echo "Nodegroup: $NODEGROUP"
echo "Current desired capacity: $DESIRED"

if [[ "$MODE" == "--execute" && "$DESIRED" != "0" ]]; then
  echo "========== REQUEST SCALE DOWN =========="

  aws eks update-nodegroup-config \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NODEGROUP" \
    --scaling-config \
      "minSize=0,maxSize=$MAX,desiredSize=0" \
    --query 'update.{Id:id,Status:status}' \
    --output json

  echo "Scale-down request submitted."
fi

echo "========== NODEGROUP STATUS =========="

aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.{Status:status,Scaling:scalingConfig,Health:health.issues}' \
  --output json

ASG=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.resources.autoScalingGroups[0].name' \
  --output text)

if [[ "$ASG" != "None" && -n "$ASG" ]]; then
  echo "========== AUTO SCALING INSTANCES =========="

  aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names "$ASG" \
    --query \
      'AutoScalingGroups[].{Name:AutoScalingGroupName,Desired:DesiredCapacity,Instances:Instances[].{Id:InstanceId,State:LifecycleState}}' \
    --output json

  echo "========== LIFECYCLE HOOKS =========="

  aws autoscaling describe-lifecycle-hooks \
    --auto-scaling-group-name "$ASG" \
    --query \
      'LifecycleHooks[].{Name:LifecycleHookName,Transition:LifecycleTransition,Timeout:HeartbeatTimeout}' \
    --output json
fi

echo "NOTE: ASG instances may remain during termination."
echo "Verify Instances=[] before treating cleanup as complete."
