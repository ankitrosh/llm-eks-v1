#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
export AWS_PAGER=""

CLUSTER="llm-pipeline-v1"
NAMESPACE="llm"
NAME="v3-byom-qwen3-smoke"
EXPECTED_ACCOUNT="657173565477"

ACCOUNT=$(aws sts get-caller-identity \
  --query Account --output text)

[[ "$ACCOUNT" == "$EXPECTED_ACCOUNT" ]] || {
  echo "ERROR: Unexpected AWS account $ACCOUNT"
  exit 1
}

CONTEXT=$(kubectl config current-context)

[[ "$CONTEXT" == *":cluster/$CLUSTER" ]] || {
  echo "ERROR: Wrong Kubernetes context: $CONTEXT"
  exit 1
}

echo "========== DELETE BYOM RESOURCES =========="

kubectl -n "$NAMESPACE" delete jobs.batch \
  "$NAME-rank0" "$NAME-rank1" \
  --ignore-not-found --wait=false

kubectl -n "$NAMESPACE" delete service \
  "$NAME-headless" \
  --ignore-not-found --wait=false

kubectl -n "$NAMESPACE" delete configmap \
  "$NAME-code" \
  --ignore-not-found --wait=false

echo "========== REQUEST WORKER SCALE DOWN =========="

for NG in gpu-a10g cpu-system; do
  DESIRED=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)

  if [[ "$DESIRED" == "0" ]]; then
    echo "PASS: $NG already desired=0"
    continue
  fi

  MIN=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --query 'nodegroup.scalingConfig.minSize' \
    --output text)

  MAX=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --query 'nodegroup.scalingConfig.maxSize' \
    --output text)

  STATUS=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --query 'nodegroup.status' \
    --output text)

  if [[ "$STATUS" == "ACTIVE" ]]; then
    aws eks update-nodegroup-config \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$NG" \
      --scaling-config \
        "minSize=$MIN,maxSize=$MAX,desiredSize=0" \
      --query 'update.id' \
      --output text

    echo "PASS: Scale-down requested for $NG"
  elif [[ "$STATUS" == "UPDATING" ]]; then
    echo "INFO: $NG update already in progress"
  else
    echo "ERROR: $NG has unexpected status $STATUS"
    exit 1
  fi
done

echo "========== FINALIZE TERMINATION HOOKS =========="

# Bounded checks. No infinite shutdown loop.
for ATTEMPT in $(seq 1 36); do
  ALL_CLEAN=1

  for NG in gpu-a10g cpu-system; do
    ASG=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$NG" \
      --query \
        'nodegroup.resources.autoScalingGroups[0].name' \
      --output text)

    EKS_DESIRED=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$NG" \
      --query 'nodegroup.scalingConfig.desiredSize' \
      --output text)

    ASG_DESIRED=$(aws autoscaling \
      describe-auto-scaling-groups \
      --auto-scaling-group-names "$ASG" \
      --query \
        'AutoScalingGroups[0].DesiredCapacity' \
      --output text)

    COUNT=$(aws autoscaling \
      describe-auto-scaling-groups \
      --auto-scaling-group-names "$ASG" \
      --query \
        'length(AutoScalingGroups[0].Instances)' \
      --output text)

    echo "$NG: EKS=$EKS_DESIRED ASG=$ASG_DESIRED instances=$COUNT"

    if [[ "$EKS_DESIRED" != "0" || "$ASG_DESIRED" != "0" ]]; then
      ALL_CLEAN=0
      continue
    fi

    # Termination hooks are completed only after both
    # EKS and Auto Scaling desire zero instances.
    WAITING=$(aws autoscaling \
      describe-auto-scaling-groups \
      --auto-scaling-group-names "$ASG" \
      --query \
        "AutoScalingGroups[0].Instances[?LifecycleState=='Terminating:Wait'].InstanceId" \
      --output text)

    if [[ -n "$WAITING" && "$WAITING" != "None" ]]; then
      TRANSITION=$(aws autoscaling describe-lifecycle-hooks \
        --auto-scaling-group-name "$ASG" \
        --lifecycle-hook-names Terminate-LC-Hook \
        --query 'LifecycleHooks[0].LifecycleTransition' \
        --output text)

      [[ "$TRANSITION" == "autoscaling:EC2_INSTANCE_TERMINATING" ]] || {
        echo "ERROR: Unexpected lifecycle hook for $NG"
        exit 1
      }

      for INSTANCE in $WAITING; do
        echo "Completing termination: $NG / $INSTANCE"

        aws autoscaling complete-lifecycle-action \
          --lifecycle-hook-name Terminate-LC-Hook \
          --auto-scaling-group-name "$ASG" \
          --lifecycle-action-result CONTINUE \
          --instance-id "$INSTANCE"
      done
    fi

    if [[ "$COUNT" != "0" ]]; then
      ALL_CLEAN=0
    fi
  done

  if [[ "$ALL_CLEAN" == "1" ]]; then
    echo "PASS: Both worker Auto Scaling groups empty"
    exit 0
  fi

  if [[ "$ATTEMPT" -lt 36 ]]; then
    sleep 10
  fi
done

echo "WARNING: Scale-down not fully complete after bounded checks"
echo "Inspect remaining lifecycle states before retrying"
exit 1
