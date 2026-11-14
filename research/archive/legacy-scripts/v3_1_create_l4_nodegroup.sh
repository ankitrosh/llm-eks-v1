#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
export AWS_PAGER=""

CLUSTER=llm-pipeline-v1
NODEGROUP=gpu-l4-small
SUBNET=subnet-0bed0e4744fc96547

echo "========== VERIFY AWS ACCOUNT =========="

ACCOUNT=$(aws sts get-caller-identity \
  --query Account --output text)

[[ "$ACCOUNT" == "657173565477" ]] || {
  echo "ERROR: Unexpected AWS account"
  exit 1
}

echo "========== VERIFY EXISTING NODE ROLE =========="

ROLE=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name gpu-a10g \
  --query 'nodegroup.nodeRole' \
  --output text)

echo "Reusing node IAM role: $ROLE"

echo "========== CHECK FOR EXISTING L4 NODEGROUP =========="

if aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  >/dev/null 2>&1; then

  echo "Nodegroup already exists"
else
  echo "========== CREATE L4 NODEGROUP =========="

  aws eks create-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NODEGROUP" \
    --node-role "$ROLE" \
    --subnets "$SUBNET" \
    --instance-types g6f.large \
    --ami-type AL2023_x86_64_NVIDIA \
    --capacity-type ON_DEMAND \
    --disk-size 60 \
    --scaling-config minSize=0,maxSize=3,desiredSize=0 \
    --labels pool=gpu-l4 \
    --query 'nodegroup.{Name:nodegroupName,Status:status,Scaling:scalingConfig}' \
    --output json
fi

echo "========== VERIFY CONFIGURATION =========="

aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.{Name:nodegroupName,Status:status,Types:instanceTypes,AmiType:amiType,Scaling:scalingConfig,Subnets:subnets,Health:health.issues}' \
  --output json

echo "PASS: Nodegroup creation/check requested"
echo "No GPU scale-up was requested"
