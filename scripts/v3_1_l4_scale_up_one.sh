#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
export AWS_PAGER=""

MODE="${1:---check}"

[[ "$MODE" == "--check" || "$MODE" == "--execute" ]] || {
  echo "Usage: $0 [--check|--execute]"
  exit 2
}

ACCOUNT=$(aws sts get-caller-identity \
  --query Account --output text)

[[ "$ACCOUNT" == "657173565477" ]] || {
  echo "ERROR: Unexpected AWS account"
  exit 1
}

CLUSTER=llm-pipeline-v1
NODEGROUP=gpu-l4-small

STATUS=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.status' \
  --output text)

[[ "$STATUS" == "ACTIVE" ]] || {
  echo "ERROR: Nodegroup not ACTIVE: $STATUS"
  exit 1
}

DESIRED=$(aws eks describe-nodegroup \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODEGROUP" \
  --query 'nodegroup.scalingConfig.desiredSize' \
  --output text)

echo "Current desired capacity: $DESIRED"

if [[ "$MODE" == "--execute" ]]; then
  [[ "$DESIRED" == "0" ]] || {
    echo "ERROR: Expected desired capacity zero"
    exit 1
  }

  echo "========== REQUEST ONE GPU INSTANCE =========="

  aws eks update-nodegroup-config \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NODEGROUP" \
    --scaling-config \
      minSize=0,maxSize=3,desiredSize=1 \
    --query 'update.{Id:id,Status:status}' \
    --output json
else
  echo "Read-only mode; no instances requested"
fi
