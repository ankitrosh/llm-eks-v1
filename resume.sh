#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
CLUSTER=llm-pipeline-v1
cd "$(dirname "$0")"

echo "Starting CPU worker..."
eksctl scale nodegroup \
  --cluster "$CLUSTER" \
  --region "$AWS_DEFAULT_REGION" \
  --name cpu-system \
  --nodes 1 --nodes-min 0 --nodes-max 1 --wait

aws eks update-kubeconfig \
  --name "$CLUSTER" --region "$AWS_DEFAULT_REGION"

echo "Starting both A10G GPU workers..."
eksctl scale nodegroup \
  --cluster "$CLUSTER" \
  --region "$AWS_DEFAULT_REGION" \
  --name gpu-a10g \
  --nodes 2 --nodes-min 0 --nodes-max 2 --wait

echo "Restoring application replicas..."
if [ -f .paused-replicas.tsv ]; then
  while IFS=$'\t' read -r kind name count; do
    [ -n "$kind" ] || continue
    if kubectl -n llm get "$kind/$name" >/dev/null 2>&1; then
      kubectl -n llm scale "$kind/$name" --replicas="$count"
    fi
  done < .paused-replicas.tsv

  rm .paused-replicas.tsv
fi

kubectl get nodes -L pool
kubectl get pods -n llm -o wide
