#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
CLUSTER=llm-pipeline-v1
cd "$(dirname "$0")"

aws eks update-kubeconfig \
  --name "$CLUSTER" --region "$AWS_DEFAULT_REGION"

# Save workload replica counts, unless already paused.
if [ ! -f .paused-replicas.tsv ]; then
  kubectl -n llm get deployments,statefulsets -o json |
    jq -r '.items[] | [.kind, .metadata.name, .spec.replicas] | @tsv' \
    > .paused-replicas.tsv
fi

# Stop application workloads before terminating nodes.
while IFS=$'\t' read -r kind name count; do
  [ -n "$kind" ] || continue
  kubectl -n llm scale "$kind/$name" --replicas=0
done < .paused-replicas.tsv

# Wait for inference pods to disappear.
kubectl -n llm wait --for=delete pod \
  -l app=qwen3 --timeout=180s || true

echo "Scaling GPU node group to zero..."
eksctl scale nodegroup \
  --cluster "$CLUSTER" \
  --region "$AWS_DEFAULT_REGION" \
  --name gpu-a10g \
  --nodes 0 --nodes-min 0 --nodes-max 2 --wait

echo "Scaling CPU node group to zero..."
eksctl scale nodegroup \
  --cluster "$CLUSTER" \
  --region "$AWS_DEFAULT_REGION" \
  --name cpu-system \
  --nodes 0 --nodes-min 0 --nodes-max 1 --wait

echo "Requested scale-to-zero for both node groups."
echo "EKS control plane, EBS volumes and ECR storage still cost money."
