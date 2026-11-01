#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:---check}"
CLUSTER="llm-pipeline-v1"
REGION="us-east-2"
EBS_ROLE="arn:aws:iam::657173565477:role/llm-v1-ebs-csi-role"

if [[ "$MODE" != "--check" && "$MODE" != "--apply" ]]; then
  echo "Usage: infrastructure/setup-addons.sh [--check|--apply]"
  exit 2
fi

echo "=== EKS addon setup: $MODE ==="

EXPECTED_CONTEXT=$(aws eks describe-cluster \
  --region "$REGION" \
  --name "$CLUSTER" \
  --query 'cluster.arn' --output text)

CURRENT_CONTEXT=$(kubectl config current-context)

if [[ "$CURRENT_CONTEXT" != "$EXPECTED_CONTEXT" ]]; then
  echo "ERROR: Kubernetes context does not match target cluster"
  exit 1
fi

if [[ "$MODE" == "--apply" &&
      "${CONFIRM_ADDON_APPLY:-}" != "YES" ]]; then
  echo "Refusing to modify cluster."
  echo "Set CONFIRM_ADDON_APPLY=YES for a fresh deployment."
  exit 1
fi

check_addon() {
  local name="$1"
  local expected="$2"
  local actual status

  actual=$(aws eks describe-addon \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --addon-name "$name" \
    --query 'addon.addonVersion' \
    --output text)

  status=$(aws eks describe-addon \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --addon-name "$name" \
    --query 'addon.status' \
    --output text)

  echo "$name: $actual ($status)"

  if [[ "$actual" != "$expected" || "$status" != "ACTIVE" ]]; then
    echo "ERROR: Addon differs from pinned configuration"
    exit 1
  fi
}

echo
echo "=== Core addons ==="

check_addon vpc-cni v1.22.4-eksbuild.10
check_addon kube-proxy v1.36.0-eksbuild.58
check_addon coredns v1.14.7-eksbuild.22
check_addon eks-pod-identity-agent v1.4.1-eksbuild.3

echo
echo "=== EBS CSI ==="

if aws eks describe-addon \
  --region "$REGION" \
  --cluster-name "$CLUSTER" \
  --addon-name aws-ebs-csi-driver \
  >/dev/null 2>&1; then

  echo "EBS CSI addon already exists"

elif [[ "$MODE" == "--apply" ]]; then

  aws eks create-addon \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --addon-name aws-ebs-csi-driver \
    --addon-version v1.67.2-eksbuild.1 \
    --pod-identity-associations \
      "serviceAccount=ebs-csi-controller-sa,roleArn=$EBS_ROLE"

  aws eks wait addon-active \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --addon-name aws-ebs-csi-driver

else
  echo "ERROR: EBS CSI addon missing"
  exit 1
fi

check_addon aws-ebs-csi-driver v1.67.2-eksbuild.1

echo
echo "=== NVIDIA device plugin ==="

if [[ "$MODE" == "--apply" ]]; then
  kubectl apply -f infrastructure/nvidia-device-plugin.yaml
else
  kubectl apply --dry-run=client \
    -f infrastructure/nvidia-device-plugin.yaml
fi

kubectl get daemonset nvidia-device-plugin-daemonset \
  -n kube-system -o wide

echo
echo "=== Storage driver ==="

kubectl get csidriver ebs.csi.aws.com

echo
echo "=== GPU availability ==="

kubectl get nodes -L pool \
  -o custom-columns='NAME:.metadata.name,POOL:.metadata.labels.pool,GPU:.status.allocatable.nvidia\.com/gpu'

echo
echo "Addon verification completed."
