#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

REGION="us-east-2"
CLUSTER="llm-pipeline-v1"
OUT="test-results/snapshots/infrastructure"

mkdir -p "$OUT"

echo "=== EBS POD IDENTITY ASSOCIATION ==="

for ID in $(aws eks list-pod-identity-associations \
  --cluster-name "$CLUSTER" \
  --region "$REGION" \
  --query 'associations[].associationId' \
  --output text); do

  aws eks describe-pod-identity-association \
    --cluster-name "$CLUSTER" \
    --association-id "$ID" \
    --region "$REGION" \
    --query 'association.{
      namespace:namespace,
      serviceAccount:serviceAccount,
      roleArn:roleArn,
      associationId:associationId
    }' \
    --output json

done

echo
echo "=== NODE LAUNCH TEMPLATES ==="

for GROUP in cpu-system gpu-a10g; do
  LT_ID=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$GROUP" \
    --region "$REGION" \
    --query 'nodegroup.launchTemplate.id' \
    --output text)

  LT_VERSION=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$GROUP" \
    --region "$REGION" \
    --query 'nodegroup.launchTemplate.version' \
    --output text)

  echo "Group: $GROUP"
  echo "Template: $LT_ID"
  echo "Version: $LT_VERSION"

  aws ec2 describe-launch-template-versions \
    --launch-template-id "$LT_ID" \
    --versions "$LT_VERSION" \
    --region "$REGION" \
    --query 'LaunchTemplateVersions[0].{
      version:VersionNumber,
      image:LaunchTemplateData.ImageId,
      instanceType:LaunchTemplateData.InstanceType,
      storage:LaunchTemplateData.BlockDeviceMappings,
      metadata:LaunchTemplateData.MetadataOptions,
      securityGroups:LaunchTemplateData.SecurityGroupIds
    }' \
    --output json | tee "$OUT/launch-$GROUP.json"
done

echo
echo "=== SUBNET DETAILS ==="

python3 - <<'PY'
import json
from pathlib import Path

root = Path("test-results/snapshots/infrastructure")

subnets = json.loads((root / "subnets.json").read_text())

for subnet in subnets:
    print(
        subnet["id"],
        "AZ:", subnet["az"],
        "Public IP:", subnet["publicIp"],
        "CIDR:", subnet["cidr"]
    )
PY

echo
echo "Infrastructure detail capture completed."
