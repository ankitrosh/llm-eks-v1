#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

REGION="us-east-2"
CLUSTER="llm-pipeline-v1"
OUT="test-results/snapshots/infrastructure"

mkdir -p "$OUT"

echo "=== 1. EKS cluster configuration ==="

aws eks describe-cluster \
  --region "$REGION" \
  --name "$CLUSTER" \
  --query 'cluster.{
    name:name,
    version:version,
    platformVersion:platformVersion,
    roleArn:roleArn,
    vpc:resourcesVpcConfig,
    network:kubernetesNetworkConfig,
    logging:logging
  }' \
  --output json > "$OUT/cluster.json"

echo "=== 2. CPU and GPU node groups ==="

for GROUP in cpu-system gpu-a10g; do
  aws eks describe-nodegroup \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$GROUP" \
    --query 'nodegroup.{
      name:nodegroupName,
      instanceTypes:instanceTypes,
      amiType:amiType,
      capacityType:capacityType,
      diskSize:diskSize,
      subnets:subnets,
      labels:labels,
      scaling:scalingConfig,
      nodeRole:nodeRole,
      launchTemplate:launchTemplate,
      remoteAccess:remoteAccess,
      version:version,
      releaseVersion:releaseVersion,
      health:health
    }' \
    --output json > "$OUT/$GROUP.json"
done

echo "=== 3. EKS addons ==="

aws eks list-addons \
  --region "$REGION" \
  --cluster-name "$CLUSTER" \
  --query 'addons' \
  --output text | tr '\t' '\n' \
  > "$OUT/addon-names.txt"

while IFS= read -r ADDON; do
  [[ -z "$ADDON" ]] && continue

  aws eks describe-addon \
    --region "$REGION" \
    --cluster-name "$CLUSTER" \
    --addon-name "$ADDON" \
    --query 'addon.{
      name:addonName,
      version:addonVersion,
      status:status,
      roleArn:serviceAccountRoleArn,
      podIdentity:podIdentityAssociations,
      configuration:configurationValues
    }' \
    --output json > "$OUT/addon-$ADDON.json"
done < "$OUT/addon-names.txt"

echo "=== 4. Pod Identity associations ==="

if ! aws eks list-pod-identity-associations \
  --region "$REGION" \
  --cluster-name "$CLUSTER" \
  --output json > "$OUT/pod-identity.json"; then
  echo "WARNING: Could not inspect Pod Identity associations"
fi

echo "=== 5. GPU plugin and Kubernetes system components ==="

kubectl get daemonsets -A -o json \
  > "$OUT/daemonsets.json"

kubectl get nodes -L pool -o wide \
  > "$OUT/nodes.txt"

echo "=== 6. Existing network configuration ==="

VPC_ID=$(python3 - "$OUT/cluster.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f)["vpc"]["vpcId"])
PY
)

if ! aws ec2 describe-subnets \
  --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[].{
    id:SubnetId,
    az:AvailabilityZone,
    cidr:CidrBlock,
    publicIp:MapPublicIpOnLaunch,
    availableIps:AvailableIpAddressCount
  }' \
  --output json > "$OUT/subnets.json"; then
  echo "WARNING: Could not inspect VPC subnets"
fi

if ! aws ec2 describe-route-tables \
  --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'RouteTables[].{
    id:RouteTableId,
    associations:Associations,
    routes:Routes
  }' \
  --output json > "$OUT/route-tables.json"; then
  echo "WARNING: Could not inspect VPC routing"
fi

echo
echo "CAPTURE COMPLETE"
echo "Saved snapshots under: $OUT"
echo "No infrastructure resources were changed."
