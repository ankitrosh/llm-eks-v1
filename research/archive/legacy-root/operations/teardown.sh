#!/usr/bin/env bash
set -euo pipefail

export AWS_DEFAULT_REGION=us-east-2
CLUSTER=llm-pipeline-v1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
cd "$(dirname "$0")"

echo "This will permanently delete:"
echo " - EKS cluster and both node groups"
echo " - Namespace llm and its model-cache PVCs"
echo " - Kubernetes-managed EBS model volumes"
echo " - EBS CSI IAM role, if permitted"
echo
echo "The ECR image and local YAML files will be retained."
read -r -p "Type DELETE to continue: " answer
[ "$answer" = "DELETE" ] || exit 1

# Avoid deleting model data if the known OIDC permission
# restriction will prevent cluster teardown.
ISSUER=$(aws eks describe-cluster \
  --name "$CLUSTER" \
  --query 'cluster.identity.oidc.issuer' \
  --output text)

OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${ISSUER#https://}"

echo "Checking OIDC permissions: $OIDC_ARN"

if ! aws iam get-open-id-connect-provider \
  --open-id-connect-provider-arn "$OIDC_ARN" \
  >/dev/null 2>oidc-preflight-error.log; then

  if ! grep -q 'NoSuchEntity' oidc-preflight-error.log; then
    cat oidc-preflight-error.log
    echo
    echo "STOP: OIDC IAM access is blocked."
    echo "No resources have been deleted by this script."
    echo "Use ./pause.sh to reduce costs."
    exit 2
  fi
fi

aws eks update-kubeconfig \
  --name "$CLUSTER" --region "$AWS_DEFAULT_REGION"

echo "Deleting application workloads..."
kubectl -n llm delete deployments,statefulsets,daemonsets \
  --all --ignore-not-found --wait=true

echo "Recording model volume identifiers..."
kubectl get pv -o json |
  jq -r '.items[] | select(.spec.claimRef.namespace == "llm") |
         .metadata.name' > .teardown-pvs.txt

echo "Deleting application PVCs..."
kubectl -n llm delete pvc --all \
  --ignore-not-found --wait=true

# Wait for EBS CSI PV cleanup while the driver is running.
while read -r pv; do
  [ -n "$pv" ] || continue
  kubectl wait --for=delete "pv/$pv" --timeout=600s
done < .teardown-pvs.txt

kubectl delete namespace llm \
  --ignore-not-found --wait=true

kubectl delete storageclass llm-gp3 \
  --ignore-not-found

echo "Deleting EKS resources through eksctl..."
eksctl delete cluster \
  --region "$AWS_DEFAULT_REGION" \
  --name "$CLUSTER" \
  --disable-nodegroup-eviction \
  --wait

echo "Removing the project EBS CSI IAM role..."
aws iam detach-role-policy \
  --role-name llm-v1-ebs-csi-role \
  --policy-arn \
    arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy

aws iam delete-role \
  --role-name llm-v1-ebs-csi-role

echo
echo "Teardown commands completed."
echo "Verify that the EKS cluster, project volumes,"
echo "load balancers and CloudFormation stacks are gone."
