#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
ACCOUNT=657173565477
REPO=qwen3-14b-pipeline
TAG=v2-qwen3-marlin-r1

REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$REGISTRY"

echo "========== BUILD MARLIN CANDIDATE =========="

docker build \
  --progress=plain \
  -f docker/Dockerfile.v2-marlin \
  -t "$IMAGE" \
  .

echo "========== VERIFY MARLIN IMPORT =========="

docker run --rm \
  --entrypoint python \
  "$IMAGE" \
  -c 'import torch; import gptqmodel_marlin_kernels; print("PASS: Marlin import")'

echo "========== PUSH IMAGE =========="

docker push "$IMAGE"

echo "========== IMAGE DIGEST =========="

aws ecr describe-images \
  --region "$REGION" \
  --repository-name "$REPO" \
  --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].imageDigest' \
  --output text
