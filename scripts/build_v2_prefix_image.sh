#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
ACCOUNT=657173565477
REPO=qwen3-14b-pipeline
TAG=v2-prefix-r1

REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$REGISTRY"

echo "========== BUILD =========="

docker build \
  --progress=plain \
  -f docker/Dockerfile.v2-prefix \
  -t "$IMAGE" \
  .

echo "========== VERIFY IMAGE =========="

docker run --rm \
  --entrypoint python \
  "$IMAGE" \
  -c 'import sys; sys.path.insert(0, "/app"); import qwen3_serve; import prefix_cache; import prefix_kv; import gptqmodel_marlin_kernels; print("PASS: Prefix cache and Marlin imports")'

echo "========== PUSH =========="

docker push "$IMAGE"

echo "========== DIGEST =========="

aws ecr describe-images \
  --region "$REGION" \
  --repository-name "$REPO" \
  --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].imageDigest' \
  --output text
