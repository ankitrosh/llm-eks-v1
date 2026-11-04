#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
ACCOUNT=657173565477
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/qwen3-14b-pipeline:v2-sdpa-profile-r1"

aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$REGISTRY"

docker build --progress=plain \
  -f docker/Dockerfile.v2-sdpa-profile \
  -t "$IMAGE" .

docker run --rm --entrypoint python "$IMAGE" \
  -c 'import sys; sys.path.insert(0, "/app"); import qwen3_stage_loader; import profile_v2_sdpa; print("PASS: SDPA profiler imports")'

docker push "$IMAGE"

DIGEST=$(aws ecr describe-images \
  --region "$REGION" \
  --repository-name qwen3-14b-pipeline \
  --image-ids imageTag=v2-sdpa-profile-r1 \
  --query 'imageDetails[0].imageDigest' \
  --output text)

echo "IMAGE_DIGEST=$DIGEST"
