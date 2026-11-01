#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

REGION="us-east-2"
ACCOUNT="657173565477"
REPOSITORY="qwen3-14b-pipeline"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

TAG="${IMAGE_TAG:-v1-rebuild-test}"
IMAGE="$REGISTRY/$REPOSITORY:$TAG"
MODE="${1:---check}"

case "$MODE" in
  --check)
    echo "=== Docker build configuration ==="
    echo "Image: $IMAGE"
    echo "Platform: linux/amd64"

    test -s docker/Dockerfile
    test -s docker/requirements.lock

    for file in \
      qwen3_serve.py \
      qwen3_nccl_generate.py \
      qwen3_prepare_step.py \
      qwen3_stage_loader.py \
      docker-entrypoint.sh
    do
      test -s "src/gpu/$file"
    done

    if command -v docker >/dev/null 2>&1; then
      docker --version
      if docker info >/dev/null 2>&1; then
        echo "Docker daemon: available"
      else
        echo "Docker daemon: unavailable"
      fi
    else
      echo "Docker CLI: unavailable on this machine"
    fi

    echo "Build files present."
    ;;

  --build)
    ./scripts/preflight-image-build.sh
    docker build \
      --platform linux/amd64 \
      -f docker/Dockerfile \
      -t "$IMAGE" \
      .

    docker run --rm \
      --platform linux/amd64 \
      --entrypoint python \
      "$IMAGE" \
      -c '
import torch
import transformers
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
print("Transformers:", transformers.__version__)
assert torch.__version__ == "2.14.1+cu130"
assert torch.version.cuda == "13.0"
'

    echo "Build and CPU-only package smoke test passed."
    echo "GPU/NCCL tests still required."
    ;;

  --push)
    if [[ "${CONFIRM_ECR_PUSH:-}" != "YES" ]]; then
      echo "Set CONFIRM_ECR_PUSH=YES to publish."
      exit 1
    fi

    docker image inspect "$IMAGE" >/dev/null

    aws ecr get-login-password \
      --region "$REGION" |
      docker login \
        --username AWS \
        --password-stdin "$REGISTRY"

    docker push "$IMAGE"

    aws ecr describe-images \
      --region "$REGION" \
      --repository-name "$REPOSITORY" \
      --image-ids "imageTag=$TAG" \
      --query 'imageDetails[0].{
        digest:imageDigest,
        size:imageSizeInBytes,
        pushed:imagePushedAt
      }' \
      --output json

    echo "Image published. Kubernetes remains unchanged."
    ;;

  *)
    echo "Usage: $0 [--check|--build|--push]"
    exit 2
    ;;
esac
