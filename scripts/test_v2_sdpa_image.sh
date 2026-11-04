#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
ACCOUNT=657173565477
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$REGISTRY"

echo "========== BUILD CPU PROBE IMAGE =========="

docker build \
  --progress=plain \
  -f docker/Dockerfile.v2-sdpa-probe \
  -t qwen3-sdpa-probe:cpu \
  .

echo "========== RUN ATTENTION CORRECTNESS =========="

docker run --rm \
  --entrypoint python \
  qwen3-sdpa-probe:cpu \
  /app/tests/test_v2_sdpa_lower_right.py

echo "PASS: SDPA lower-right correctness test"
