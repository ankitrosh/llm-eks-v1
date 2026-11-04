#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-2
ACCOUNT=657173565477
REPO=qwen3-14b-pipeline
TAG=v3-plan-r2

REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

echo "========== ECR LOGIN =========="

aws ecr get-login-password --region "$REGION" |
  docker login \
    --username AWS \
    --password-stdin "$REGISTRY"

echo "========== BUILD V3 IMAGE =========="

docker build \
  --progress=plain \
  -f docker/Dockerfile.v3-plan \
  -t "$IMAGE" \
  .

echo "========== VERIFY FRAMEWORK IMPORTS =========="

docker run --rm \
  --entrypoint python \
  "$IMAGE" \
  -c '
import qwen3_serve
import qwen3_stage_loader
import prefix_cache
import prefix_kv
import gptqmodel_marlin_kernels
from llm_runtime.v3_qwen3_adapter import Qwen3Adapter
from llm_runtime.v3_stage_bridge import load_qwen3_stage_v3
from llm_runtime.v3_plan_loader import load_runtime_plan

print("PASS: V3 runtime and Marlin imports")
'

echo "========== VERIFY COMPILED PLAN =========="

docker run --rm \
  --entrypoint python \
  "$IMAGE" \
  -c '
import json
from pathlib import Path
from llm_runtime.v3_plan_loader import digest

path = Path("/app/v3/plans/qwen3-v3-runtime.json")
plan = json.loads(path.read_text())

expected = digest({
    k: v for k, v in plan.items()
    if k != "plan_sha256"
})

assert plan["plan_sha256"] == expected
assert plan["pipeline"]["num_ranks"] == 2
assert plan["pipeline"]["split"] == 20
assert plan["runtime_environment"]["LLM_STAGE_IMPL"] == "v3_adapter"
assert plan["runtime_environment"]["LLM_GPTQ_BACKEND"] == "marlin"

print("PASS: V3 execution plan integrity")
'

echo "========== PUSH TO ECR =========="

docker push "$IMAGE"

echo "========== IMMUTABLE IMAGE DIGEST =========="

DIGEST=$(aws ecr describe-images \
  --region "$REGION" \
  --repository-name "$REPO" \
  --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].imageDigest' \
  --output text)

test "$DIGEST" != "None"

echo "PASS: V3 image published"
echo "IMAGE=${REGISTRY}/${REPO}@${DIGEST}"
