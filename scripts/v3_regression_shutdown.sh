#!/usr/bin/env bash
set -uo pipefail

REGION=us-east-2
CLUSTER=llm-pipeline-v1
NS=llm

OLD_IMAGE="657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:92c72a89d24febfbc6d31f36bb895d621298ca80b991596bd541b52c7b110856"

echo "========== STOP WORKLOADS =========="

kubectl -n "$NS" scale \
  deployment/qwen3-coordinator \
  deployment/qwen3-recovery \
  deployment/qwen3-gpu-observer \
  deployment/qwen3-maintenance-supervisor \
  --replicas=0

kubectl -n "$NS" scale sts/qwen3 --replicas=0

echo "========== WAIT FOR RANK POD REMOVAL =========="

for i in $(seq 1 60); do
  PODS=$(kubectl -n "$NS" get pods \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    2>/dev/null | grep -E '^qwen3-[01]$' || true)

  if [ -z "$PODS" ]; then
    echo "PASS: Both rank pods removed"
    break
  fi

  sleep 5
done

echo "========== RESTORE V2 CONFIGURATION =========="

kubectl -n "$NS" set image \
  sts/qwen3 runtime="$OLD_IMAGE"

kubectl -n "$NS" set env sts/qwen3 \
  --containers=runtime \
  LLM_STAGE_IMPL- \
  LLM_RUNTIME_PLAN- \
  LLM_PREFIX_CACHE_ENABLED=1 \
  LLM_ATTENTION_BACKEND=sdpa \
  LLM_GPTQ_BACKEND=marlin \
  LLM_SDPA_PREFIX_MASK=explicit

echo "========== SCALE NODEGROUPS TO ZERO =========="

for NG in gpu-a10g cpu-system; do
  MAX=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.maxSize' \
    --output text)

  aws eks update-nodegroup-config \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NG" \
    --region "$REGION" \
    --scaling-config \
      "minSize=0,maxSize=${MAX},desiredSize=0" \
    --query 'update.id' \
    --output text || echo "WARNING: Retry scale-down for $NG"
done

echo "========== VERIFY NODEGROUP SCALE-DOWN =========="

for i in $(seq 1 90); do
  GPU_DESIRED=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name gpu-a10g \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)

  CPU_DESIRED=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name cpu-system \
    --region "$REGION" \
    --query 'nodegroup.scalingConfig.desiredSize' \
    --output text)

  echo "GPU desired=$GPU_DESIRED CPU desired=$CPU_DESIRED"

  if [[ "$GPU_DESIRED" == "0" && "$CPU_DESIRED" == "0" ]]; then
    echo "PASS: Both nodegroups desired capacity is zero"
    break
  fi

  sleep 10
done

echo "========== FINALIZE AWS SHUTDOWN =========="

for i in $(seq 1 12); do
  python3 scripts/finalize_eks_shutdown.py \
    2>&1 | tee /tmp/v3-regression-finalize.log

  if grep -Fq \
    "PASS: All EKS worker instances removed" \
    /tmp/v3-regression-finalize.log; then
    echo "PASS: V3 regression shutdown complete"
    exit 0
  fi

  sleep 15
done

echo "WARNING: Termination not fully confirmed."
echo "Run this shutdown script again until it prints PASS."
exit 1
