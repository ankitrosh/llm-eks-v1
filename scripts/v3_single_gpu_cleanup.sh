#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

REGION=us-east-2
CLUSTER=llm-pipeline-v1

echo "========== DELETE TEST JOB =========="

kubectl -n llm delete job v3-gpt2-single-gpu \
  --ignore-not-found \
  --wait=true \
  --timeout=180s || true

echo "========== REQUEST NODEGROUP SHUTDOWN =========="

for NG in gpu-a10g cpu-system; do
  for attempt in $(seq 1 60); do
    STATUS=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$NG" \
      --region "$REGION" \
      --query 'nodegroup.status' \
      --output text) || continue

    DESIRED=$(aws eks describe-nodegroup \
      --cluster-name "$CLUSTER" \
      --nodegroup-name "$NG" \
      --region "$REGION" \
      --query 'nodegroup.scalingConfig.desiredSize' \
      --output text) || continue

    if [ "$DESIRED" = "0" ]; then
      break
    fi

    if [ "$STATUS" = "ACTIVE" ]; then
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
          "minSize=0,maxSize=${MAX},desiredSize=0"

      break
    fi

    sleep 10
  done
done

echo "========== FINAL TERMINATION VERIFICATION =========="

for attempt in $(seq 1 40); do
  OUTPUT=$(python3 scripts/finalize_eks_shutdown.py 2>&1)
  printf '%s\n' "$OUTPUT"

  if printf '%s\n' "$OUTPUT" |
    grep -Fq "PASS: All EKS worker instances removed"; then
    exit 0
  fi

  sleep 15
done

echo "WARNING: Worker termination not confirmed."
echo "Rerun this script until the final PASS appears."
exit 1
