#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

echo "========== DELETE DISTRIBUTED JOB =========="

kubectl -n llm delete job v3-gpt2-dist \
  --ignore-not-found \
  --wait=true \
  --timeout=180s || true

echo "========== DELETE RENDEZVOUS SERVICE =========="

kubectl -n llm delete service v3-gpt2-dist-headless \
  --ignore-not-found || true

echo "========== SHUT DOWN EKS WORKERS =========="

bash scripts/v3_single_gpu_cleanup.sh
