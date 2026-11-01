#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?RANK must be 0 or 1}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"

MASTER_PORT="${MASTER_PORT:-29618}"
MODEL_DIR="${MODEL_DIR:-/models/qwen3-14b}"
MODE="${MODE:-benchmark}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

DIST_ARGS=(
  --nnodes=2
  --nproc-per-node=1
  --node-rank="$RANK"
  --master-addr="$MASTER_ADDR"
  --master-port="$MASTER_PORT"
)

if [[ "$MODE" == "benchmark" ]]; then
  exec torchrun "${DIST_ARGS[@]}" \
    /app/qwen3_nccl_generate.py \
    --rank "$RANK" \
    --model-dir "$MODEL_DIR" \
    --split 20 \
    --prompt-len "${PROMPT_LEN:-128}" \
    --output-len "${OUTPUT_LEN:-64}" \
    --warmups "${WARMUPS:-2}" \
    --runs "${RUNS:-10}" \
    --tag "${TAG:-docker_v1}"

elif [[ "$MODE" == "serve" ]]; then
  exec torchrun "${DIST_ARGS[@]}" \
    /app/qwen3_serve.py \
    --rank "$RANK" \
    --model-dir "$MODEL_DIR" \
    --split "${SPLIT:-20}" \
    --host "${SERVE_HOST:-0.0.0.0}" \
    --port "${SERVE_PORT:-8000}" \
    --queue-capacity "${QUEUE_CAPACITY:-8}" \
    --max-prompt-tokens "${MAX_PROMPT_TOKENS:-2048}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --request-timeout "${REQUEST_TIMEOUT_SECONDS:-180}" \
    --heartbeat-seconds "${HEARTBEAT_SECONDS:-5}" \
    --collective-timeout-seconds "${NCCL_TIMEOUT_SECONDS:-60}" \
    --warmup-output-tokens "${WARMUP_OUTPUT_TOKENS:-16}"

else
  echo "Unknown MODE: $MODE" >&2
  exit 1
fi
