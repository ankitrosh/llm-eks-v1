#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="test-results/image-provenance"
mkdir -p "$OUT"

kubectl exec -i -n llm qwen3-0 -c runtime -- \
  bash -s <<'REMOTE' > "$OUT/runtime-environment.txt"
set -e

echo "=== OS ==="
cat /etc/os-release

echo
echo "=== Python and PyTorch ==="
python - <<'PY'
import sys
import torch
import transformers
import fastapi
import uvicorn

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch location:", torch.__file__)
print("CUDA:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
print("Transformers:", transformers.__version__)
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
PY

echo
echo "=== CUDA installation ==="
ls -ld /usr/local/cuda* 2>/dev/null || true
command -v nvcc || true
command -v nvidia-smi || true

echo
echo "=== CUDA/NCCL libraries ==="
ldconfig -p 2>/dev/null \
  | grep -E 'libnccl|libcudart|libcuda.so' \
  | head -30 || true

echo
echo "=== Python package locations ==="
python -m site

echo
echo "=== GPU entrypoint ==="
ls -l /app/docker-entrypoint.sh

REMOTE

echo "Saved: $OUT/runtime-environment.txt"
