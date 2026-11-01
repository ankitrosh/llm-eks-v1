#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

echo "=== DOCKER BUILD PREFLIGHT ==="

echo
echo "1. Docker environment"
docker version --format 'Client: {{.Client.Version}}'
docker info --format 'Server architecture: {{.Architecture}}'
docker info --format 'Docker root: {{.DockerRootDir}}'

ROOT=$(docker info --format '{{.DockerRootDir}}')

echo
echo "2. Available storage"
df -h "$ROOT" .
docker system df

echo
echo "3. Locked dependencies"

python3 - <<'PY'
from pathlib import Path

lock = Path("docker/requirements.lock")
lines = [
    line.strip()
    for line in lock.read_text().splitlines()
    if line.strip()
]

assert lines, "Dependency lock is empty"

assert not any(
    line.lower().startswith("torch==")
    for line in lines
), "PyTorch must be installed from the CUDA 13.0 index"

unportable = [
    line for line in lines
    if line.startswith(("-e ", "file:", "--"))
    or " @ file://" in line
]

assert not unportable, (
    f"Nonportable package references: {unportable}"
)

print("Pinned packages:", len(lines))

print("\nCUDA-related dependencies:")
for line in lines:
    if line.lower().startswith((
        "nvidia-", "triton", "torchvision", "torchaudio"
    )):
        print(" ", line)

print("\nPASS: Dependency file is structurally valid")
PY

echo
echo "4. Source completeness"

for file in \
  qwen3_serve.py \
  qwen3_nccl_generate.py \
  qwen3_prepare_step.py \
  qwen3_stage_loader.py \
  docker-entrypoint.sh
do
  test -s "src/gpu/$file"
  echo "PASS: $file"
done

echo
echo "5. Free space threshold"

AVAILABLE_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
AVAILABLE_GIB=$((AVAILABLE_KB / 1024 / 1024))

echo "Available Docker filesystem space: ${AVAILABLE_GIB} GiB"

if (( AVAILABLE_GIB < 60 )); then
  echo
  echo "WARNING: Less than 60 GiB available."
  echo "Recommend building on an EC2 or CodeBuild environment"
  echo "with more storage rather than risking CloudShell exhaustion."
  exit 1
fi

echo
echo "PASS: Storage preflight completed"
