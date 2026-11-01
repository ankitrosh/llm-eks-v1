FROM ubuntu:24.04

# Triton requires a C compiler during GPU kernel initialization
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc CXX=/usr/bin/g++


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/llm-venv \
    PATH=/opt/llm-venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python 3.12 is Ubuntu 24.04's default Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        ca-certificates \
        libgomp1 \
        libnuma1 \
        libibverbs1 \
        libstdc++6 \
        bash && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/llm-venv

# Install the exact CUDA-enabled PyTorch release.
RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.14.1+cu130'

# Install the remaining versions captured from the live image.
COPY docker/requirements.lock /tmp/requirements.lock

RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r /tmp/requirements.lock && \
    python -m pip check

# Validate versions at image build time.
RUN python - <<'PY'
import torch
import transformers
import fastapi
import uvicorn

assert torch.__version__ == "2.14.1+cu130"
assert torch.version.cuda == "13.0"
assert transformers.__version__ == "4.56.2"
assert fastapi.__version__ == "0.162.0"
assert uvicorn.__version__ == "0.54.0"

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("NCCL:", torch.cuda.nccl.version())
print("Python environment verified")
PY

WORKDIR /app

COPY src/gpu/ /app/

RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
