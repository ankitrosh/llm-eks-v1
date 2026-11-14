# Workflow 1: Default Distributed Inference

This workflow uses the repository's validated default
Qwen3-14B configuration.

## Requirements

For offline manifest generation:
- Python 3.10+
- A local clone of this repository
- An editable installation of the Python package

For actual inference:
- A compatible Kubernetes cluster
- Two NVIDIA GPU workers with sufficient memory
- NVIDIA GPU support in Kubernetes
- Model checkpoint files available through the expected PVCs
- Working NCCL networking and distributed vLLM runtime

The validated reference environment used two NVIDIA A10G GPUs
on AWS EKS, one GPU per worker.

## Offline quick start

From the repository root:

```bash
python3 -m pip install -e .
python3 scripts/v3_3_generate_k8s.py
python3 scripts/v3_5_verify_default.py
```

The generator reads the checked-in Qwen3-14B model
metadata and writes deployment manifests under:

    deploy/v3_3/rendered/

It also refreshes the execution plan and scheduling policy
in the checked-in default model profile.

No GPU is allocated and no Kubernetes resources are applied.

The generated deployments have zero replicas.

## What is generated?

- Two vLLM GPU worker deployments
- GPU worker services
- CPU coordinator and recovery deployments
- Supporting services and ConfigMaps
- Namespace-scoped recovery permissions
- An initially paused recovery state

## Important limitations

This offline workflow validates deployment preparation,
not live inference.

Actual execution requires the model files, suitable GPU
capacity, Kubernetes infrastructure, and a deliberate
deployment/startup procedure.

Do not scale the workers or apply the manifests to a
production cluster without reviewing the configuration.

The default generator currently targets the fixed
Qwen3-14B, two-node reference profile.

It is not a generic arbitrary-model deployment CLI.
