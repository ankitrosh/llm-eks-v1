# Distributed LLM Inference Framework

An extensible framework for distributed LLM inference,
with Kubernetes deployment tooling, model-aware planning,
runtime customization, and recovery coordination.

The framework supports explicit model inspection, validated
execution plans, interchangeable runtime components, and
user-defined extensions.

The current reference implementation uses vLLM with
pipeline parallelism across two GPU workers.

## Project status

**V3.5 — Public release preparation**

Validated reference configuration:

- Model: Qwen3-14B
- Hardware: two NVIDIA A10G GPUs
- Platform: AWS EKS
- Backend: vLLM
- Pipeline parallelism: 2
- Tensor parallelism: 1
- Communication: NCCL
- Precision: BF16

The reference configuration has successfully performed
real distributed GPU inference.

Qwen3-8B has additionally been validated for offline
checkpoint inspection, distributed planning, and isolated
Kubernetes manifest generation.

The project is currently a research/framework release,
not a production-certified serving platform.

## Key capabilities

### Model-aware execution planning

- Architecture-specific checkpoint inspection
- Verified checkpoint-index layout metadata
- Runtime and model capability validation
- Two-node distributed vLLM execution planning
- Explicit rejection of incompatible configurations

### Extensible inference architecture

- Strategy-based behavior selection
- Plugin registry and factory construction
- Backend adapter contracts
- External model inspector registration
- User-defined scheduling policies
- CPU/memory/GPU resource validation

### Distributed Kubernetes serving

- Two-worker pipeline-parallel deployment
- GPU worker and CPU coordinator components
- Gateway readiness and health checks
- Recovery controller with durable state
- Deployment manifests generated offline
- Safe default: zero deployment replicas

### Runtime customization

- External execution backend adapters
- Custom model inspection
- Scheduling parameter customization
- Kubernetes resource overrides
- Native vLLM scheduler class replacement

Native vLLM scheduler replacement was validated with
real GPU inference using the pinned reference runtime.

## Quick start

Requires Python 3.10+.

Clone the repository and install the framework:

```bash
git clone https://github.com/ankitrosh/llm-eks-v1.git
cd llm-eks-v1
git checkout v3.5-public-release

python3 -m pip install -e .
```

Until the release branch is merged into the default
branch, check out `v3.5-public-release` before installation.

To generate and validate the default Kubernetes manifests:

```bash
python3 scripts/v3_3_generate_k8s.py
python3 scripts/v3_5_verify_default.py
```

Expected validation marker:

    V35_DEFAULT_WORKFLOW_PASS

This operation is CPU-only. It does not deploy anything
or allocate GPU resources.

## Three usage workflows

### 1. Default distributed inference

Use the validated Qwen3-14B reference configuration
to generate deployment manifests.

Guide: [Default workflow](examples/basic/README.md)

### 2. Replace the model configuration

Inspect compatible Qwen3 checkpoint metadata, validate
distributed execution parameters, and render an isolated
GPU-worker Kubernetes bundle.

Qwen3-8B has been tested through this offline process.

Guide: [Model replacement](examples/model_swap/README.md)

### 3. Customize framework behavior

Provide custom Python implementations for supported
extension contracts.

Examples cover:

- Scheduling policies and resource configuration
- External architecture inspection
- Custom inference backend integration
- Native vLLM scheduler replacement

Guide: [Full customization](examples/custom_extensions/README.md)

## Architecture

The framework uses three complementary patterns:

**Strategy:** Select interchangeable implementation behavior.

**Registry / Factory:** Explicitly register, validate,
and construct compatible extensions.

**Adapter:** Present consistent framework-facing
interfaces around externally supplied backends.

See the V3.4 freeze report for validated capabilities:

[Technical validation and limitations](docs/V3_4_FREEZE_2026-11-13.md)

## Repository organization

| Directory | Purpose |
|---|---|
| `src/llm_runtime/` | Framework runtime and extension contracts |
| `scripts/` | Supported utilities and validation tools |
| `examples/` | User workflows and extension examples |
| `deploy/` | Kubernetes manifests and templates |
| `infrastructure/` | Infrastructure profiles and historical configurations |
| `tests/` | Regression and contract tests |
| `docs/` | Architecture, validation, and operational notes |
| `research/archive/` | Historical experiments and development backups |

## Run CPU regression tests

From the repository root:

```bash
python3 -m unittest discover \
  -s tests -p 'test_v3_3_*.py'

python3 -m unittest discover \
  -s tests -p 'test_v3_4*.py'

python3 -m unittest discover \
  -s tests -p 'test_v3_5*.py'
```

The V3.3, V3.4, and V3.5 regression groups currently
contain 75 passing CPU tests.

GPU tests are separate and require compatible hardware,
model checkpoint files, and distributed infrastructure.

## Deployment safety

Generated Kubernetes manifests are not automatically
applied to a cluster.

Reference worker and coordinator deployments start
with zero replicas.

Before launching GPU inference, provision and verify
the necessary cluster, networking, model storage,
GPU scheduling, and security requirements.

Deployment resources and storage may incur cloud costs.

Do not assume that offline model planning guarantees
GPU memory compatibility or successful live inference.

## Current limitations

- Arbitrary model architectures are not automatically supported.
- The public model-swap deployment renderer currently
  targets compatible non-quantized BF16 Qwen3 models.
- Alternative-model bundles do not yet include a dedicated
  coordinator/recovery deployment or storage provisioning.
- Native vLLM KV-cache replacement is not integrated.
- Research-runtime attention customization is not equivalent
  to native vLLM attention-kernel replacement.
- Custom vLLM scheduler classes use version-sensitive
  upstream internals.
- Production security, durability, and sustained-load
  readiness have not been certified.

## Development history

The project evolved through custom multi-GPU execution,
distributed communication, scheduling and caching
research, and later integration with vLLM.

Historical development files remain available under
`research/archive/` and through Git history.

## License

A license has not yet been selected for the public release.
Until a license is added, no open-source license permissions
are granted by this repository.

## AWS setup and GPU quotas

For instructions on configuring an AWS account, locating
EKS clusters and GPU resources, and requesting additional
EC2 GPU vCPU quota, see:

[AWS Setup and GPU Quota Guide](docs/AWS_SETUP_AND_GPU_QUOTAS.md)

A proposed deployment configuration template is available
at `examples/aws/deployment.example.yaml`.

Read-only AWS account, region and EKS cluster discovery
is available through `scripts/v3_5_aws_configure.py`.
GPU node, quota, storage and deployment validation remain
planned.
