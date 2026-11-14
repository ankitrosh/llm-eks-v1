# LLM-EKS

LLM-EKS is an open-source framework for running and customizing distributed LLM inference on Kubernetes. It helps users inspect supported models, prepare execution plans, generate deployment resources, and manage a distributed serving setup.

The working reference uses **Qwen3-14B, vLLM, and two NVIDIA A10G GPUs on AWS EKS**, with pipeline parallelism across the GPUs. It is a validated reference setup, not a production-certified service or a universal model loader.

## Three ways to use it

**1. Use the reference setup.** Keep the validated Qwen3-14B configuration and generate the Kubernetes resources. See [Default workflow](examples/basic/README.md).

**2. Change the model.** For a compatible BF16 Qwen3 checkpoint, inspect its metadata, generate a new plan, and render a separate GPU-worker bundle. This path does not yet create new storage or a complete coordinator and recovery stack. See [Model swap](examples/model_swap/README.md).

**3. Fully customize.** Register a custom model inspector, backend, scheduler policy, or supported resource override through the extension interfaces. Your implementation must meet its own runtime and deployment requirements. See [Custom extensions](examples/custom_extensions/README.md).

## Reference architecture

Two GPU workers run vLLM with PP=2 and TP=1. A CPU gateway checks serving readiness and forwards requests to rank 0. The CPU recovery controller watches worker health and manages recovery state.

Both CPU components existed in the original V1 project and were adapted for vLLM in V3.3.

See the [architecture diagram](docs/ARCHITECTURE.md) and [repository map](docs/REPOSITORY_MAP.md).

## Start with the offline workflow

Requires Python 3.10+.

```bash
git clone https://github.com/ankitrosh/llm-eks-v1.git
cd llm-eks-v1
python3 -m pip install -e '.[config]'
```

To generate and validate the default Kubernetes manifests:

```bash
python3 scripts/v3_3_generate_k8s.py
python3 scripts/v3_5_verify_default.py
```

Expected validation marker:

    V35_DEFAULT_WORKFLOW_PASS

This operation is CPU-only. It does not deploy anything
or allocate GPU resources.

## Public CLI

The installed `llm-eks` command supports five operations:

| Command | Purpose |
|---|---|
| `llm-eks configure` | Local configuration and optional AWS discovery |
| `llm-eks plan` | Backend-specific execution planning |
| `llm-eks render` | Offline artifact generation |
| `llm-eks doctor` | Optional environment and compatibility diagnostics |
| `llm-eks test` | Explicitly selected tests only |

Example reference workflow:

```bash
llm-eks configure --aws
llm-eks plan
llm-eks render
```

Custom backends use the framework extension registry,
with `--allow-custom-code` required to import trusted
external Python modules.

Framework regression tests run in development/CI,
not automatically in end-user commands. The custom
backend sample demonstrates extensibility, not
inference correctness.

See the [CLI Guide](docs/CLI_GUIDE.md) for full
usage, plugin examples and limitations.

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
contain 142 passing CPU regression tests.

GPU tests are separate and require compatible hardware,
model checkpoint files, and distributed infrastructure.

`llm-eks` also provides `configure`, `plan`, `render`, `doctor`, and opt-in `test`. See the [CLI guide](docs/CLI_GUIDE.md).

## Documentation

- [Architecture and three user journeys](docs/ARCHITECTURE.md)
- [Repository map](docs/REPOSITORY_MAP.md)
- [Project journey and results](docs/PROJECT_JOURNEY.md)
- [Design decisions and current limits](docs/ARCHITECTURE_COMPATIBILITY_REVIEW.md)
- [What each test file checks](docs/TEST_GUIDE.md)
- [Benchmark procedure](docs/BENCHMARK.md)

## Limits and next steps

The reference deployment is fixed to a validated two-node Qwen3-14B configuration. Model swaps are restricted to supported Qwen3 layouts.

Custom backends are not automatically deployable. The runtime adapter and CLI planning interfaces are separate. The earlier custom attention and KV-cache implementations are not native vLLM replacements.

Future work will focus on **Mixture-of-Experts (MoE) inference and observability**.

See [V3.4 validation](docs/V3_4_FREEZE_2026-11-13.md) for established capabilities and limits.

## License

This framework is released under the
[Apache License 2.0](LICENSE).

Third-party dependencies, inference engines, models,
and container images retain their own licenses.
See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

## AWS setup and GPU quotas

For instructions on configuring an AWS account, locating
EKS clusters and GPU resources, and requesting additional
EC2 GPU vCPU quota, see:

[AWS Setup and GPU Quota Guide](docs/AWS_SETUP_AND_GPU_QUOTAS.md)

A proposed deployment configuration template is available
at `examples/aws/deployment.example.yaml`.

Read-only AWS account, region and EKS cluster discovery
is available through `scripts/v3_5_aws_configure.py`.
GPU node, quota, storage and deployment diagnostics are
available through the V3.5 CLI and reference utilities.
