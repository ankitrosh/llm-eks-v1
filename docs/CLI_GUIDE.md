# LLM-EKS CLI Guide

The `llm-eks` command provides configuration discovery,
execution planning, deployment artifact generation,
diagnostics, and optional user-selected testing.

The CLI does not deploy resources or automatically
execute inference correctness tests.

## Installation

Requires Python 3.10+.

From the repository checkout:

```bash
python3 -m pip install -e '.[config]'
llm-eks --help
```

The AWS reference workflows currently require the
repository checkout because some utilities and frozen
reference templates are loaded from its directories.

## Commands

| Command | Purpose |
|---|---|
| configure | Create configuration, optionally discover AWS resources |
| plan | Generate a backend-specific execution plan |
| render | Generate JSON artifacts without deployment |
| doctor | Run optional diagnostics |
| test | Execute an explicitly supplied test command |

No command implicitly runs the framework regression suite.

## 1. AWS reference workflow

Discover the AWS identity, cluster, node groups,
regional GPU quota and existing model PVCs:

```bash
llm-eks configure \
  --aws \
  --output .llm-eks/deployment.yaml
```

This uses the current AWS CLI identity. You may supply
--region, --cluster, --profile or --gpu-instance.

Discovery can leave unresolved values when access is
unavailable or more than one choice is possible.

Review the generated configuration before continuing.

For the validated Qwen3-14B reference configuration:

```bash
llm-eks plan \
  --config .llm-eks/deployment.yaml \
  --output test-results/reference-plan.json
```

Render the isolated Kubernetes resource bundle:

```bash
llm-eks render \
  --config .llm-eks/deployment.yaml \
  --output-dir test-results/reference-manifests
```

The reference renderer is deliberately limited to
the validated Qwen3-14B BF16 / vLLM PP=2 TP=1
configuration on two g5.xlarge workers.

It emits 14 isolated resources with zero replicas.
Automatic recovery remains disabled.

It does not create or modify AWS/Kubernetes resources.

Output paths must not already exist; the CLI refuses
to overwrite existing artifacts.

## 2. Optional diagnostics

Inspect the saved reference configuration:

```bash
llm-eks doctor \
  --config .llm-eks/deployment.yaml
```

Add read-only Kubernetes checks:

```bash
llm-eks doctor \
  --config .llm-eks/deployment.yaml \
  --check-k8s
```

The doctor distinguishes:

- PASS: a checked condition was satisfied
- WARN: incomplete, conditional or unverified readiness
- FAIL: a required reference configuration check failed
- UNVERIFIED: a custom provider did not supply evidence

Doctor diagnostics are informative and optional.
The reference renderer still enforces its required
configuration contracts independently.

A clean doctor report does not guarantee successful
GPU inference or arbitrary runtime correctness.

## 3. Custom backend workflow

Custom Python modules can register backend strategies
using the existing framework PluginRegistry and
PluginSpec contracts.

The built-in example configuration is:

    examples/v3_5_cli/config.yaml

Plan using a trusted custom backend:

```bash
llm-eks plan \
  --config examples/v3_5_cli/config.yaml \
  --allow-custom-code \
  --output test-results/custom-plan.json
```

Render using its optional render(plan) capability:

```bash
llm-eks render \
  --config examples/v3_5_cli/config.yaml \
  --allow-custom-code \
  --output-dir test-results/custom-artifacts
```

The CLI selects plugin mode when the configuration
includes a plugins section, or through --mode plugin.

Plugin mode does not enforce Qwen3 model metadata,
vLLM flags, or reference GPU hardware requirements.

Required extension registration and backend selection
still fail closed for incompatible contracts.

A selected custom backend may implement:

    plan() -> dict

and, for artifact rendering:

    render(execution_plan) -> dict[str, dict]

A backend without render() can still support planning.
Not every custom backend or returned artifact is
necessarily a deployable Kubernetes implementation.

The sample backend demonstrates the plugin interface;
it does not implement LLM inference.

Importing external Python modules executes trusted code.
Only use --allow-custom-code with reviewed code.

## 4. Custom diagnostics

```bash
llm-eks doctor \
  --config examples/v3_5_cli/config.yaml \
  --allow-custom-code
```

A custom backend may optionally provide diagnose().

Without provider-specific diagnostics, the CLI reports
runtime correctness and deployment readiness as
UNVERIFIED, not incorrectly as PASS or FAIL.

## 5. Opt-in testing

The CLI never automatically executes user tests.

Without a supplied command:

```bash
llm-eks test
```

No tests run.

To execute a selected test command:

```bash
llm-eks test -- \
  python3 -m unittest discover \
  -s my_tests -p 'test_*.py'
```

The CLI runs only the command explicitly supplied
after -- and propagates its exit status.

Users are responsible for selecting meaningful
tests for their implementation.

Passing a test does not prove untested runtime behavior.

## Framework tests versus user tests

The repository's V3.3, V3.4 and V3.5 regression
suites validate framework implementation behavior.

They run during framework development and CI.

They do not run during configure, plan or render,
and are not universal requirements for user plugins.

Reference-specific tests remain valuable for
maintaining the reference implementation, but
custom backends are not required to pass reference
Qwen3-only or vLLM-only test assumptions.

## Safety and current limitations

- Configuration discovery does not provision infrastructure.
- Rendering does not apply Kubernetes manifests.
- No CLI deploy command is provided in V3.5.
- External plugin code must be explicitly trusted.
- Custom runtime correctness is not inferred automatically.
- Reference rendering requires supported model metadata
  and compatible infrastructure configuration.
- Saved quota values do not establish unused EC2 capacity.
- Bound PVCs do not establish checkpoint contents.
- CPU and GPU node readiness may remain unverified.
- A separately packaged wheel does not include the
  repository's reference scripts and deployment templates.

Keep generated configurations under .llm-eks/ and
local rendered artifacts under test-results/ so they
remain excluded from Git by the repository ignore rules.

For AWS setup and GPU quota instructions, see:

[AWS Setup and GPU Quotas](AWS_SETUP_AND_GPU_QUOTAS.md)
