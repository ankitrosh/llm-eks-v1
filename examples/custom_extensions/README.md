# Workflow 3: Full Customization

This workflow demonstrates how developers can provide
extensions without editing the framework source.

The examples reuse the V3.4 extension contracts and
the validated default Qwen3-14B reference metadata.

All commands below are run from the repository root
with the Python package installed.

## 1. Scheduling policy and Kubernetes resources

The example configuration is:

    examples/custom_extensions/extensions.json

It loads the external policy module:

    examples.v3_4_1.custom_scheduler

The module registers `user-capacity-policy` through
`register_extensions(registry)`.

The policy selects:
- max_num_seqs = 2
- max_num_batched_tokens = 384
- scheduling_policy = fcfs

It also applies validated CPU and memory overrides
while retaining the existing one-GPU-per-worker topology.

Render without modifying the default manifests:

```bash
python3 scripts/v3_4_1_render_extensions.py \
  examples/custom_extensions/extensions.json \
  --output-dir test-results/custom-extension-render
```

These are offline manifest changes. They do not
demonstrate a new native scheduling algorithm.

## 2. External model inspection

Provide a Python module containing:

    register_inspectors(registry)

The module registers an `InspectorSpec` with
an architecture name and an inspection callable.

Example:

```bash
python3 scripts/v3_4_generate_model_artifacts.py \
  --config examples/custom_extensions/demo_model_config.json \
  --model-id User/DemoModel \
  --revision demo-only \
  --inspector-module examples.v3_4_1.custom_inspector \
  --output-dir test-results/custom-inspector
```

The example returns a synthetic three-layer layout.

It is intentionally marked `config_inferred_unverified`.
It cannot be used as proof of a real checkpoint's
weights or deployment compatibility.

Production model inspectors must validate the actual
architecture, checkpoint layout, and capabilities.

## 3. Custom inference backend

See:

    examples/v3_4_1/custom_backend.py

The module uses `PluginSpec(kind="backend", ...)`
to register an implementation.

`ExternalBackendAdapter` requires these methods:

    validate(context)
    prepare(execution_plan)
    start()
    generate(request)
    cancel(request_id)
    health()
    stop()

Run the local HTTP integration example:

```bash
python3 scripts/v3_4_1_test_custom_backend.py
```

This example is a CPU-only mock implementation.
It is not a real LLM inference engine.

The minimal HTTP adapter handles non-streaming
completions and rejects unsupported streaming.

## 4. Native vLLM scheduler replacement

The real runtime plugin lives at:

    examples/v3_4_1/runtime_plugin/src/user_vllm_ext/scheduler.py

The validated Kubernetes demonstration is at:

    deploy/v3_4_1/scheduler-mount-test/

It mounts the user-owned module using a ConfigMap,
sets PYTHONPATH, and selects the custom scheduler
through `--scheduler-cls`.

The custom class was instantiated during a
Qwen3-14B two-A10G GPU test in V3.4.

This test proves class replacement and successful
inference, not a scheduling-performance improvement.

Important: the custom scheduler subclasses vLLM
internal classes and is version-sensitive. The
tested configuration disables asynchronous scheduling.

These manifests should not be applied without
reviewing the target runtime and environment.

## Architecture

The framework combines:

- Strategy: interchangeable selected behavior
- Registry/factory: explicit registration, capability
  checking, and construction
- Adapter: framework-facing contracts around
  external backend implementations

Unsupported or incompatible extensions are rejected.

## Current limits

Native vLLM KV-cache implementation replacement is
not integrated.

Research-runtime attention customization does not
establish support for arbitrary native vLLM attention
kernel replacement.

The default Kubernetes generator remains tied to
the verified Qwen3-14B reference deployment.

GPU execution, model compatibility, and operational
readiness must be validated for each new environment.

See docs/V3_4_FREEZE_2026-11-13.md for the
validated scope and known limitations.
