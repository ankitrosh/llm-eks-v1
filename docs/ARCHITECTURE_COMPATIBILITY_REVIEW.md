# Design decisions and limits

The project kept the same core objective from V1 through V3.5: distribute supported model inference across GPUs and make the surrounding runtime and Kubernetes setup easier to customize.

The main change was which component owns inference execution.

## Current capabilities

| Area | Current status |
|---|---|
| GPU inference | Qwen3-14B BF16 with vLLM across two A10Gs has been validated |
| CPU coordination and recovery | Present since V1, adapted to vLLM in V3.3 |
| Strategy, Registry/Factory, Adapter | Implemented for compatible extensions and external backends |
| Model changes | Compatible non-quantized BF16 Qwen3 checkpoints can be planned and rendered offline |
| Custom backend | Supported extension contracts, but no automatic Kubernetes deployment |
| Batching and KV cache | vLLM owns these in the current reference |
| Automatic recovery | Recovery logic exists, but is disabled by default |
| Deployment | CLI generates artifacts and diagnostics, without applying them or allocating GPUs |

## Why vLLM was adopted

The custom runtime made the performance and correctness challenges of distributed inference visible.

As batching and KV-cache management grew more complex, vLLM became a better serving engine for the reference path.

The framework continues to manage the surrounding planning, customization, deployment preparation, coordination, and lifecycle responsibilities.

This was a deliberate division of responsibilities rather than removal of the original design.

## Extension architecture

Three patterns organize supported customization:

- **Strategy:** Select implementation behavior.
- **Registry/Factory:** Register, validate, and construct implementations.
- **Adapter:** Provide a common interface to external components.

The different customization interfaces are not one universal plugin contract.

For example, a CLI backend can generate a plan and manifests, while a runtime backend needs generation, health, and lifecycle functionality.

Supporting one contract does not automatically satisfy the other.

## Current limitations

The following are not established as general capabilities:

- Arbitrary model architecture support
- Automatic custom backend deployment
- Native vLLM attention or KV-cache replacement through the research interfaces
- Production-grade security and automatic recovery
- Automatic multi-cloud deployment
- Guaranteed GPU memory compatibility from offline planning

A passing plan or test does not automatically prove successful live inference.

See [Architecture](ARCHITECTURE.md), [CLI guide](CLI_GUIDE.md), and [V3.4 validation](V3_4_FREEZE_2026-11-13.md).
