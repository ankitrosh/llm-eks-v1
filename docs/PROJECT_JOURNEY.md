# Project journey

## Why I started

I started LLM-EKS to explore whether larger language models could run across more accessible GPUs instead of relying on a single high-memory accelerator.

Quantization was one option, but pipeline parallelism offered another: place different transformer layers on different GPUs and connect them through distributed communication.

The goal grew from a working inference experiment into a reusable Kubernetes-oriented framework.

## V1: Make distributed inference work

I split Qwen3-14B across two NVIDIA A10G GPUs, with 20 transformer layers on each.

The custom PyTorch implementation used NCCL to move activations between the workers. I added Docker, persistent serving, and AWS EKS deployment.

**A CPU coordinator and a CPU recovery controller were already part of V1**, including health checks and recovery of an unhealthy pipeline.

V1 established that the approach worked, not that it was faster than every single-GPU option.

## V2: Optimize and understand the trade-offs

I experimented with GPTQ INT8 and Marlin to run larger checkpoints, including Qwen3-32B, across the same GPU class.

I also investigated SDPA attention and causal masking.

Prefix KV caching was especially useful. Repeated prompts could reuse previously computed keys and values, reducing work during prefill.

I checked that cached and uncached requests generated identical token IDs in the tested cases.

## V3 to V3.2: Build reusable interfaces

I moved model loading, stage construction, transport, scheduling, and optimization choices toward configurable interfaces.

Concurrent inference exposed how difficult sequence ordering, per-request KV state, and batched correctness can become.

The custom runtime produced useful results, but some sequential-versus-concurrent numerical checks were still unresolved.

## V3.3: Adopt vLLM for serving

I integrated vLLM rather than continuing to rebuild every part of the inference engine.

vLLM now handles core model execution, batching, and KV-cache management in the reference setup.

The existing CPU coordinator and recovery concepts were adapted to the new GPU worker deployment.

This allowed the project to focus on deployment, validation, extensibility, and lifecycle behavior.

## V3.4 and V3.5: Make it usable and extensible

I completed the Strategy, Registry/Factory, and Adapter architecture for supported customizations.

This includes custom model inspection, backend integration, scheduling policies, and Kubernetes resource settings.

A custom vLLM scheduler class was instantiated during real GPU inference.

V3.5 added a CLI, examples, release packaging, documentation, and safer default deployment behavior.

## Results

| Experiment | Recorded result | Context |
|---|---|---|
| V1 Qwen3-14B BF16 | About 4.48 s median E2E | Custom Docker run, 128 input and 64 output tokens |
| V2 repeated-prefix cache | TTFT 949.3 ms cold, 52.8 ms warm | Single cold/warm comparison, 860 input tokens |
| V2 Qwen3-32B GPTQ INT8 | About 2.93 output tokens/s | Different model and quantization, feasibility result |
| V3 custom concurrent pipeline | 19.63 aggregate output tokens/s | Four simultaneous requests, custom eager-attention engine |
| V3.3 vLLM concurrent serving | About 55.19 aggregate output tokens/s | Earlier benchmark record for four concurrent requests; reproducibility check pending |

The numbers come from different workloads and runtimes, so they should not be interpreted as a controlled speedup comparison.

The V3.3 figure was recorded during an earlier benchmark session, but its original output log has not yet been verified in the current repository.

See [benchmark instructions](BENCHMARK.md) for reproducing the test.

## What comes next

Next I want to explore **Mixture-of-Experts inference** and improve **observability**, particularly latency, GPU usage, scheduling behavior, recovery, and infrastructure cost.

Both directions build on what the project has already taught me about distributed execution and operating inference services.

For more details, see:

- [V2 milestones](V2_MILESTONE_REPORT.md)
- [V3 freeze](V3_FREEZE_2026-11-07.md)
- [V3.4 validation](V3_4_FREEZE_2026-11-13.md)
