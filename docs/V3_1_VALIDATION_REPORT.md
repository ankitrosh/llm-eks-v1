# V3.1 — Configurable Distributed Inference Framework

Validation date: 2026-11-05 UTC
Branch: v3-framework
Status: VALIDATED — Qwen3 two-rank reference implementation

## Architecture implemented

- YAML model and pipeline configuration
- Checkpoint metadata discovery and validation
- Manual and equal-layer partition planning
- Per-layer attention configuration and declarations
- Explicit custom plugin discovery and interface checks
- Model adapter registry and Qwen3 adapter
- Validated JSON execution-plan compiler
- Checkpoint configuration consistency checks
- Adapter-backed serving integration
- V2-compatible NCCL and prefix-cache execution

## Reference configuration

Model: JunHowie/Qwen3-14B-GPTQ-Int8
Revision: 9b876cd2f6f2c870afd18bb5b458fb38ab3e3e00
Partition: 20/20
Hardware: 2 x NVIDIA A10G
Quantization backend: Marlin
Attention backend: PyTorch SDPA
Prefix mask: explicit
Generation: greedy, 24 output tokens

Validated Docker image:
657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:514663c208715815ed3516b2dbce2e94158207ffd7adf11348449d95441fd747

## Validation results

CPU test suite: 62/62 passing.

The legacy and V3 adapter paths produced identical
24-token generated ID sequences.

| Metric (ms unless noted) | Legacy | V3 |
|---|---:|---:|
| End-to-end | 1616.817 | 1588.063 |
| TTFT | 485.018 | 478.541 |
| Decode interval median | 48.571 | 47.788 |
| Stage 0 compute | 768.088 | 755.261 |
| Peak memory (MiB) | 8083.6 | 8083.6 |

## Distributed prefix cache validation

| Metric | Cold | Warm |
|---|---:|---:|
| TTFT (ms) | 485.573 | 57.797 |
| End-to-end (ms) | 1644.504 | 1198.283 |
| Prefill GPU time (ms) | 238.656 | 23.191 |
| Prefix cache hit | false | true |
| Reused tokens | 0 | 283 |

Cold and warm requests produced identical token IDs.

Observed warm-cache TTFT speedup: 8.4x.
Observed prefill GPU speedup: 10.3x.

These are single-run measurements and should not be
interpreted as statistically significant improvements.

## Scope and limitations

Validated:
- Qwen3 GPTQ INT8 reference checkpoint
- Two-rank CUDA/NCCL serving
- V3 adapter delegation to the existing Qwen3 loader
- SDPA and Marlin integration
- Distributed prefix-cache reuse

Not yet validated:
- Arbitrary model architectures
- N-rank execution
- CPU/Gloo distributed model execution
- Production execution of custom attention plugins
- Concurrent multi-request pipeline inference
- Continuous batching
- Horizontal replica autoscaling
- Compute-aware layer partitioning

A checkpoint configuration hash does not verify the
integrity of every weight shard.

## Next milestone: V3.2

Implement a genuine N-rank transport and execution
interface. Use DistilGPT2 with CPU/Gloo to test a
three-stage distributed pipeline before extending
the GPU/NCCL runtime.
