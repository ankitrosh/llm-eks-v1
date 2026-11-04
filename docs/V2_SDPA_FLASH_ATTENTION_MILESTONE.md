# V2 SDPA / FlashAttention Optimization

Date: 2026-11-03
Status: CPU tests, CUDA microbenchmarks and initial E2E comparison passed

## Environment

- Model: Qwen3-14B GPTQ INT8
- Backend: Marlin
- GPUs: 2 x NVIDIA A10G, 24 GiB each
- Pipeline split: 20/20 transformer layers
- PyTorch: 2.14.1+cu130
- Transformers: 4.56.2
- Prefix caching enabled
- NCCL for stage communication

## Key Discovery

PyTorch SDPA already invokes a FlashAttention kernel for
normal prefill, without requiring standalone flash-attn.

For partial-prefix prefill:
- Explicit 4D causal masks select efficient attention.
- causal_lower_right() selects FlashAttention in tested cases.

Both masks provide the required lower-right causal alignment.

## CPU Correctness

Four comparisons passed:
- 8 cached, 2 new
- 64 cached, 8 new
- 256 cached, 16 new
- 512 cached, 32 new

Maximum reported difference: 0.0000000

## CUDA Attention Microbenchmarks

GPU: NVIDIA A10G
Measurements: per attention operation, in milliseconds

Cached / New | Explicit | Lower-right
64 / 8       | 0.1858   | 0.2277
512 / 32     | 0.1994   | 0.2307
1792 / 256   | 0.5029   | 0.3613

The lower-right path was faster only for the largest
tested cached-prefix workload.

All GPU correctness comparisons passed.

Largest measured maximum absolute difference: 0.003906

## Full Distributed Inference Comparison

Model: Qwen3-14B GPTQ INT8 with Marlin
Prompt: 1181 tokens
Reused prefix: 696 tokens
New suffix: 485 tokens
Generation: 24 tokens

Metric             | Explicit | Lower-right
Rank-0 prefill GPU | 184.331  | 172.833 ms
TTFT               | 436.898  | 360.613 ms
E2E                | 1537.792 | 1458.702 ms

Speedups:
- Rank-0 prefill GPU: 1.067x
- TTFT: 1.212x
- E2E: 1.054x

Generated token IDs matched exactly.

Cold seed prefill:
- Explicit: 417.39 ms
- Lower-right: 417.07 ms

These results are single-run observations, not
statistically reliable benchmark medians.

## Implementation

LLM_ATTENTION_BACKEND=sdpa

Experimental setting:
LLM_SDPA_PREFIX_MASK=lower_right

Existing default:
LLM_SDPA_PREFIX_MASK=explicit

Modified:
- src/gpu/qwen3_stage_loader.py

Profiler:
- scripts/profile_v2_sdpa.py

E2E benchmark:
- scripts/benchmark_v2_sdpa_e2e.py
- scripts/run_v2_sdpa_comparison.sh

Results:
- benchmarks/qwen3_sdpa_mask_gpu_profile_20261103.log
- benchmarks/sdpa_e2e_explicit.json
- benchmarks/sdpa_e2e_lower_right.json

Experimental image digest:
sha256:7a18a04e571a6d74027c145023299df91ae856207838f141e9f15d0b1c8f6e09

## Lessons

- Do not assume fused attention is always faster.
- Mask structure changes CUDA kernel selection.
- Small cached suffixes favored efficient attention.
- Larger cached suffixes favored lower-right FlashAttention.
- Microbenchmark speedups do not directly predict E2E speedups.
- OnDelete StatefulSets require explicit pod readiness checks.
- Shutdown verification must inspect ASG instances, not
  just successful EKS nodegroup updates.

## Decision

Keep explicit masking as the default.

Retain lower-right SDPA masking as an opt-in experimental path.
Do not add standalone flash-attn yet.

Future work:
1. Repeated-run benchmarks for more reliable latency distributions.
2. Optional automatic mask strategy selection.
3. Model/backend compatibility validation.
4. Wider model-family support.
5. Longer-running memory and correctness tests.
