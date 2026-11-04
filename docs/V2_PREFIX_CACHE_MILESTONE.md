# V2 Prefix KV Cache — Implementation and Results

Date: 2026-11-03
Status: Distributed correctness smoke tests passed
Infrastructure: AWS EKS, us-east-2

## 1. Objective

Reduce repeated prompt-prefill computation in the custom
two-stage pipeline-parallel LLM inference framework.

Maintain correctness, preserve the Marlin backend, and avoid
introducing vLLM, Ray Serve, or a new scheduling architecture.

## 2. Tested Environment

Model:
- JunHowie/Qwen3-14B-GPTQ-Int8
- GPTQ INT8 weights, BF16 runtime activations
- 40 transformer layers, 20 layers per stage

Hardware:
- 2 x NVIDIA A10G, each with 24 GiB VRAM
- AWS g5.xlarge EC2 workers
- EKS managed nodegroup: gpu-a10g
- Separate t3.large CPU system node

Runtime:
- PyTorch: 2.14.1+cu130
- Transformers: 4.56.2
- NCCL communication
- Marlin GPTQ execution backend
- FastAPI persistent serving
- One active request at a time

Prefix-cache image:
657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:92c72a89d24febfbc6d31f36bb895d621298ca80b991596bd541b52c7b110856

Parent Marlin image:
sha256:f2700e99e613ac648fc034026e1f705039c25972b7424e2f4c6435d38177cac6

## 3. Architecture

Each GPU retains independent KV tensors for its owned layers.

Rank 0:
- Tokenizes incoming prompt
- Broadcasts token IDs to rank 1
- Finds the longest locally reusable prefix
- Processes uncached prompt tokens through layers 0-19
- Sends intermediate activations to rank 1

Rank 1:
- Finds the longest locally reusable prefix
- Processes received activations through layers 20-39
- Computes next-token logits

Both ranks:
- Agree on the minimum reusable token count using NCCL
- Restore a private KV copy for the agreed prefix
- Fall back to normal prefill if restoration fails
- Snapshot prompt-only KV after prefill, before decoding

## 4. Cache Design

- Exact token-ID prefix matching
- Whole-prefix snapshots
- LRU eviction
- Configurable entry count and byte budget
- GPU-local storage
- Model-key and namespace separation
- No cross-request persisted disk cache
- No KV transfer between GPUs
- At least one uncached prompt token retained to compute logits

Environment:
LLM_PREFIX_CACHE_ENABLED=1
LLM_PREFIX_CACHE_MAX_MIB=256
LLM_PREFIX_CACHE_MAX_ENTRIES=4
LLM_PREFIX_CACHE_MIN_TOKENS=64
LLM_GPTQ_BACKEND=marlin

The cache is disabled by default in the image.

Cache capacity is configured per process/GPU stage.

## 5. Source Files

src/gpu/prefix_cache.py
- Token-prefix lookup
- LRU storage and eviction
- Capacity and reuse statistics

src/gpu/prefix_kv.py
- DynamicCache prompt snapshot
- Independent KV restoration
- Partial-prefix cropping

src/gpu/qwen3_serve.py
- Distributed cache-hit agreement
- Prompt token broadcast
- Prefill integration
- Snapshot creation
- Per-request cache metrics

Docker:
docker/Dockerfile.v2-prefix
buildspec-v2-prefix.yml

Tests:
tests/test_v2_prefix_cache.py
tests/test_v2_prefix_kv.py
scripts/test_v2_prefix_distributed.py
scripts/test_v2_prefix_eviction.py

## 6. CPU Tests

13/13 passed:
- Prefix cache hits and misses
- Longest prefix selection
- Exact prompt replay
- Model and namespace separation
- LRU eviction
- Oversized-entry rejection
- Partial KV restoration
- Snapshot immutability
- Layer validation

These tests do not replace GPU correctness testing.

## 7. Distributed Correctness

Both Kubernetes pods:
- Ready: true
- Restarts: 0

Initial distributed test:
- 3 completed requests
- 0 failed requests

Cold Request A:
- Input tokens: 860
- Output tokens: 48
- Cache reused: 0
- Snapshot stored: yes
- Prefix cache memory: 68.67 MiB
- Stage-0 prefill GPU: 469.838 ms
- TTFT: 949.321 ms
- E2E: 3199.242 ms

Warm Request A:
- Input tokens: 860
- Output tokens: 48
- Cache reused: 859
- Snapshot stored: yes
- Stage-0 prefill GPU: 22.898 ms
- TTFT: 52.754 ms
- E2E: 2319.481 ms

Correctness:
Cold and warm generated token IDs matched exactly.

Shared-prefix Request B:
- Input tokens: 861
- Reused tokens: 843
- Stage-0 prefill GPU: 33.614 ms
- TTFT: 126.461 ms
- E2E: 2390.254 ms
- Snapshot stored: yes

## 8. Observed Speedups

Cold A vs Warm A:
- Stage-0 prefill GPU: 20.52x faster
- TTFT: 18.00x faster
- End-to-end latency: 1.38x faster

These are single-run comparisons, not benchmark medians.
The first long request may include first-use overhead.

## 9. LRU and Partial-Prefix Validation

Four independent eviction prompts were processed.

Observed cache entries: 3
Observed memory: approximately 200 MiB
Configured capacity: 256 MiB

Cold Question B:
- Reused tokens: 0
- TTFT: 657.34 ms

Warm Question B:
- Reused tokens: 860
- TTFT: 52.57 ms
- TTFT speedup: approximately 12.50x

Correctness:
- Original partial-prefix output
- Newly computed cold output
- Fully cached output

All generated token ID sequences matched exactly.

The 256 MiB budget prevented unbounded KV retention.
Eviction was confirmed by the cold Question B miss.

## 10. Benchmark Artifacts

benchmarks/prefix_correctness_20261103T093306Z.json
benchmarks/prefix_eviction_20261103T093714Z.json

Earlier Marlin-only benchmark:
benchmarks/qwen3_14b_marlin_full_20261103T081558Z.json

Marlin-only 130-input/64-output baseline:
- E2E median: 3144.55 ms
- TTFT median: 108.54 ms
- Decode interval median: 48.00 ms/token

Do not directly compare the 130-token baseline with
the 860-token prefix-cache experiment as an overhead measure.

## 11. Limitations and Outstanding Work

- No controlled same-prompt cache-disabled versus
  cache-enabled cold performance comparison yet
- No extensive repeated-run latency distribution
- No block-level caching or shared-prefix deduplication
- No concurrency support
- Cache state lost on process restart
- Memory accounting covers retained snapshot tensor data,
  not all allocator fragmentation or temporary allocations
- No cross-pod KV recovery
- Only dense DynamicLayer snapshots supported
- Security-domain isolation currently relies on configuration;
  not suitable for arbitrary untrusted multi-tenant sharing
- Dynamic model identity needs stronger weight/revision
  fingerprinting before full framework generalization
- Additional model families and attention backends untested
- Longer-duration load and eviction stress tests pending

## 12. Next Optimization

FlashAttention/SDPA comparison.

First establish which attention kernel the current SDPA path
actually uses. Then evaluate a compatible FlashAttention-2
implementation while preserving:

- GPTQ INT8 and Marlin
- Correct RoPE positions
- Prefix KV reuse
- Causal masking for nonempty cached prefixes
- Pipeline-stage correctness
- Existing distributed request protocol

Benchmark cold and warm prefill, decode, end-to-end latency,
peak VRAM, and generated-token correctness.

## 13. Shutdown Operations

See docs/EKS_SHUTDOWN_RUNBOOK.md.

Always:
1. Scale qwen3 StatefulSet to 0.
2. Wait for both GPU pods to terminate.
3. Scale GPU and CPU nodegroups to 0.
4. Complete Terminate-LC-Hook for any instance in
   Terminating:Wait after confirming workloads are stopped.
5. Verify actual EC2 instance termination and ASG states.

The CPU t3.large has repeatedly lingered in Terminating:Wait.
The GPU group can also experience this.

Keep EBS model volumes for future deployments.
