# Distributed LLM Inference on AWS EKS
## V2 Milestone Report — Qwen3-32B GPTQ INT8

**Milestone completed:** November 2, 2026 (Pacific) / November 3, 2026 (UTC)

**Status:** Functional inference and benchmarking complete.
Deployment reproducibility changes built and configured, but not
yet validated end-to-end on GPUs.

---

## 1. Project Objective

Build an open-source, PyTorch-based distributed inference framework
that allows models larger than an individual GPU's usable memory
to run across multiple smaller GPUs.

The long-term framework should manage:

- Model partitioning and stage loading
- Distributed communication and synchronization
- Attention backend selection
- Model-specific quantization support
- KV-cache management
- Warmup and inference scheduling
- Batching and concurrent requests
- Fault handling and recovery
- Kubernetes deployment and lifecycle management

The user should eventually select a compatible model and hardware
configuration without having to implement these optimizations.

The current implementation remains Qwen3-specific and is not yet
a universal PyTorch model execution framework.

## 2. Architecture

Current architecture:

    Client
       |
       v
    Rank 0: FastAPI HTTP server
       |
       | Qwen3 embeddings + layers 0-31
       | Local KV cache
       |
       | NCCL activation transfer
       v
    Rank 1: layers 32-63
       | Final normalization + LM head
       | Local KV cache
       |
       | Distributed coordination / token result
       v
    Rank 0: response assembly
       |
       v
    Client

Runtime characteristics:

- Two separate Kubernetes pods
- One A10G GPU assigned to each rank
- PyTorch distributed with NCCL communication
- NCCL over network sockets, without RDMA
- Persistent FastAPI serving process
- Rank-local transformer execution and KV caches
- Sequential request execution (batch size 1)
- SDPA attention backend
- Triton GPTQ quantized linear operations
- Stateless inference recovery through pipeline restart

The exact execution and communication protocol is implemented in
src/gpu/qwen3_serve.py and related runtime files.

The deployment is a Kubernetes StatefulSet named qwen3, using
stable pod identities qwen3-0 and qwen3-1.

## 3. Infrastructure

AWS account: 657173565477
Region: us-east-2
Cluster: llm-pipeline-v1
Namespace: llm

GPU nodegroup: gpu-a10g
GPU instances: 2 x g5.xlarge
GPU hardware: NVIDIA A10G, nominally 24 GiB each

CPU nodegroup: cpu-system

Each GPU rank has its own 80 GiB EBS-backed persistent volume.
The volumes preserve model checkpoint data across pod restarts
and nodegroup scale-downs.

The Kubernetes cluster, EBS volumes, ECR storage, and any other
retained AWS resources may continue incurring charges even after
EC2 worker nodes are stopped.

## 4. Development History

### 4.1 Early distributed inference proof of concept

The project began with distributed transformer execution,
communication experiments, and profiling across CPU and GPU
environments.

Early experiments demonstrated:

- Splitting transformer blocks across processes/machines
- Passing activation tensors between execution stages
- Comparing monolithic and pipeline execution
- Isolating computation, serialization, and communication costs
- Establishing distributed execution correctness

The CPU prototype used DistilGPT2 and FastAPI.
Subsequent GPU experiments used NCCL.

These experiments informed the later Qwen3 pipeline design.

### 4.2 V1: Qwen3-14B BF16

V1 established:

- Qwen3-14B distributed execution
- Two A10G GPUs
- Transformer partition: 20 layers / 20 layers
- NCCL inter-rank communication
- Persistent inference serving
- Docker containerization
- Kubernetes StatefulSet deployment
- Basic health endpoints and operational metrics
- Per-rank persistent model caching

Original V1 image:

657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:6886820e0696db161f219d6b2b45d9851cbb4a9da57a4879333f70fc5f02e718

The original V1 deployment remains available.

### 4.3 V2: SDPA attention optimization

Implemented a configurable SDPA attention backend.

Near-matched Qwen3-14B p128_o64 results:

| Metric | Original BF16 | SDPA BF16 |
|---|---:|---:|
| E2E median | 4470.408 ms | 4372.993 ms |
| TTFT median | 122.332 ms | 118.719 ms |
| Decode interval median | 68.941 ms | 67.425 ms |

Observed E2E improvement: approximately 2.18%.

The attention backend remains configurable through
LLM_ATTENTION_BACKEND.

### 4.4 V2: GPTQ INT8 model support

Selected checkpoint:

JunHowie/Qwen3-32B-GPTQ-Int8

Pinned revision:

74e7e07c621c17241654d9a22290d3f9cb08d25b

Model properties:

- Qwen3-32B architecture
- 64 transformer layers
- GPTQ weight-only INT8 quantization
- 8-bit packed weights
- Group size 128
- Symmetric quantization
- desc_act = false
- 448 quantized linear projections
- Nine checkpoint shards, approximately 32.66 GiB total

The model is distributed as:

Rank 0:
- Layers 0-31
- Input token embeddings
- Required rank-local checkpoint shards

Rank 1:
- Layers 32-63
- Final normalization
- Language model head
- Required rank-local checkpoint shards

The downloader uses the safetensors tensor-to-shard index
to determine which shards each rank requires.

One checkpoint shard can contain tensors from both ranks,
so rank-local shard sets may overlap.

### 4.5 GPTQ integration

The implementation integrates GPTQModel's TritonV2QuantLinear
execution path into the custom Qwen3 stage loader.

The checkpoint uses GPTQ v1 zero-point format. The loader
converts the zero-point representation to the expected v2
format when loading the quantized layers.

Validation performed during development included:

- Model metadata inspection
- Layer ownership and checkpoint shard checks
- GPTQ tensor metadata validation
- GPTQ v1-to-v2 zero-point conversion checks
- Synthetic GPU Triton kernel tests
- Full 32B layer loading on both GPUs
- NCCL synchronization and distributed warmup
- End-to-end generated text validation

Passing these tests establishes support for this checkpoint
and execution configuration, not every GPTQ model.

## 5. Major Problems and Solutions

### 5.1 GPTQ quantization format incompatibility

Problem:
The checkpoint used GPTQ v1 zero-point formatting while
the selected Triton execution path required v2 formatting.

Solution:
Convert the relevant quantized zero-point tensors as part
of stage loading.

Validation:
CPU conversion tests and synthetic GPU execution passed.

### 5.2 Checkpoint too large for a single A10G

Problem:
The checkpoint weights alone are larger than the usable
memory of one A10G.

Solution:
Partition 64 transformer layers into two rank-local
32-layer stages, loading only the relevant weights per rank.

Validation:
Both ranks loaded the full assigned layer sets successfully.

### 5.3 Missing chat template

Problem:
The quantized checkpoint did not provide a chat template
recognized by the tokenizer, causing warmup failure.

Observed error:
Tokenizer chat_template is not set.

Initial workaround:
Install the official Qwen3-32B chat template into
chat_template.jinja on the rank 0 persistent volume.

Permanent source fix:
The model preparation script now checks for a local chat
template and can fetch one from an explicitly selected
base tokenizer repository.

Base tokenizer:
Qwen/Qwen3-32B

Pinned tokenizer revision:
9216db5781bf21249d130ec9da846c4624c16137

The permanent fix is included in the final V2 image,
but the updated init-container flow has not yet been
retested end-to-end on GPUs.

### 5.4 Distributed warmup and rank restarts

After the initial tokenizer failure, subsequent rank
restarts produced NCCL disconnection and unexpected
warmup-control errors.

These errors were observed during failed startup attempts.
The original cause of every subsequent error was not
fully captured, so the precise failure sequence is
not established.

Following tokenizer repair and clean restart,
distributed warmup and serving completed successfully.

Future improvements should capture first-failure logs
from both ranks before automatic restarts obscure them.

### 5.5 Image and deployment reproducibility

Problem:
During V2 experimentation, the runtime used the new
32B image while the model-download init container
still used the original V1 image and downloader.

Solution:
Build a V2 image containing the corrected downloader
and create a pinned model deployment profile.

The final configuration uses the same immutable image
for runtime and model initialization.

## 6. Successful Functional Test

Both ranks loaded all assigned Qwen3-32B GPTQ layers.

Initial allocated GPU memory after loading:

- Rank 0: approximately 16.43 GiB
- Rank 1: approximately 14.98 GiB

The application successfully started FastAPI,
completed distributed warmup, and generated coherent text.

Example request:

"Explain why GPUs are useful for machine learning
in one sentence."

Observed response:

"GPUs are useful for machine learning because they
can perform many parallel computations quickly,
significantly speeding up the training of complex models."

The response returned HTTP 200 with finish_reason=stop.

During the subsequent benchmark session:

- 14 completed requests were reported after the first
  functional test and baseline benchmark.
- Zero failed requests were reported.
- Both pods remained running with zero restarts.

## 7. Benchmark Methodology

Benchmark environment:

- Two A10G GPUs
- Qwen3-32B GPTQ INT8
- 32/32 transformer layer split
- SDPA attention
- Triton GPTQ execution
- Sequential inference requests
- Fixed prompts and max output lengths
- No concurrent load

Baseline benchmark:

- Three warmup requests
- Ten measured requests
- 130 actual input tokens
- 64 actual generated tokens per request

Warmups were excluded from summary statistics.

The benchmark reports application-measured latency
and host-observed request latency separately.

## 8. Qwen3-32B Baseline Results

| Metric | Median | Mean | p95 |
|---|---:|---:|---:|
| E2E latency (ms) | 21942.05 | 21939.59 | 21945.40 |
| TTFT (ms) | 456.63 | 456.69 | 457.47 |
| Decode interval (ms/token) | 341.03 | 340.99 | 341.08 |
| Stage 0 GPU compute (ms) | 10833.80 | 10833.80 | 10835.32 |
| Stage 0 blocking communication (ms) | 11093.02 | 11092.29 | 11098.05 |
| Peak PyTorch allocated (MiB) | 17110.90 | 17110.90 | 17110.90 |
| Client wall latency (ms) | 21950.39 | 21948.16 | 21954.18 |

Approximate decode rate based on median decode interval:
2.93 tokens per second.

The communication-blocking metric includes time
waiting for distributed operations and other-rank work.
It must not be interpreted as pure network transmission
latency.

The measurements were highly consistent under this
sequential fixed-length workload.

Raw benchmark file:

benchmarks/qwen3_32b_int8_20261103T020948Z.json

## 9. Prompt-Length Sweep

For each input-size target:

- One warmup request
- Three measured requests
- 64 generated output tokens
- Median statistics across measured requests

| Input tokens | E2E (ms) | TTFT (ms) | Decode interval (ms) | Peak allocated (MiB) |
|---:|---:|---:|---:|---:|
| 130 | 21935.0 | 456.1 | 340.8 | 17110.9 |
| 484 | 22286.8 | 775.4 | 341.4 | 17200.3 |
| 956 | 22886.4 | 1346.8 | 341.8 | 17319.9 |
| 1782 | 23891.4 | 2252.4 | 343.4 | 17527.9 |

The test prompts targeted approximately 128, 512,
1024, and 2048 input tokens. Actual counts differed
because of tokenizer behavior and prompt construction.

Observations:

1. Longer prompts substantially increase TTFT.
2. Decode interval remains approximately constant.
3. GPU memory allocation increases with sequence length.
4. For 64 output tokens, decoding dominates E2E latency.
5. All measured outputs contained exactly 64 tokens.

Raw sweep file:

benchmarks/qwen3_32b_prompt_sweep_20261103T021935Z.json

## 10. Final GPU Memory Snapshot

NVIDIA-SMI reported:

| Rank | Used memory | Total reported memory |
|---|---:|---:|
| Rank 0 | 18216 MiB | 23028 MiB |
| Rank 1 | 18218 MiB | 23028 MiB |

This corresponds to approximately 17.79 GiB
device-level usage per rank at the time of the snapshot.

Available reported headroom was approximately 4.70 GiB
per device.

These are point-in-time NVIDIA-SMI measurements,
not guaranteed maximum memory usage.

PyTorch peak allocated memory for rank 0 is reported
separately in the benchmark results.

## 11. Comparison with Qwen3-14B BF16

Near-matched 128/130-input, 64-output workloads:

| Metric | 14B BF16 original | 32B GPTQ INT8 |
|---|---:|---:|
| E2E median | 4470.408 ms | 21942.05 ms |
| TTFT median | 122.332 ms | 456.63 ms |
| Decode interval median | 68.941 ms | 341.03 ms |

32B versus original 14B:

- E2E latency: approximately 4.91 times higher
- TTFT: approximately 3.73 times higher
- Decode interval: approximately 4.95 times higher

This is not a controlled quantization benchmark.

The experiments differ in model size, weights,
quantization backend, and slightly in prompt length.

The slowdown cannot be attributed exclusively to GPTQ,
the number of layers, or network communication.

The main success criterion is fitting and serving
the larger model across the two GPUs.

## 12. Build Artifacts

Working, GPU-benchmarked V2 integrated image:

657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:91bea9d0dbb08cd619914719359f0f952a169aa742ec8604025ad59dc4f16510

Final V2 image including downloader fix:

657173565477.dkr.ecr.us-east-2.amazonaws.com/qwen3-14b-pipeline@sha256:bcd962ac9b7deaf2af31591be0d5d336ba5eff7d7d613f2c072391af48290dce

Final image tag:
v2-qwen3-32b-gptq-r2

Final image CodeBuild status:
SUCCEEDED

Important distinction:
The GPU benchmarks were performed using the earlier
integrated image, not the final downloader-fix image.

## 13. Source and Configuration References

Primary implementation:

- src/gpu/qwen3_stage_loader.py
- src/gpu/qwen3_serve.py
- src/gpu/qwen3_prepare_step.py

Build files:

- docker/Dockerfile.v2-integrated
- buildspec-v2-integrated.yml
- buildspec-v2-final.yml

V2 image reference:

- infrastructure/v2-final-image.txt

V2 deployment profile:

- infrastructure/profiles/qwen3-32b-gptq-int8.json

V2 Kubernetes patch:

- infrastructure/profiles/qwen3-32b-gptq-int8.patch.json

Preserved V2 pod template:

- infrastructure/profiles/qwen3-32b-v2-template.json

Original V1 backup:

- infrastructure/backups/qwen3-before-v2.yaml

Additional probe scripts and test fixtures are retained
in the repository for further inspection.

## 14. Deployment and Rollback

Current state:

- Qwen3 StatefulSet restored to original V1
- V1 runtime image restored
- V1 model-download init container restored
- MODE=serve
- MODEL_DIR=/models/qwen3-14b
- SPLIT=20
- Desired replicas=0
- GPU nodegroup scaled down to zero
- CPU nodegroup desired capacity set to zero

The V2 deployment profile has been saved separately.

A future V2 deployment should:

1. Verify AWS resources and persistent volumes.
2. Confirm the final ECR image digest exists.
3. Check that the model configuration is correct.
4. Apply the saved V2 strategic patch to the StatefulSet.
5. Scale GPU capacity to two nodes.
6. Scale the StatefulSet to two replicas.
7. Verify initialization, download cache, and layer loading.
8. Verify NCCL initialization and warmup.
9. Run a short correctness request.
10. Collect health and performance metrics.

Note:
The final V2 initialization flow requires one additional
end-to-end GPU validation before claiming it is fully
reproducible.

For rollback, use the saved original V1 deployment backup
or the prepared V1 restore patch, with replicas kept at zero
until a deliberate deployment is requested.

Do not assume that scaling Kubernetes replicas to zero
automatically terminates EC2 instances. EKS managed nodegroup
capacity must also be scaled down and verified.

## 15. Current Limitations

- Only the Qwen3 execution family is validated.
- Custom arbitrary PyTorch models are not yet supported.
- The existing deployment uses a fixed image.
- The framework does not yet provide automatic support
  for every quantization format.
- Current requests are sequential, batch size 1.
- No continuous batching or concurrent execution.
- No prefix caching.
- KV caches are rank-local and not recoverable
  after a pipeline restart.
- No advanced stage recovery or request replay.
- No dynamic model partition planning.
- GPTQ Triton decode latency is relatively high.
- No controlled comparison against fused GPTQ kernels.
- No validated vLLM performance comparison.
- No long-duration soak test or load test.
- The final downloader-fix image requires an end-to-end test.
- Peak memory across every execution phase was not measured.

## 16. Next Engineering Milestones

### Milestone B: Faster quantized inference

First investigate the GPTQ matrix multiplication backend.

Proposed work:

1. Profile the decode path on the current 32B checkpoint.
2. Separate quantized GEMM time from other model operations.
3. Identify compatible optimized INT8 GPTQ kernels.
4. Evaluate fused implementations where supported.
5. Compare against the existing TritonV2 implementation.
6. Preserve correctness and checkpoint compatibility.
7. Select backends automatically based on validated support.

Do not assume a kernel optimized for 4-bit GPTQ necessarily
supports this checkpoint's 8-bit packed format.

Baseline to beat:

- E2E median: 21942.05 ms
- TTFT median: 456.63 ms
- Decode interval median: 341.03 ms/token

Use comparable actual prompt/output token counts,
hardware, and precision when benchmarking.

### Milestone C: Batching and concurrency

- Introduce coordinated multi-request execution.
- Extend rank communication for multiple active sequences.
- Implement KV-cache allocation and request lifecycle tracking.
- Add batching and scheduling.
- Measure latency-throughput tradeoffs.
- Test under realistic concurrent traffic.

### Milestone D: Caching and optimized serving

- Prefix KV-cache reuse
- Cache limits and eviction policies
- Better health checks and failure recovery
- Compare against vLLM or other suitable serving engines
- Refine attention backend selection

### Milestone E: Framework generalization

- Model capability validation
- Model-independent configuration contracts
- Adapter interfaces for supported model families
- Automatic layer assignment and checkpoint mapping
- Configurable container and dependency handling
- Easier user-provided model integration
- More robust orchestration and recovery

Generalization should be incremental and tested.
The current success with Qwen3 does not yet prove
compatibility with arbitrary PyTorch architectures.

## 17. Milestone Conclusion

The primary V2 proof of concept succeeded.

Qwen3-32B GPTQ INT8 was loaded and served using a custom
two-rank NCCL pipeline on two NVIDIA A10G GPUs.

We validated coherent response generation, stable
sequential inference, multiple prompt lengths, and
per-rank memory feasibility.

The largest remaining performance issue is decode latency,
which motivates evaluating a more efficient quantized
execution backend before expanding into batching
and concurrent serving.

The next phase should preserve the working Qwen3-32B
baseline and optimize the inference engine without
unnecessarily redesigning the distributed architecture.
