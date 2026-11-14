# Repeating the vLLM benchmark

An earlier V3.3 benchmark record reported approximately **55.19 aggregate output tokens/sec** with four concurrent requests on two NVIDIA A10Gs.

This is a historical measurement, not a new benchmark result. Its original output log has not yet been verified in the current repository.

The repository contains `scripts/v3_3_benchmark_gateway.py` for repeating the benchmark.

## Configuration

- Model: Qwen3-14B BF16
- Hardware: Two NVIDIA A10Gs
- Backend: vLLM
- Pipeline parallelism: 2
- Tensor parallelism: 1
- Concurrency: 1, 2, and 4
- Output: Up to 64 tokens per request
- Streaming responses with token usage collection

The benchmark performs warm-up and repeated requests.

It reports:

- Aggregate generated tokens per second
- Time to first token (TTFT)
- Time per output token (TPOT)
- End-to-end request latency (E2E)

Different prompt prefixes help avoid accidental prefix-cache reuse.

## Step 1: Verify the deployment

Use the correct Kubernetes context.

```bash
kubectl config current-context

kubectl -n llm get deploy \
  v33-vllm-rank0 \
  v33-vllm-rank1 \
  v33-coordinator \
  v33-recovery

kubectl -n llm get pods -o wide
```

Do not start or scale GPU workers just for benchmarking without reviewing the cluster state and AWS cost.

## Step 2: Forward the gateway port

Once the reference deployment is healthy:

```bash
kubectl -n llm port-forward \
  svc/v33-coordinator 8080:8080
```

Keep this running in one CloudShell terminal.

In another terminal:

```bash
curl -fsS http://127.0.0.1:8080/readyz \
  | python3 -m json.tool
```

Proceed only when the gateway reports readiness.

## Step 3: Run the benchmark

```bash
mkdir -p test-results/v33-bench

python3 scripts/v3_3_benchmark_gateway.py \
  | tee "test-results/v33-bench/vllm-$(date -u +%Y%m%dT%H%M%SZ).log"
```

The script benchmarks concurrency levels 1, 2, and 4.

## Understanding the result

Aggregate throughput counts all generated output tokens across the concurrent requests and divides by wall-clock time.

It is not the generation speed of an individual request.

When comparing runs, preserve the model, precision, hardware, prompt lengths, output-token settings, runtime image, and scheduler configuration.

The historical custom-runtime and vLLM measurements used different configurations. They should not be described as a controlled percentage speedup.

Save the complete output log before shutting down.

## AWS cost reminder

After a GPU benchmark, follow the project's shutdown runbook.

Scaling a nodegroup to zero does not necessarily mean the EC2 instances have already terminated.

See [EKS shutdown runbook](EKS_SHUTDOWN_RUNBOOK.md).
