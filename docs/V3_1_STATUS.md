# V3.1 — Development Milestone

## Implemented
- Configurable N-rank pipeline parallelism.
- Manual and memory-aware automatic partitioning.
- Qwen3 stage-only checkpoint loading.
- GPTQ INT8 Marlin and custom quantization provider interfaces.
- NCCL/Gloo communication and ordered execution.
- Per-request KV caching and concurrent scheduling.
- Launch-plan validation and failure detection.
- Config-driven Docker entrypoint and reusable Kubernetes Job.

## Configuration
- configs/v3_1/: model, partition, quantization and transport.
- deploy/templates/v3_1_indexed_nccl_job.yaml: Kubernetes settings.
- scripts/v3_1_configured_nccl.py: one-shot inference entrypoint.

The Kubernetes rank count must match pipeline.num_ranks.

## Validation
- Qwen3-0.6B GPTQ INT8 on 2 AWS A10Gs.
- 14/14 layer partition with NCCL Socket.
- Both ranks completed and launch hashes matched.
- 4 generated tokens: [279, 3491, 374, 311].
- 80 selected CPU regression tests passed.
- Frozen V3 checksums passed.
- Kubernetes test resources removed.
- GPU ASG returned to desired=0, instances=[].

## Limitations
- New Kubernetes entrypoint is one-shot, not an HTTP server.
- Config-driven GPU validation currently covers 2 ranks.
- Arbitrary custom quantization backends are not GPU validated.
- Kubernetes deployment requires a compatible image and checkpoint.

## Next: V3.2
- Continuous batching.
- Dynamic request admission.
- Batched prefill and decode.
- Performance and concurrency benchmarks.

First official release: after V3.2.
