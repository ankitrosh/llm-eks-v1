# V3.1 GPU Validation

Date: 2026-11-07
AWS region: us-east-2
EKS cluster: llm-pipeline-v1

## Verified CPU execution

- Real three-process Gloo communication: PASS
- Three-rank Qwen3 reference parity: PASS
- Four-rank Qwen3 reference parity: PASS
- 27 V3.1 unit tests: PASS

## Verified fractional-L4 execution

Hardware: AWS g6f.large / NVIDIA L4-3Q
CUDA: PyTorch 2.14.1+cu130
GPU memory: approximately 2.79 GiB visible per GPU

### N=3, Gloo-staged GPU transport

- Distributed Qwen3 reference parity: PASS
- Three requests, three generated tokens each
- Remote forwards: 9 per remote rank
- Remote cache releases: 3 per remote rank
- Coordinator: sent=9, received=9, hops=9, releases=3
- Generated output sequences:
  - (4, 4, 64)
  - (22, 73, 22)
  - (48, 48, 48)

### N=4, Gloo-staged GPU transport

- Distributed Qwen3 reference parity: PASS
- Three requests, three generated tokens each
- Remote forwards: 9 per remote rank
- Remote cache releases: 3 per remote rank
- Coordinator: sent=9, received=9, hops=18, releases=3
- Generated output sequences match N=3 results.

The Qwen3 integration tests use a tiny synthetic six-layer
checkpoint, not a pretrained production-scale model.

## NCCL status

- Existing V3 two-A10G NCCL pipeline: previously validated.
- V3.1 N-rank NCCL on fractional L4-3Q: FAILED during
  communicator initialization.
- NCCL version: 2.30.7
- Error: CUDA operation not supported
- V3.1 N=2 NCCL on full A10Gs: PENDING
- V3.1 N>2 NCCL: UNVERIFIED

## Scope

These results establish correct N=3 and N=4 GPU execution
using staged Gloo for the tested workloads.

They do not establish production model performance, automatic
memory-aware deployment, complete fault recovery, or NCCL
correctness beyond the configurations explicitly tested.
