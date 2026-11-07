# V3.1.6 — Model Adapter and Attention Extension Validation

## Goal

Verify that external model adapters and per-layer attention
extensions execute correctly through the framework before
introducing concurrent pipeline scheduling.

## Phase A: CPU

- [ ] External model adapter discovered and instantiated
- [ ] Tiny GPT-2 reference model executes
- [ ] Two-stage logits match unsplit execution
- [ ] Attention plugin installs only on selected layers
- [ ] Identity attention wrapper preserves outputs
- [ ] Controlled attention modification changes outputs
- [ ] Nonselected attention layers remain native
- [ ] Prefill and incremental decoding agree
- [ ] KV-cache and causal mask correctness verified
- [ ] Invalid extensions fail with clear errors

## Phase B: Single GPU

- [ ] Adapter and attention plugins execute on CUDA
- [ ] Native and identity-wrapped outputs agree within tolerance
- [ ] Multi-token generation succeeds
- [ ] Supported dtype and device combinations pass
- [ ] No unexpected GPU memory growth

## Phase C: Distributed GPU

- [ ] Real two-rank execution with adapter
- [ ] Custom attention executes on selected rank/layers
- [ ] Reference and custom-identity outputs agree
- [ ] Multi-token generation works end to end
- [ ] No NCCL deadlock or unexpected GPU memory growth
- [ ] Worker shutdown verified

## Regression requirements

- [ ] Existing V3.1 CPU tests pass
- [ ] Existing Qwen3 sequential runtime remains supported
- [ ] All test outputs and limitations documented
- [ ] Changes committed before concurrency development

## Constraints

Keep production Qwen3 implementation backward compatible.
Do not add N-stage scheduling or concurrent execution here.
Do not claim arbitrary-model universal compatibility.
