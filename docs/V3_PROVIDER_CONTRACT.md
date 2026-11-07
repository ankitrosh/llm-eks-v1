# V3 — Bring Your Own Model Provider Contract

## Purpose

V3 allows model-specific execution code to be supplied through
a configurable factory without modifying the scheduler or transport.

Automatic support for arbitrary Hugging Face model IDs is not a goal.

## Factory contract

Configuration provides:

    factory_path: module:Factory
    kwargs: dictionary

The framework executes:

    constructor = resolve_factory(factory_path)
    provider = constructor(**kwargs)
    executor = provider(rank)

The executor must be callable:

    executor(task: StageTask, cache) -> StageResult

`GenericStageFactory` wraps the executor in `ValidatedStage`.

## StageTask

The currently supported task fields are:

- request_id: identity of an inference request
- step: generation step
- position: positional context supplied by the scheduler
- payload: token IDs or incoming stage activations

Do not assume the payload always has the same shape on all ranks.

## StageResult

Return:

- payload: new activations or final-stage output
- cache_state: updated request-specific cache

The framework treats cache_state as opaque.

The provider owns the cache representation and the correctness
of model-specific cache updates.

## What the user implements

A new model integration typically supplies:

1. A factory accepting configuration and a stage rank.
2. Partition-specific stage construction.
3. Checkpoint loading for each stage.
4. A forward pass for prefill and decode.
5. Attention behavior and model-specific KV-cache updates.
6. Correct input and output tensor shapes.
7. Final logits suitable for a token selector.

The provider may reuse an existing Hugging Face attention module.
Writing a new attention kernel is not normally required.

## What the framework provides

- Dynamic provider loading
- StageTask and StageResult validation
- Request admission and pipeline scheduling
- Per-request cache lifecycle coordination
- Stage-to-stage communication
- Request completion and release signaling
- Scheduler and transport metrics

## Current generic NCCL runner constraints

- Two ranks, one CUDA device per rank
- Decoder-only autoregressive execution
- PyTorch-compatible stage executors
- NCCL communication between stages
- Floating-point activation tensor transport
- Model-specific checkpoint preparation
- Fixed output-token count for the generic example
- Default final-stage logits shape: [1, sequence, vocabulary]

A custom token selector can adapt compatible final-stage outputs.

The runner is not a universal multimodal or MoE execution engine.

## Attention configuration

The framework does not require a particular attention algorithm.

The provider can:

- Reuse its model's existing attention modules
- Choose a supported PyTorch/Transformers attention backend
- Supply custom attention code

Backend availability and compatibility are the provider's
responsibility unless separately validated by framework tooling.

Do not claim automatic FlashAttention selection solely because
the provider accepts an attention_backend parameter.

## Verification

First validate imports and configuration:

    python scripts/v3_provider_preflight.py \
      --config examples/v3_byom_qwen3.json

Run CPU contract tests:

    python -m pytest -q tests/test_v3_byom_contract.py

For a new model, add model-specific tests covering:

- Partition coverage and stage ownership
- Correct checkpoint weight mapping
- Prefill versus incremental decode
- Position IDs and attention masks
- Cache reuse and request isolation
- Final logits versus an unsplit reference implementation
- Concurrent versus sequential generated token IDs
- Error handling and cleanup

Then validate the distributed execution path on actual hardware.

## Reference implementations

- examples/v3_toy_provider.py
- src/llm_runtime/v3_execution_factory.py
- src/llm_runtime/v3_partition_utils.py
- src/llm_runtime/providers/v3_qwen3_stage_only_factory.py
- tests/test_v3_execution_factory.py
- tests/test_v3_byom_contract.py
- scripts/v3_generic_nccl_runner.py
- docs/V3_FREEZE_2026-11-07.md

## Compatibility commitment

A supported custom provider should not require modifying:

- v3_pipeline_scheduler.py
- v3_ordered_rank_transport.py
- v3_execution_factory.py

Providers with requirements outside the current stage or
transport contracts may need future framework extensions.
