# Test guide

This guide briefly describes the purpose of each test module in the repository.

Some tests belong to the current framework and others preserve coverage for earlier custom inference experiments.

Passing CPU tests does not automatically prove that live distributed GPU inference works.

## Model discovery and V2

- `test_model_metadata.py`: Model metadata parsing and validation.
- `test_optimization_registry.py`: Optimization capability checks and selection.
- `test_runtime_discovery.py`: Runtime feature and compatibility discovery.
- `test_v2_prefix_cache.py`: Prefix matching, cache limits, and eviction.
- `test_v2_prefix_kv.py`: KV-cache snapshot and restoration.
- `test_v2_sdpa_lower_right.py`: Causal masking with cached prefixes.

## V3.1 partitioning and quantization

- `test_v3_1_checkpoint_profile.py`: Checkpoint configuration and layout profiles.
- `test_v3_1_communication_api.py`: Distributed communication interfaces.
- `test_v3_1_custom_quantization.py`: Custom quantization provider integration.
- `test_v3_1_generation_supervisor.py`: Worker lifecycle and failure handling.
- `test_v3_1_gptq_int8_inspection.py`: GPTQ INT8 checkpoint metadata.
- `test_v3_1_gptq_int8_qwen3_checkpoint.py`: Qwen3 GPTQ checkpoint assumptions.
- `test_v3_1_gptq_int8_torch.py`: GPTQ execution helpers.
- `test_v3_1_gptq_marlin_provider.py`: Marlin backend integration.
- `test_v3_1_launch_spec.py`: Distributed launch configuration.
- `test_v3_1_multistage_scheduler.py`: Scheduling across model stages.
- `test_v3_1_nrank_transport.py`: Multi-rank transport interfaces.
- `test_v3_1_partition_planner.py`: Layer partition planning.
- `test_v3_1_public_gptq_config.py`: GPTQ configuration validation.
- `test_v3_1_quantization_api.py`: Quantization extension contracts.
- `test_v3_1_quantized_qwen3_execution.py`: Quantized Qwen3 stage execution.
- `test_v3_1_qwen3_layout.py`: Qwen3 model layer layout.
- `test_v3_1_qwen3_tied_embeddings.py`: Shared embedding handling.
- `test_v3_1_runtime_config.py`: Runtime configuration validation.
- `test_v3_1_startup_protocol.py`: Distributed startup and compatibility checks.

## V3.2 batching and KV-cache research

- `test_v3_2_async_gloo.py`: Asynchronous CPU/Gloo communication.
- `test_v3_2_async_gloo_qwen3.py`: Asynchronous Gloo with Qwen3 stages.
- `test_v3_2_batch_planner.py`: Request batching decisions.
- `test_v3_2_block_attention.py`: Block-oriented attention.
- `test_v3_2_block_engine.py`: Engine integration with KV blocks.
- `test_v3_2_concurrent_planner.py`: Concurrent request planning.
- `test_v3_2_continuous_engine.py`: Continuous batching behavior.
- `test_v3_2_custom_batch_policy.py`: Custom batching policies.
- `test_v3_2_gloo_failure.py`: Gloo transport failure behavior.
- `test_v3_2_gloo_qwen3.py`: Qwen3 execution over CPU/Gloo.
- `test_v3_2_gloo_release.py`: Request release and transport cleanup.
- `test_v3_2_gloo_wire.py`: Gloo message protocol.
- `test_v3_2_kv_attention_oracle.py`: KV-aware attention correctness checks.
- `test_v3_2_kv_blocks.py`: KV block allocation and ownership.
- `test_v3_2_kv_cleanup_concurrency.py`: KV cleanup during concurrency.
- `test_v3_2_kv_storage.py`: KV storage contracts.
- `test_v3_2_microbatch_transport.py`: Microbatch communication.
- `test_v3_2_mixed_transport.py`: Mixed transport behavior.
- `test_v3_2_nrank_cpu_batch.py`: CPU batching across multiple ranks.
- `test_v3_2_pipeline_policy.py`: Pipeline scheduling policies.
- `test_v3_2_policy_transport.py`: Policy and transport integration.
- `test_v3_2_qwen3_block_decode.py`: Qwen3 decoding with KV blocks.
- `test_v3_2_qwen3_cpu_batch.py`: Qwen3 CPU batch execution.
- `test_v3_2_qwen3_late_arrival.py`: Requests joining active batches.
- `test_v3_2_qwen3_variable_batch.py`: Changing batch sizes.
- `test_v3_2_scheduler_api.py`: Scheduler interface validation.

## V3.3 vLLM and Kubernetes coordination

- `test_v3_3_artifact_context.py`: Model metadata and execution context.
- `test_v3_3_coordinator.py`: Gateway readiness, proxying, and streaming.
- `test_v3_3_engine_controller.py`: GPU deployment lifecycle and health.
- `test_v3_3_extensions.py`: Backend and scheduler extension contracts.
- `test_v3_3_recovery_watchdog.py`: Recovery state and failure handling.
- `test_v3_3_vllm_backend.py`: Distributed vLLM launch planning.

## V3.4 customization

- `test_v3_4_1_k8s_resources.py`: CPU, memory, and GPU resource overrides.
- `test_v3_4_external_inspector_cli.py`: External model inspector loading.
- `test_v3_4_inspection.py`: Inspector registry and layout validation.

## V3.5 CLI and deployment tools

- `test_v3_5_aws_configure.py`: AWS discovery configuration.
- `test_v3_5_aws_cpu.py`: CPU nodegroup discovery.
- `test_v3_5_aws_doctor.py`: Read-only deployment diagnostics.
- `test_v3_5_aws_enrich.py`: GPU nodegroup and quota information.
- `test_v3_5_aws_storage.py`: Existing model storage detection.
- `test_v3_5_cli.py`: CLI commands and safe behavior.
- `test_v3_5_custom_extensions.py`: Custom extension workflows.
- `test_v3_5_model_swap.py`: Compatible model replacement planning.
- `test_v3_5_render_discovered.py`: Configuration-based manifest generation.
- `test_v3_5_render_model_swap.py`: Model-swap GPU resource generation.

## V3 core framework

- `test_v3_activation_protocol.py`: Activation messages and tasks.
- `test_v3_attention_plan.py`: Attention compatibility and planning.
- `test_v3_byom_contract.py`: Bring-your-own-model provider interfaces.
- `test_v3_execution_factory.py`: Custom stage factory construction.
- `test_v3_generic_nccl_runner_config.py`: NCCL runner configuration.
- `test_v3_manifest.py`: Runtime and model manifests.
- `test_v3_model_inspector.py`: Model inspection behavior.
- `test_v3_pipeline_scheduler.py`: Pipeline request scheduling.
- `test_v3_pipeline_scheduler_lifecycle.py`: Request lifecycle and cleanup.
- `test_v3_pipeline_scheduler_reliability.py`: Cancellation and failures.
- `test_v3_plan_loader.py`: Execution plan loading.
- `test_v3_plan_to_stage.py`: Mapping plans to model stages.
- `test_v3_plugin_registry.py`: Plugin registration and compatibility.
- `test_v3_process_transport.py`: Process-level transport.
- `test_v3_provider_preflight.py`: Provider compatibility checks.
- `test_v3_qwen3_adapter.py`: Qwen3 model adapter.
- `test_v3_qwen3_last_logits.py`: Final-token logits handling.
- `test_v3_qwen3_partition.py`: Qwen3 layer partitioning.
- `test_v3_qwen3_stage_only.py`: Stage-specific model loading.
- `test_v3_runtime_plan.py`: Runtime execution plan validation.
- `test_v3_stage_bridge.py`: Stage input and output handling.
- `test_v3_tensor_limits.py`: Tensor sizes and shape limits.

## Running the maintained CPU tests

From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_v3_3_*.py'
python3 -m unittest discover -s tests -p 'test_v3_4*.py'
python3 -m unittest discover -s tests -p 'test_v3_5*.py'
```

These cover the maintained framework interfaces and reference deployment preparation.

GPU inference tests require compatible hardware, model weights, and a running distributed environment.

See [benchmark instructions](BENCHMARK.md) for live inference measurements.
