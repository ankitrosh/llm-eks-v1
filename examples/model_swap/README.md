# Workflow 2: Model Configuration Replacement

This example validates model metadata and generates an
offline distributed vLLM launch plan.

It does not deploy a new model to Kubernetes.

## Supported planning configuration

- Qwen3ForCausalLM architecture
- Checkpoint-index-verified model layout
- BF16, non-quantized checkpoints
- Two nodes with one GPU each
- Pipeline parallelism 2, tensor parallelism 1

Other configurations are not validated by this example.

## Run the reference profile

From the repository root, with the package installed:

```bash
python3 scripts/v3_5_plan_model_swap.py \
  --profile examples/model_swap/profile.json \
  --output-dir test-results/model-swap
```

The output is `execution_plan.json`, containing a
validated distributed vLLM command for each rank.

## Prepare another Qwen3 checkpoint

Obtain the checkpoint's actual `config.json` and
`model.safetensors.index.json`.

Generate new metadata using:

```bash
python3 scripts/v3_4_generate_model_artifacts.py \
  --config /path/to/config.json \
  --index /path/to/model.safetensors.index.json \
  --model-id YOUR_MODEL_ID \
  --revision YOUR_PINNED_REVISION \
  --output-dir /tmp/qwen3-model-artifacts
```

Copy `profile.json` and update:

- expected_model_id
- model_capabilities_path
- model_layout_path
- deployment.model_path
- deployment.max_model_len, if needed

Then run the planner with the updated profile.

The model metadata must describe the same checkpoint
that will be mounted at deployment.model_path.

## Limitations

Model metadata validation is not a full checkpoint-weight
integrity check or a GPU memory capacity test.

Successful planning does not prove that a new model
can fit on the selected GPUs or run successfully.

The existing V3.3 Kubernetes manifest generator still
targets the fixed Qwen3-14B reference deployment.

Generating and deploying Kubernetes manifests for
alternative models requires separate deployment
integration and validation.

No AWS or Kubernetes operations are performed by
this workflow.
