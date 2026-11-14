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

The V3.5 renderer supports an isolated GPU-worker
manifest bundle for validated alternative Qwen3 models.

It does not generate a complete independent coordinator,
recovery controller, namespace, or model PVC resources.

No AWS or Kubernetes operations are performed by
this workflow.

## Generate an isolated Kubernetes bundle

After generating and validating the alternative model's
execution plan, run:

```bash
python3 scripts/v3_5_render_model_swap.py \
  --profile test-results/v35-qwen3-8b-profile.json \
  --plan test-results/v35-qwen3-8b-plan/execution_plan.json \
  --output-dir test-results/v35-qwen3-8b-bundle \
  --namespace llm-model-swap \
  --name v35-qwen3-8b \
  --pvc-rank0 model-files-qwen3-8b-0 \
  --pvc-rank1 model-files-qwen3-8b-1
```

The resulting bundle contains five Kubernetes manifests:
two GPU deployments, two services, and model metadata.

The renderer validates the supplied plan against the
model profile, preserves the pinned reference image and
runtime policy, and rewrites the distributed master
address for the isolated namespace.

All generated deployments start with zero replicas.

### Before deploying

The namespace, two model PVCs, checkpoint files, GPU
nodes, networking, and NVIDIA device plugin must exist
and be correctly configured.

PVC names supplied to the renderer are references,
not instructions to create storage.

The model files mounted at deployment time must match
the checkpoint used to produce the metadata.

A dedicated coordinator and recovery controller for
the alternative namespace are not generated. The
metadata ConfigMap is not mounted into a controller
by this GPU-only bundle.

The Qwen3-8B path has been validated for offline
inspection, planning, and manifest generation.
GPU loading, memory capacity, and live inference
have not been validated for that model.

Do not apply this bundle to a live cluster without
reviewing and completing its deployment dependencies.
