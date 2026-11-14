"""Validate a Qwen3 profile and generate a CPU-only vLLM plan."""

import argparse
import json
from pathlib import Path

from llm_runtime.extensions import (
    PluginRegistry,
    load_plugin_context,
)
from llm_runtime.backends.vllm import register_vllm_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())

    required = {
        "schema_version",
        "backend",
        "expected_model_id",
        "model_capabilities_path",
        "model_layout_path",
        "deployment",
    }

    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("Invalid model-swap profile fields")

    if profile["schema_version"] != 1:
        raise ValueError("Unsupported profile schema")

    if profile["backend"] != "vllm":
        raise ValueError("This workflow requires vLLM")

    deployment = profile["deployment"]

    if not isinstance(deployment, dict):
        raise ValueError("Deployment must be an object")

    topology = (
        deployment.get("nnodes"),
        deployment.get("gpus_per_node"),
        deployment.get("pp"),
        deployment.get("tp"),
    )

    if topology != (2, 1, 2, 1):
        raise ValueError(
            "This workflow currently supports "
            "2 nodes, 1 GPU/node, PP=2, TP=1"
        )

    if deployment.get("dtype") != "bfloat16":
        raise ValueError("This workflow requires BF16")

    ctx = load_plugin_context(
        model_capabilities_path=profile[
            "model_capabilities_path"
        ],
        model_layout_path=profile["model_layout_path"],
        backend="vllm",
        runtime={"deployment": deployment},
        require_verified=True,
    )

    if ctx.model["model_id"] != profile["expected_model_id"]:
        raise ValueError("Model ID does not match profile")

    if (
        ctx.model["architecture"] != "Qwen3ForCausalLM"
        or ctx.model.get("model_type") != "qwen3"
    ):
        raise ValueError("Unsupported model architecture")

    if ctx.model.get("quantization_config") is not None:
        raise ValueError(
            "Quantized models are not validated by this workflow"
        )

    if ctx.model.get("checkpoint_dtype") != "bfloat16":
        raise ValueError("Expected BF16 checkpoint metadata")

    if ctx.model["num_layers"] < 2:
        raise ValueError(
            "Two pipeline stages require at least two layers"
        )

    registry = PluginRegistry()
    register_vllm_backend(registry)

    backend = registry.build("backend", "vllm", ctx)
    plan = backend.plan()

    assert len(plan["nodes"]) == 2

    output = args.output_dir / "execution_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(plan, indent=2) + "\n"
    )

    print("Model:", ctx.model["model_id"])
    print("Architecture:", ctx.model["architecture"])
    print("Layers:", ctx.model["num_layers"])
    print("Ranks:", len(plan["nodes"]))
    print("Execution plan:", output)
    print("GPU memory compatibility: NOT VERIFIED")
    print("No Kubernetes changes performed")
    print("V35_MODEL_SWAP_PLAN_PASS")


if __name__ == "__main__":
    main()
