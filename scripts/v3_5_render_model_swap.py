"""Render isolated vLLM GPU manifests from a validated model plan.

Offline only. No Kubernetes or AWS API calls.
"""

import argparse
import copy
import json
import re
from pathlib import Path

from llm_runtime.backends.vllm import register_vllm_backend
from llm_runtime.extensions import PluginRegistry, load_plugin_context


ROOT = Path("infrastructure/profiles/v3-3/qwen3-14b")
TEMPLATES = Path("deploy/v3_3/rendered")
DNS = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def command_flag(command, flag, value):
    require(command.count(flag) == 1, f"Missing/duplicate {flag}")
    command[command.index(flag) + 1] = str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pvc-rank0", required=True)
    parser.add_argument("--pvc-rank1", required=True)
    args = parser.parse_args()

    for value in (
        args.namespace, args.name,
        args.pvc_rank0, args.pvc_rank1,
    ):
        require(bool(DNS.fullmatch(value)),
                f"Invalid Kubernetes resource name: {value}")

    require(
        len(args.name) <= 54,
        "Name too long for generated resources",
    )
    require(
        args.name not in {"v33-vllm", "v33-vllm-api"},
        "Reserved default deployment name",
    )
    require(
        args.pvc_rank0 != args.pvc_rank1,
        "Separate per-rank PVC names required",
    )

    profile = read(args.profile)
    plan = read(args.plan)

    require(
        profile.get("schema_version") == 1
        and profile.get("backend") == "vllm",
        "Invalid model profile",
    )

    deployment = profile["deployment"]

    require(
        (
            deployment.get("nnodes"),
            deployment.get("gpus_per_node"),
            deployment.get("pp"),
            deployment.get("tp"),
        ) == (2, 1, 2, 1),
        "Only the validated two-worker topology is supported",
    )
    require(
        deployment.get("dtype") == "bfloat16",
        "Only BF16 supported",
    )

    ctx = load_plugin_context(
        model_capabilities_path=profile[
            "model_capabilities_path"
        ],
        model_layout_path=profile["model_layout_path"],
        backend="vllm",
        runtime={"deployment": deployment},
        require_verified=True,
    )

    require(
        ctx.model["model_id"] == profile["expected_model_id"],
        "Model ID mismatch",
    )
    require(
        ctx.model["architecture"] == "Qwen3ForCausalLM"
        and ctx.model.get("model_type") == "qwen3"
        and ctx.model.get("quantization_config") is None
        and ctx.model.get("checkpoint_dtype") == "bfloat16"
        and ctx.model["num_layers"] >= 2,
        "Unsupported Qwen3 checkpoint configuration",
    )

    registry = PluginRegistry()
    register_vllm_backend(registry)
    expected = registry.build(
        "backend", "vllm", ctx
    ).plan()

    require(
        plan == expected,
        "Execution plan does not match validated profile",
    )

    baseline = read(ROOT / "execution_plan.json")
    image = baseline["image"]
    master = (
        f"{args.name}-rank0."
        f"{args.namespace}.svc.cluster.local"
    )

    outputs = {}

    for rank, pvc in (
        (0, args.pvc_rank0),
        (1, args.pvc_rank1),
    ):
        key = f"vllm-rank{rank}"
        doc = read(TEMPLATES / f"{key}.json")
        doc = copy.deepcopy(doc)

        pod = doc["spec"]["template"]
        container = pod["spec"]["containers"][0]

        require(
            doc["spec"]["replicas"] == 0,
            "Baseline deployment must be paused",
        )
        require(
            container["image"] == image,
            "Baseline image mismatch",
        )

        baseline_command = baseline["nodes"][rank]["command"]
        original_command = container["command"]

        require(
            original_command[:len(baseline_command)]
            == baseline_command,
            "Baseline command differs from validated plan",
        )

        policy_suffix = original_command[len(baseline_command):]
        require(
            "--gpu-memory-utilization" in policy_suffix
            and "--max-num-seqs" in policy_suffix,
            "Missing validated vLLM runtime policy",
        )

        command = list(plan["nodes"][rank]["command"])
        command_flag(command, "--master-addr", master)
        command.extend(policy_suffix)

        container["command"] = command

        doc["metadata"]["name"] = f"{args.name}-rank{rank}"
        doc["metadata"]["namespace"] = args.namespace

        labels = {"app": args.name, "rank": str(rank)}
        doc["spec"]["selector"]["matchLabels"] = labels
        pod["metadata"]["labels"] = labels

        model_volumes = [
            volume for volume in pod["spec"]["volumes"]
            if volume["name"] == "model"
        ]
        require(len(model_volumes) == 1, "Missing model volume")
        model_volumes[0]["persistentVolumeClaim"][
            "claimName"
        ] = pvc

        outputs[f"{key}.json"] = doc

    for filename, service_name in (
        ("vllm-master-service.json", f"{args.name}-rank0"),
        ("vllm-api-service.json", f"{args.name}-api"),
    ):
        doc = read(TEMPLATES / filename)
        doc["metadata"]["name"] = service_name
        doc["metadata"]["namespace"] = args.namespace
        doc["spec"]["selector"]["app"] = args.name
        outputs[filename] = doc

    capabilities = Path(
        profile["model_capabilities_path"]
    ).read_text()
    layout = Path(profile["model_layout_path"]).read_text()

    outputs["model-metadata.json"] = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{args.name}-metadata",
            "namespace": args.namespace,
        },
        "data": {
            "model_capabilities.json": capabilities,
            "model_layout.json": layout,
        },
    }

    # Resolve absolute, relative and symlinked paths before
    # checking whether the destination is protected.
    destination = args.output_dir.resolve()
    protected = TEMPLATES.resolve()

    require(
        destination != protected
        and protected not in destination.parents,
        "Cannot overwrite the default deployment manifests",
    )
    require(
        not args.output_dir.exists()
        or not any(args.output_dir.iterdir()),
        "Output directory must be empty",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, document in sorted(outputs.items()):
        (args.output_dir / filename).write_text(
            json.dumps(document, indent=2) + "\n"
        )

    print("Model:", ctx.model["model_id"])
    print("Layers:", ctx.model["num_layers"])
    print("Namespace:", args.namespace)
    print("Private API service:", f"{args.name}-api")
    print("Manifests:", len(outputs))
    print("GPU worker replicas: 0")
    print("PVCs must be provisioned separately")
    print("CPU coordinator/recovery not included")
    print("No Kubernetes changes performed")
    print("V35_MODEL_SWAP_RENDER_PASS")


if __name__ == "__main__":
    main()
