"""Render isolated V3.5 reference manifests without cluster access."""

import argparse
import copy
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy/v3_3/rendered"
EXPECTED = {
    "cpu-coordinator-service.json",
    "cpu-coordinator.json",
    "cpu-recovery-service.json",
    "cpu-recovery.json",
    "cpu-source.json",
    "model-metadata.json",
    "recovery-role-binding.json",
    "recovery-role.json",
    "recovery-service-account.json",
    "recovery-state.json",
    "vllm-api-service.json",
    "vllm-master-service.json",
    "vllm-rank0.json",
    "vllm-rank1.json",
}
DNS_LABEL = re.compile(
    r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_cpu_nodegroup(config, override=None):
    """Use verified discovery, with explicit legacy fallback."""
    deployment = (
        config.get("deployment")
        if isinstance(config, dict)
        else None
    )

    require(
        isinstance(deployment, dict),
        "Missing deployment configuration",
    )

    discovered = deployment.get("cpu_nodegroup")
    selector = deployment.get("cpu_node_selector")

    if discovered is not None:
        require(
            isinstance(discovered, str) and bool(discovered),
            "Invalid discovered CPU node group",
        )

        require(
            selector == {
                "eks.amazonaws.com/nodegroup": discovered
            },
            "Discovered CPU selector mismatch",
        )

        require(
            override is None or override == discovered,
            "CPU node-group override conflicts with discovery",
        )

        return discovered

    require(
        selector is None,
        "CPU selector configured without node-group identity",
    )

    require(
        override is not None,
        "CPU node group not discovered; "
        "provide --cpu-nodegroup",
    )

    return override


def validate(config, cpu_nodegroup):
    require(
        isinstance(config, dict)
        and config.get("schema_version") == 1,
        "Unsupported configuration schema",
    )

    for section in (
        "aws", "model", "runtime",
        "deployment", "aws_discovery",
        "storage_discovery",
    ):
        require(
            isinstance(config.get(section), dict),
            f"Missing {section} section",
        )

    aws = config["aws"]
    model = config["model"]
    runtime = config["runtime"]
    deploy = config["deployment"]
    discovery = config["aws_discovery"]
    storage = config["storage_discovery"]

    namespace = aws.get("namespace")
    require(
        isinstance(namespace, str)
        and len(namespace) <= 63
        and DNS_LABEL.fullmatch(namespace),
        "Invalid Kubernetes namespace",
    )

    require(
        isinstance(aws.get("account_id"), str)
        and re.fullmatch(
            r"\d{12}", aws["account_id"]
        ),
        "AWS account identity not configured",
    )

    for field in ("region", "cluster_name"):
        require(
            isinstance(aws.get(field), str)
            and bool(aws[field])
            and aws[field] != "auto",
            f"AWS {field} not configured",
        )

    require(
        isinstance(cpu_nodegroup, str)
        and len(cpu_nodegroup) <= 63
        and DNS_LABEL.fullmatch(cpu_nodegroup),
        "Valid CPU node group required",
    )

    require(
        (
            model.get("id"),
            model.get("checkpoint_path"),
        ) == (
            "Qwen/Qwen3-14B",
            "/models/qwen3-14b",
        ),
        "This renderer supports only the validated "
        "Qwen3-14B reference model",
    )

    require(
        (
            runtime.get("backend"),
            runtime.get("pipeline_parallelism"),
            runtime.get("tensor_parallelism"),
            runtime.get("dtype"),
        ) == ("vllm", 2, 1, "bfloat16"),
        "Unsupported backend or pipeline topology",
    )

    require(
        runtime.get("gpu_memory_utilization") == 0.85,
        "Reference GPU memory utilization must be 0.85",
    )

    require(
        deploy.get("replicas") == 0
        and deploy.get("auto_recovery") is False,
        "Unsafe deployment or recovery configuration",
    )

    require(
        deploy.get("capacity_type") == "ON_DEMAND",
        "Only validated On-Demand reference supported",
    )

    require(
        deploy.get("gpu_instance_type") == "g5.xlarge",
        "Only validated g5.xlarge reference supported",
    )

    group = deploy.get("gpu_nodegroup")
    selector = deploy.get("gpu_node_selector")

    require(
        isinstance(group, str)
        and bool(group)
        and selector == {
            "eks.amazonaws.com/nodegroup": group
        },
        "Verified GPU node-group selector required",
    )

    require(
        isinstance(discovery.get("nodegroup_max_size"), int)
        and not isinstance(
            discovery["nodegroup_max_size"], bool
        )
        and discovery["nodegroup_max_size"] >= 2,
        "GPU node-group maximum below two",
    )

    required = discovery.get(
        "required_vcpus_for_two_nodes"
    )
    quota = discovery.get("quota_value")

    require(
        required == 8
        and not isinstance(required, bool)
        and isinstance(quota, (int, float))
        and not isinstance(quota, bool)
        and quota >= required,
        "GPU vCPU quota cannot support reference topology",
    )

    pvcs = (
        model.get("pvc_rank0"),
        model.get("pvc_rank1"),
    )

    require(
        all(
            isinstance(name, str)
            and bool(name)
            and len(name) <= 253
            and all(
                len(part) <= 63
                and DNS_LABEL.fullmatch(part)
                for part in name.split(".")
            )
            for name in pvcs
        )
        and pvcs[0] != pvcs[1],
        "Two distinct valid model PVC names required",
    )

    require(
        storage.get("pvc_rank0_bound") is True
        and storage.get("pvc_rank1_bound") is True,
        "Bound PVC discovery evidence missing",
    )

    return namespace


def transform_strings(value, namespace):
    """Rewrite only the known V3.3 resource identifiers."""
    if isinstance(value, dict):
        return {
            key: transform_strings(item, namespace)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            transform_strings(item, namespace)
            for item in value
        ]

    if isinstance(value, str):
        value = value.replace(
            "qwen3-recovery.llm.svc.cluster.local",
            f"v35-recovery.{namespace}.svc.cluster.local",
        )
        value = value.replace("v33-", "v35-")
        value = value.replace(
            ".llm.svc.cluster.local",
            f".{namespace}.svc.cluster.local",
        )
        return value

    return value


def build(config, cpu_nodegroup=None):
    """Produce 14 new documents entirely in memory."""
    cpu_nodegroup = resolve_cpu_nodegroup(
        config, cpu_nodegroup
    )
    namespace = validate(config, cpu_nodegroup)

    paths = list(SOURCE.glob("*.json"))

    require(
        {path.name for path in paths} == EXPECTED,
        "Frozen V3.3 manifest inventory changed",
    )

    docs = {}

    for path in paths:
        original = json.loads(path.read_text())
        doc = transform_strings(
            copy.deepcopy(original), namespace
        )

        require(
            doc["metadata"].get("namespace") == "llm",
            "Unexpected source namespace",
        )

        doc["metadata"]["namespace"] = namespace

        if doc["kind"] == "RoleBinding":
            for subject in doc.get("subjects", []):
                if subject.get("namespace") == "llm":
                    subject["namespace"] = namespace

        docs[path.name] = doc

    deploy = config["deployment"]
    model = config["model"]

    for rank in (0, 1):
        doc = docs[f"vllm-rank{rank}.json"]
        spec = doc["spec"]["template"]["spec"]

        spec["nodeSelector"] = copy.deepcopy(
            deploy["gpu_node_selector"]
        )

        volumes = [
            volume
            for volume in spec["volumes"]
            if volume.get("name") == "model"
        ]

        require(
            len(volumes) == 1,
            "Expected exactly one model volume",
        )

        volumes[0]["persistentVolumeClaim"]["claimName"] = (
            model[f"pvc_rank{rank}"]
        )

    for filename in (
        "cpu-coordinator.json",
        "cpu-recovery.json",
    ):
        spec = docs[filename]["spec"]["template"]["spec"]
        spec["nodeSelector"] = {
            "eks.amazonaws.com/nodegroup": cpu_nodegroup
        }

    source = docs["cpu-source.json"]["data"]

    recovery = source["v33_recovery_controller.py"]
    require(
        'NS = "llm"' in recovery,
        "Recovery namespace source contract changed",
    )
    source["v33_recovery_controller.py"] = recovery.replace(
        'NS = "llm"',
        f'NS = "{namespace}"',
    )

    engine = source["vllm_engine_controller.py"]
    require(
        'namespace="llm"' in engine,
        "Engine controller namespace contract changed",
    )
    source["vllm_engine_controller.py"] = engine.replace(
        'namespace="llm"',
        f'namespace="{namespace}"',
    )

    metadata = docs["model-metadata.json"]["data"]

    capabilities = json.loads(
        metadata["model_capabilities.json"]
    )
    require(
        capabilities.get("model_id") == "Qwen/Qwen3-14B",
        "Frozen model metadata mismatch",
    )

    plan = json.loads(metadata["execution_plan.json"])

    require(
        plan.get("pipeline_parallel_size") == 2
        and plan.get("tensor_parallel_size") == 1
        and len(plan.get("nodes", [])) == 2,
        "Frozen execution plan mismatch",
    )

    expected_master = (
        f"v35-vllm-rank0.{namespace}.svc.cluster.local"
    )

    for rank in (0, 1):
        command = docs[
            f"vllm-rank{rank}.json"
        ]["spec"]["template"]["spec"]["containers"][0][
            "command"
        ]

        require(
            "--master-addr" in command,
            "Missing worker master address",
        )

        address_index = command.index("--master-addr") + 1

        require(
            command[address_index] == expected_master,
            "Incorrect worker master address",
        )

    for node in plan["nodes"]:
        command = node["command"]
        require(
            "--master-addr" in command,
            "Invalid embedded execution plan",
        )
        require(
            command[command.index("--master-addr") + 1]
            == expected_master,
            "Embedded execution plan master mismatch",
        )

    for doc in docs.values():
        if doc["kind"] == "Deployment":
            require(
                doc["spec"]["replicas"] == 0,
                "Nonzero workload replicas",
            )

    recovery_env = docs["cpu-recovery.json"][
        "spec"
    ]["template"]["spec"]["containers"][0]["env"]

    require(
        any(
            item.get("name") == "AUTO_RECOVER"
            and item.get("value") == "false"
            for item in recovery_env
        ),
        "Automatic recovery unexpectedly enabled",
    )

    state = json.loads(
        docs["recovery-state.json"]["data"]["state"]
    )

    require(
        state.get("mode") == "PAUSED",
        "Recovery state must remain PAUSED",
    )

    return docs


def validate_output_dir(path):
    destination = path.resolve()

    require(
        not path.is_symlink() and not path.exists(),
        "Output already exists; refusing overwrite",
    )

    for protected in (
        SOURCE,
        ROOT / "deploy/v3_3",
        ROOT / "deploy/v3_4_1",
    ):
        protected = protected.resolve()

        require(
            destination != protected
            and protected not in destination.parents,
            "Cannot write inside frozen deployment directories",
        )

    require(
        destination != ROOT.resolve(),
        "Cannot render over repository root",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config", required=True, type=Path
    )
    parser.add_argument(
        "--cpu-nodegroup"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path
    )
    args = parser.parse_args()

    try:
        config = yaml.safe_load(
            args.config.read_text()
        )

        validate_output_dir(args.output_dir)
        docs = build(config, args.cpu_nodegroup)

        args.output_dir.mkdir(
            parents=True,
            mode=0o700,
            exist_ok=False,
        )

        for filename, doc in sorted(docs.items()):
            (args.output_dir / filename).write_text(
                json.dumps(doc, indent=2) + "\n"
            )

    except (
        OSError, ValueError, KeyError,
        TypeError, json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        parser.error(str(exc))

    print("Reference model: Qwen3-14B")
    print("Backend: vLLM PP=2 / TP=1")
    print("Generated manifests:", len(docs))
    print("Resource prefix: v35-")
    print("Namespace:", config["aws"]["namespace"])
    print("GPU workloads: 2, replicas=0")
    print("CPU workloads: 2, replicas=0")
    print("Automatic recovery: disabled")
    print("Output:", args.output_dir)
    print("No Kubernetes resources modified")
    print("V35_CONFIG_RENDER_PASS")


if __name__ == "__main__":
    main()
