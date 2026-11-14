"""Read-only V3.5 deployment readiness checks."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def assess(config, kube=None):
    """Return (level, code, explanation) tuples."""
    findings = []

    def record(level, code, message):
        findings.append((level, code, message))

    if not isinstance(config, dict) or config.get("schema_version") != 1:
        return [("FAIL", "schema", "Expected schema_version: 1")]

    aws = config.get("aws")
    deploy = config.get("deployment")
    runtime = config.get("runtime")
    model = config.get("model")
    discovery = config.get("aws_discovery")

    if not all(
        isinstance(x, dict)
        for x in (aws, deploy, runtime, model, discovery)
    ):
        return [(
            "FAIL", "schema",
            "Required configuration sections are missing",
        )]

    if all(
        isinstance(aws.get(k), str)
        and aws[k] not in ("", "auto")
        for k in (
            "account_id", "region",
            "cluster_name", "namespace",
        )
    ):
        record(
            "PASS", "aws_target",
            "AWS target selected (not independently refreshed)",
        )
    else:
        record(
            "FAIL", "aws_target",
            "AWS identity or namespace incomplete",
        )

    topology = (
        runtime.get("backend"),
        runtime.get("pipeline_parallelism"),
        runtime.get("tensor_parallelism"),
    )

    if topology == ("vllm", 2, 1):
        record(
            "PASS", "topology",
            "Validated vLLM PP=2/TP=1 reference topology",
        )
    else:
        record(
            "FAIL", "topology",
            "Topology differs from validated reference",
        )

    if deploy.get("replicas") == 0:
        record(
            "PASS", "safe_replicas",
            "Deployment remains scaled to zero",
        )
    else:
        record(
            "WARN", "safe_replicas",
            "Nonzero replicas configured; no changes made",
        )

    group = deploy.get("gpu_nodegroup")
    selector = deploy.get("gpu_node_selector")

    if (
        isinstance(group, str)
        and bool(group)
        and selector == {
            "eks.amazonaws.com/nodegroup": group
        }
    ):
        record(
            "PASS", "gpu_group",
            "Managed GPU group and selector configured",
        )
    else:
        record(
            "FAIL", "gpu_group",
            "Verified GPU group and selector required",
        )

    required = discovery.get(
        "required_vcpus_for_two_nodes"
    )
    limit = discovery.get("quota_value")

    if (
        not isinstance(required, (int, float))
        or isinstance(required, bool)
        or required <= 0
    ):
        record(
            "FAIL", "quota",
            "Invalid required-vCPU count",
        )
    elif (
        isinstance(limit, (int, float))
        and not isinstance(limit, bool)
    ):
        if limit < required:
            record(
                "FAIL", "quota",
                "Regional quota below required vCPUs",
            )
        else:
            record(
                "WARN", "quota",
                "Quota total sufficient; usage and "
                "reservations unverified",
            )
    else:
        record(
            "WARN", "quota",
            "Applied quota could not be verified",
        )

    maximum = discovery.get("nodegroup_max_size")
    desired = discovery.get("nodegroup_desired_size")

    if (
        isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and maximum >= 2
    ):
        record(
            "PASS", "nodegroup_max",
            "Managed group permits at least two nodes",
        )
    else:
        record(
            "FAIL", "nodegroup_max",
            "Node-group maximum must permit two nodes",
        )

    if desired == 0:
        record(
            "WARN", "nodegroup_desired",
            "GPU node group is scaled to zero",
        )
    elif (
        isinstance(desired, int)
        and not isinstance(desired, bool)
        and desired >= 2
    ):
        record(
            "PASS", "nodegroup_desired",
            "Two nodes requested; readiness unverified",
        )
    else:
        record(
            "WARN", "nodegroup_desired",
            "Fewer than two GPU nodes requested or unknown",
        )

    pvc_names = [
        model.get("pvc_rank0"),
        model.get("pvc_rank1"),
    ]

    pvc_valid = (
        all(
            isinstance(name, str) and bool(name)
            for name in pvc_names
        )
        and pvc_names[0] != pvc_names[1]
    )

    if pvc_valid:
        record(
            "PASS", "pvc_config",
            "Two distinct model PVC names configured",
        )
    else:
        record(
            "FAIL", "pvc_config",
            "Two distinct model PVC names required",
        )

    if kube is None:
        record(
            "WARN", "kubernetes",
            "Live Kubernetes checks not requested",
        )
        return findings

    if any(
        level == "FAIL" and code == "aws_target"
        for level, code, _ in findings
    ):
        record(
            "FAIL", "kubernetes",
            "Cannot inspect Kubernetes without valid target",
        )
        return findings

    namespace = aws["namespace"]

    try:
        kube(["get", "namespace", namespace])
    except (OSError, RuntimeError, ValueError):
        record(
            "FAIL", "kubernetes",
            "Selected Kubernetes namespace inaccessible",
        )
        return findings

    record(
        "PASS", "kubernetes",
        "Selected Kubernetes namespace accessible",
    )

    try:
        response = kube(["get", "nodes"])
        nodes = response.get("items")

        if not isinstance(nodes, list):
            raise ValueError("Invalid nodes response")

        ready_gpus = 0
        ready_gpu_nodes = 0

        for node in nodes:
            if not isinstance(node, dict):
                continue

            metadata = node.get("metadata") or {}
            spec = node.get("spec") or {}
            status = node.get("status") or {}
            labels = metadata.get("labels") or {}

            if (
                not isinstance(selector, dict)
                or not all(
                    labels.get(key) == value
                    for key, value in selector.items()
                )
                or spec.get("unschedulable", False)
            ):
                continue

            ready = any(
                item.get("type") == "Ready"
                and item.get("status") == "True"
                for item in status.get("conditions", [])
            )

            if ready:
                try:
                    gpu_count = int(
                        (status.get("allocatable") or {})
                        .get("nvidia.com/gpu", "0")
                    )
                    if gpu_count > 0:
                        ready_gpu_nodes += 1
                        ready_gpus += gpu_count
                except (TypeError, ValueError):
                    pass

        if ready_gpu_nodes >= 2 and ready_gpus >= 2:
            record(
                "PASS", "gpu_nodes",
                "At least two Ready GPU nodes match selector",
            )
        else:
            record(
                "WARN", "gpu_nodes",
                "Fewer than two matching Ready GPU nodes",
            )

    except (
        OSError, RuntimeError, ValueError, AttributeError
    ):
        record(
            "WARN", "gpu_nodes",
            "Kubernetes GPU readiness unverified",
        )

    if pvc_valid:
        try:
            response = kube([
                "get", "pvc", "-n", namespace
            ])
            items = response.get("items")

            if not isinstance(items, list):
                raise ValueError("Invalid PVC response")

            phases = {
                (item.get("metadata") or {}).get("name"):
                (item.get("status") or {}).get("phase")
                for item in items
                if isinstance(item, dict)
            }

            if all(
                phases.get(name) == "Bound"
                for name in pvc_names
            ):
                record(
                    "PASS", "model_pvcs",
                    "Both model PVCs Bound; contents unverified",
                )
            else:
                record(
                    "FAIL", "model_pvcs",
                    "Model PVCs missing or not Bound",
                )

        except (OSError, RuntimeError, ValueError):
            record(
                "FAIL", "model_pvcs",
                "Unable to verify model PVCs",
            )

    return findings


def kubectl_client(context):
    def call(args):
        try:
            result = subprocess.run(
                [
                    "kubectl", "--context", context,
                    *args, "-o", "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (
            OSError, subprocess.TimeoutExpired
        ) as exc:
            raise RuntimeError(
                "Kubernetes query failed or timed out"
            ) from exc

        if result.returncode:
            raise RuntimeError(
                "Kubernetes API query failed"
            )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid Kubernetes response"
            ) from exc

    return call


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--check-k8s",
        action="store_true",
        help="Run read-only checks against EKS context",
    )

    args = parser.parse_args()

    try:
        config = yaml.safe_load(
            args.config.read_text()
        )
    except (OSError, yaml.YAMLError) as exc:
        parser.error(
            f"Cannot read configuration: {type(exc).__name__}"
        )

    client = None

    if args.check_k8s:
        aws = (
            config.get("aws", {})
            if isinstance(config, dict)
            else {}
        )

        context = (
            f"arn:aws:eks:{aws.get('region')}:"
            f"{aws.get('account_id')}:"
            f"cluster/{aws.get('cluster_name')}"
        )
        client = kubectl_client(context)

    findings = assess(config, client)

    for level, code, message in findings:
        print(f"[{level}] {code}: {message}")

    failures = sum(
        level == "FAIL"
        for level, _, _ in findings
    )
    warnings = sum(
        level == "WARN"
        for level, _, _ in findings
    )

    print(
        f"Summary: {failures} FAIL, {warnings} WARN"
    )
    print("No AWS or Kubernetes resources modified")
    print("V35_DOCTOR_AUDIT_COMPLETE")

    raise SystemExit(2 if failures else 0)


if __name__ == "__main__":
    main()
