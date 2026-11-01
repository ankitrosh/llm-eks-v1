#!/usr/bin/env python3
"""Export sanitized, reusable V1 manifests from the live EKS cluster.

Read-only: does not modify Kubernetes resources.
"""

import copy
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K8S = ROOT / "kubernetes"
INFRA = ROOT / "infrastructure"
NS = "llm"

K8S.mkdir(exist_ok=True)
INFRA.mkdir(exist_ok=True)


def fetch(kind, name, namespace=NS):
    args = ["kubectl", "get", kind, name]

    if namespace:
        args += ["-n", namespace]

    args += ["-o", "json"]

    print("Reading:", kind, name, flush=True)

    return json.loads(subprocess.check_output(args, text=True))


def clean(obj):
    """Keep desired configuration, remove server-generated metadata."""
    result = {
        "apiVersion": obj["apiVersion"],
        "kind": obj["kind"],
        "metadata": {
            "name": obj["metadata"]["name"]
        },
    }

    if obj["metadata"].get("namespace"):
        result["metadata"]["namespace"] = obj["metadata"]["namespace"]

    kind = obj["kind"]

    if kind in ("StatefulSet", "Deployment"):
        result["spec"] = copy.deepcopy(obj["spec"])

        annotations = (
            result["spec"]
            .get("template", {})
            .get("metadata", {})
            .get("annotations", {})
        )

        annotations.pop(
            "kubectl.kubernetes.io/restartedAt", None
        )

        # Keep volume claim templates declarative.
        for claim in result["spec"].get("volumeClaimTemplates", []):
            claim["metadata"] = {
                "name": claim["metadata"]["name"]
            }

    elif kind == "Service":
        spec = copy.deepcopy(obj["spec"])

        # Kubernetes generates these for normal Services.
        for key in (
            "clusterIPs",
            "ipFamilies",
            "ipFamilyPolicy",
            "internalTrafficPolicy",
        ):
            spec.pop(key, None)

        if spec.get("clusterIP") != "None":
            spec.pop("clusterIP", None)

        result["spec"] = spec

    elif kind == "ConfigMap":
        result["data"] = copy.deepcopy(obj.get("data", {}))

    elif kind == "Role":
        result["rules"] = copy.deepcopy(obj["rules"])

    elif kind == "RoleBinding":
        result["subjects"] = copy.deepcopy(obj["subjects"])
        result["roleRef"] = copy.deepcopy(obj["roleRef"])

    elif kind == "ServiceAccount":
        for key in ("automountServiceAccountToken", "imagePullSecrets"):
            if key in obj:
                result[key] = copy.deepcopy(obj[key])

    elif kind == "StorageClass":
        for key in (
            "provisioner",
            "parameters",
            "reclaimPolicy",
            "volumeBindingMode",
            "allowVolumeExpansion",
            "mountOptions",
            "allowedTopologies",
        ):
            if key in obj:
                result[key] = copy.deepcopy(obj[key])

    else:
        raise ValueError(f"Unsupported kind: {kind}")

    return result


def render(obj):
    # Prefer readable YAML if PyYAML is already installed.
    try:
        import yaml
        return yaml.safe_dump(obj, sort_keys=False)
    except ImportError:
        # JSON is also valid Kubernetes manifest syntax.
        return json.dumps(obj, indent=2) + "\n"


def save(path, objects):
    if not isinstance(objects, list):
        objects = [objects]

    content = "\n---\n".join(render(obj).rstrip() for obj in objects)

    path.write_text(content + "\n")
    print("Created:", path.relative_to(ROOT))


def resource(kind, name, namespace=NS):
    return clean(fetch(kind, name, namespace))


# Namespace
save(
    K8S / "namespace.yaml",
    {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": NS},
    },
)

# Inference workers
save(
    K8S / "qwen3-statefulset.yaml",
    resource("statefulset", "qwen3"),
)

# GPU networking
save(
    K8S / "qwen3-services.yaml",
    [
        resource("service", "qwen3-headless"),
        resource("service", "qwen3-api"),
    ],
)

# CPU gateway
save(
    K8S / "coordinator.yaml",
    [
        resource("deployment", "qwen3-coordinator"),
        resource("service", "qwen3-coordinator"),
    ],
)

# CPU recovery controller
save(
    K8S / "recovery.yaml",
    [
        resource("deployment", "qwen3-recovery"),
        resource("service", "qwen3-recovery"),
    ],
)

# Exact RBAC configuration
save(
    K8S / "recovery-rbac.yaml",
    [
        resource("serviceaccount", "qwen3-recovery"),
        resource("role", "qwen3-recovery"),
        resource("rolebinding", "qwen3-recovery"),
        resource("role", "qwen3-recovery-state-writer"),
        resource("rolebinding", "qwen3-recovery-state-writer"),
    ],
)

# Fresh recovery state.
# NEVER deploy stale pod identities or runtime recovery status.
fresh_state = {
    "mode": "RUNNING",
    "phase": "STARTING",
    "generation": 0,
    "recoveries": 0,
    "attempt": 0,
    "baseline": None,
    "started_at": None,
    "last_error": None,
    "last_recovery_seconds": None,
}

save(
    K8S / "recovery-state.yaml",
    {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "qwen3-recovery-state",
            "namespace": NS,
        },
        "data": {
            "state": json.dumps(fresh_state, indent=2)
        },
    },
)

# EBS StorageClass
save(
    INFRA / "storageclass.yaml",
    resource("storageclass", "llm-gp3", None),
)

print("\nAll reusable Kubernetes manifests exported.")
print("No live cluster resources were changed.")
