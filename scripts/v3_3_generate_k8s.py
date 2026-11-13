"""Generate V3.3 Kubernetes manifests. No cluster mutations."""

import json
from pathlib import Path

from llm_runtime.extensions import (
    PluginRegistry,
    load_plugin_context,
)
from llm_runtime.backends.vllm import register_vllm_backend


ROOT = Path("infrastructure/profiles/v3-3/qwen3-14b")
OUT = Path("deploy/v3_3/rendered")
OUT.mkdir(parents=True, exist_ok=True)

NS = "llm"
IMAGE = "vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
MODEL_PATH = "/models/qwen3-14b"

MASTER = "v33-vllm-rank0.llm.svc.cluster.local"

policy = {
    "schema_version": 1,
    "backend": "vllm",
    "max_num_seqs": 4,
    "max_num_batched_tokens": 512,
    "gpu_memory_utilization": 0.85,
    "enforce_eager_for_smoke": True,
}

runtime = {
    "deployment": {
        "model_path": MODEL_PATH,
        "nnodes": 2,
        "gpus_per_node": 1,
        "pp": 2,
        "tp": 1,
        "master_addr": MASTER,
        "master_port": 29501,
        "max_model_len": 2048,
        "dtype": "bfloat16",
        "port": 8000,
    }
}

ctx = load_plugin_context(
    model_capabilities_path=ROOT / "model_capabilities.json",
    model_layout_path=ROOT / "model_layout.json",
    backend="vllm",
    runtime=runtime,
)

registry = PluginRegistry()
register_vllm_backend(registry)
backend = registry.build("backend", "vllm", ctx)

plan = backend.plan()
plan["scheduler_policy"] = policy
plan["image"] = IMAGE

assert ctx.model["num_layers"] == 40
assert plan["pipeline_parallel_size"] == 2
assert plan["tensor_parallel_size"] == 1
assert len(plan["nodes"]) == 2

for name, document in (
    ("execution_plan.json", plan),
    ("scheduler_policy.json", policy),
):
    (ROOT / name).write_text(
        json.dumps(document, indent=2) + "\n"
    )


def object_meta(name):
    return {"name": name, "namespace": NS}


def save(name, document):
    (OUT / (name + ".json")).write_text(
        json.dumps(document, indent=2) + "\n"
    )


def service(name, selector, ports, *, headless=False):
    spec = {
        "selector": selector,
        "ports": [
            {"name": key, "port": value, "targetPort": value}
            for key, value in ports.items()
        ],
    }

    if headless:
        spec["clusterIP"] = "None"
        spec["publishNotReadyAddresses"] = True

    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": object_meta(name),
        "spec": spec,
    }


def deployment(name, labels, container, *, gpu=False):
    spec = {
        "nodeSelector": (
            {"pool": "gpu"}
            if gpu else
            {"eks.amazonaws.com/nodegroup": "cpu-system"}
        ),
        "containers": [container],
    }

    if gpu:
        rank = int(labels["rank"])

        spec["terminationGracePeriodSeconds"] = 30
        spec["volumes"] = [
            {
                "name": "model",
                "persistentVolumeClaim": {
                    "claimName": f"model-files-qwen3-{rank}"
                },
            },
            {
                "name": "dshm",
                "emptyDir": {
                    "medium": "Memory",
                    "sizeLimit": "1Gi",
                },
            },
        ]
    else:
        spec["volumes"] = [
            {
                "name": "source",
                "configMap": {
                    "name": "v33-cpu-source"
                },
            },
            {
                "name": "model-metadata",
                "configMap": {
                    "name": "v33-model-metadata"
                },
            },
        ]

    if name == "v33-recovery":
        spec["serviceAccountName"] = "v33-recovery"

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": object_meta(name),
        "spec": {
            "replicas": 0,
            "strategy": {
                "type": "Recreate"
            },
            "selector": {
                "matchLabels": labels
            },
            "template": {
                "metadata": {"labels": labels},
                "spec": spec,
            },
        },
    }


def env(name, value):
    return {"name": name, "value": str(value)}


# GPU workers ---------------------------------------------------

for rank in (0, 1):
    labels = {"app": "v33-vllm", "rank": str(rank)}

    command = list(plan["nodes"][rank]["command"])

    command.extend([
        "--gpu-memory-utilization",
        str(policy["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(policy["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(policy["max_num_batched_tokens"]),
        "--distributed-timeout-seconds",
        "900",
    ])

    if policy["enforce_eager_for_smoke"]:
        command.append("--enforce-eager")

    container = {
        "name": "vllm",
        "image": IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "env": [
            env("NCCL_IB_DISABLE", "1"),
            env("NCCL_SOCKET_IFNAME", "eth0"),
            env("GLOO_SOCKET_IFNAME", "eth0"),
            {
                "name": "VLLM_HOST_IP",
                "valueFrom": {
                    "fieldRef": {
                        "fieldPath": "status.podIP"
                    }
                },
            },
        ],
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "8Gi",
                "nvidia.com/gpu": "1",
            },
            "limits": {
                "memory": "14Gi",
                "nvidia.com/gpu": "1",
            },
        },
        "volumeMounts": [
            {
                "name": "model",
                "mountPath": "/models",
                "readOnly": True,
            },
            {
                "name": "dshm",
                "mountPath": "/dev/shm",
            },
        ],
    }

    if rank == 0:
        container["ports"] = [
            {"name": "api", "containerPort": 8000},
            {"name": "master", "containerPort": 29501},
        ]

        container["startupProbe"] = {
            "httpGet": {
                "path": "/health",
                "port": 8000,
            },
            "periodSeconds": 10,
            "failureThreshold": 120,
        }

        container["readinessProbe"] = {
            "httpGet": {
                "path": "/health",
                "port": 8000,
            },
            "periodSeconds": 5,
            "failureThreshold": 3,
        }

    save(
        f"vllm-rank{rank}",
        deployment(
            f"v33-vllm-rank{rank}",
            labels,
            container,
            gpu=True,
        ),
    )

save(
    "vllm-master-service",
    service(
        "v33-vllm-rank0",
        {"app": "v33-vllm", "rank": "0"},
        {"master": 29501},
        headless=True,
    ),
)

save(
    "vllm-api-service",
    service(
        "v33-vllm-api",
        {"app": "v33-vllm", "rank": "0"},
        {"api": 8000},
    ),
)


# CPU source and metadata ----------------------------------------

source_files = {
    "vllm_coordinator.py":
        "src/llm_runtime/coordination/vllm_coordinator.py",
    "v33_recovery_controller.py":
        "src/llm_runtime/coordination/v33_recovery_controller.py",
    "vllm_engine_controller.py":
        "src/llm_runtime/coordination/vllm_engine_controller.py",
}

save(
    "cpu-source",
    {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": object_meta("v33-cpu-source"),
        "data": {
            name: Path(path).read_text()
            for name, path in source_files.items()
        },
    },
)

metadata_files = (
    "model_layout.json",
    "model_capabilities.json",
    "execution_plan.json",
    "scheduler_policy.json",
)

save(
    "model-metadata",
    {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": object_meta("v33-model-metadata"),
        "data": {
            name: (ROOT / name).read_text()
            for name in metadata_files
        },
    },
)


def cpu_mounts():
    return [
        {
            "name": "source",
            "mountPath": "/opt/v33",
            "readOnly": True,
        },
        {
            "name": "model-metadata",
            "mountPath": "/opt/modelmeta",
            "readOnly": True,
        },
    ]


# CPU coordinator ------------------------------------------------

coord_container = {
    "name": "coordinator",
    "image": "python:3.12-slim",
    "command": [
        "python3",
        "-u",
        "/opt/v33/vllm_coordinator.py",
    ],
    "env": [
        env("PORT", 8080),
        env(
            "VLLM_UPSTREAM_URL",
            "http://v33-vllm-api.llm.svc.cluster.local:8000",
        ),
        env(
            "RECOVERY_STATUS_URL",
            "http://v33-recovery.llm.svc.cluster.local:8081",
        ),
        env("REQUIRE_RECOVERY", "true"),
    ],
    "resources": {
        "requests": {
            "cpu": "100m",
            "memory": "128Mi",
        },
        "limits": {
            "memory": "512Mi",
        },
    },
    "volumeMounts": cpu_mounts(),
    "ports": [
        {"containerPort": 8080}
    ],
    "livenessProbe": {
        "httpGet": {
            "path": "/healthz",
            "port": 8080,
        },
    },
    "readinessProbe": {
        "httpGet": {
            "path": "/readyz",
            "port": 8080,
        },
    },
}

save(
    "cpu-coordinator",
    deployment(
        "v33-coordinator",
        {"app": "v33-coordinator"},
        coord_container,
    ),
)

save(
    "cpu-coordinator-service",
    service(
        "v33-coordinator",
        {"app": "v33-coordinator"},
        {"http": 8080},
    ),
)


# CPU recovery ---------------------------------------------------

recovery_container = {
    "name": "recovery",
    "image": "python:3.12-slim",
    "command": [
        "python3",
        "-u",
        "/opt/v33/v33_recovery_controller.py",
    ],
    "env": [
        env("AUTO_RECOVER", "false"),
        env(
            "V33_VLLM_HEALTH_URL",
            "http://v33-vllm-api.llm.svc.cluster.local:8000/health",
        ),
    ],
    "resources": {
        "requests": {
            "cpu": "100m",
            "memory": "128Mi",
        },
        "limits": {
            "memory": "512Mi",
        },
    },
    "volumeMounts": cpu_mounts(),
    "ports": [
        {"containerPort": 8081}
    ],
    "readinessProbe": {
        "tcpSocket": {"port": 8081},
        "periodSeconds": 5,
    },
}

save(
    "cpu-recovery",
    deployment(
        "v33-recovery",
        {"app": "v33-recovery"},
        recovery_container,
    ),
)

save(
    "cpu-recovery-service",
    service(
        "v33-recovery",
        {"app": "v33-recovery"},
        {"http": 8081},
    ),
)


# Recovery state and namespace-scoped RBAC -----------------------

state = json.loads(
    Path(
        "deploy/v3_3/v33-recovery-state.json"
    ).read_text()
)

assert state["metadata"]["name"] == "v33-recovery-state"
assert json.loads(state["data"]["state"])["mode"] == "PAUSED"

save("recovery-state", state)

save(
    "recovery-service-account",
    {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": object_meta("v33-recovery"),
    },
)

save(
    "recovery-role",
    {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": object_meta("v33-recovery"),
        "rules": [
            {
                "apiGroups": [""],
                "resources": ["configmaps"],
                "resourceNames": ["v33-recovery-state"],
                "verbs": ["get", "patch"],
            },
            {
                "apiGroups": [""],
                "resources": ["pods"],
                "verbs": ["list"],
            },
            {
                "apiGroups": ["apps"],
                "resources": ["deployments/scale"],
                "resourceNames": [
                    "v33-vllm-rank0",
                    "v33-vllm-rank1",
                ],
                "verbs": ["get", "patch"],
            },
        ],
    },
)

save(
    "recovery-role-binding",
    {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": object_meta("v33-recovery"),
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": "v33-recovery",
                "namespace": NS,
            },
        ],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": "v33-recovery",
        },
    },
)

print("V33_K8S_GENERATION_OK")
print("Model:", ctx.model["model_id"])
print("Image:", IMAGE)
print("Pipeline stages: 2")
print("GPU Deployments: 2 (replicas=0)")
print("CPU Deployments: 2 (replicas=0)")
print("Automatic recovery: disabled")
print("Generated manifests:", len(list(OUT.glob("*.json"))))
print("No cluster changes performed")
