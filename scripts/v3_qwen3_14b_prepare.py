"""Prepare isolated two-pod Qwen3-14B V3 benchmark.

Writes local Kubernetes YAML only. Does not create AWS resources.
"""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "deploy/manifests/v3-qwen3-uneven-nccl.yaml"
OUT = ROOT / "deploy/manifests/v3-qwen3-14b-benchmark.yaml"
CM_OUT = Path("/tmp/v3-qwen3-14b-benchmark-code-configmap.yaml")

NAME = "v3-qwen3-14b-benchmark"
CM_NAME = NAME + "-code"
SERVICE = NAME + "-headless"

RENDEZVOUS = "29658"
CONTROL = "29659"

source = list(yaml.safe_load_all(BASE.read_text()))

service = deepcopy(
    next(
        x for x in source
        if x and x.get("kind") == "Service"
    )
)

base_job = next(
    x for x in source
    if x and x.get("kind") == "Job"
)

files = {
    "partition": "src/llm_runtime/providers/v3_qwen3_partition.py",
    "stage_only_provider": "src/llm_runtime/providers/v3_qwen3_stage_only_factory.py",
    "provider": "src/llm_runtime/providers/v3_qwen3_process_factory.py",
    "experiments_init": "src/llm_runtime/providers/__init__.py",
    "framework_init": "src/llm_runtime/__init__.py",
    "factory": "src/llm_runtime/v3_execution_factory.py",
    "scheduler": "src/llm_runtime/v3_pipeline_scheduler.py",
    "transport": "src/llm_runtime/v3_ordered_rank_transport.py",
    "runner": "scripts/v3_qwen3_14b_benchmark.py",
}

data = {}
items = []

for key, file in files.items():
    path = ROOT / file

    if not path.is_file():
        raise FileNotFoundError(path)

    content = path.read_text()

    if "experiments.v3_" in content:
        raise ValueError(
            f"Historical module import still in {file}"
        )

    data[key] = content

    target = (
        "v3_qwen3_14b_benchmark.py"
        if key == "runner"
        else file
    )

    items.append({
        "key": key,
        "path": target,
    })

size = sum(
    len(v.encode("utf-8"))
    for v in data.values()
)

if size >= 800_000:
    raise ValueError(
        f"ConfigMap sources too large: {size}"
    )

service["metadata"]["name"] = SERVICE
service["spec"]["selector"] = {"app": NAME}

for port in service["spec"]["ports"]:
    new_port = int(
        RENDEZVOUS
        if port["name"] == "rendezvous"
        else CONTROL
    )

    port["port"] = new_port
    port["targetPort"] = new_port

jobs = []

for rank in (0, 1):
    job = deepcopy(base_job)

    job["metadata"]["name"] = f"{NAME}-rank{rank}"
    job["spec"]["completions"] = 1
    job["spec"]["parallelism"] = 1
    job["spec"].pop("completionMode", None)
    job["spec"]["backoffLimit"] = 0
    job["spec"]["activeDeadlineSeconds"] = 2700

    template = job["spec"]["template"]
    template["metadata"]["labels"] = {
        "app": NAME,
        "rank": str(rank),
    }

    pod = template["spec"]

    pod["hostname"] = f"{NAME}-rank{rank}"
    pod["subdomain"] = SERVICE
    pod["nodeSelector"] = {"pool": "gpu"}

    affinity = pod["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]

    affinity[0]["labelSelector"]["matchLabels"] = {
        "app": NAME
    }

    container = pod["containers"][0]

    container["command"] = [
        "python",
        "-u",
        "/workspace/v3_qwen3_14b_benchmark.py",
    ]

    container["resources"]["requests"]["memory"] = "6Gi"
    container["resources"]["limits"]["memory"] = "12Gi"

    assert str(
        container["resources"]["requests"]["nvidia.com/gpu"]
    ) == "1"

    assert str(
        container["resources"]["limits"]["nvidia.com/gpu"]
    ) == "1"

    envs = container["env"]

    for env in envs:
        if env["name"] == "RANK":
            env.pop("valueFrom", None)
            env["value"] = str(rank)

        elif env["name"] == "MASTER_ADDR":
            env["value"] = (
                f"{NAME}-rank0.{SERVICE}.llm.svc.cluster.local"
            )

        elif env["name"] == "MASTER_PORT":
            env["value"] = RENDEZVOUS

        elif env["name"] == "V3_CTRL_PORT":
            env["value"] = CONTROL

    envs.extend([
        {
            "name": "MODEL_DIR",
            "value": "/models/qwen3-14b",
        },
        {
            "name": "HF_HUB_OFFLINE",
            "value": "1",
        },
        {
            "name": "TOKENIZERS_PARALLELISM",
            "value": "false",
        },
        {
            "name": "OMP_NUM_THREADS",
            "value": "1",
        },
    ])

    container["volumeMounts"].append({
        "name": "model-files",
        "mountPath": "/models",
        "readOnly": True,
    })

    code_volume = next(
        x for x in pod["volumes"]
        if x["name"] == "test-code"
    )

    code_volume["projected"]["sources"] = [{
        "configMap": {
            "name": CM_NAME,
            "items": items,
        }
    }]

    pod["volumes"].append({
        "name": "model-files",
        "persistentVolumeClaim": {
            "claimName": f"model-files-qwen3-{rank}",
            "readOnly": True,
        },
    })

    jobs.append(job)

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": CM_NAME,
        "namespace": "llm",
    },
    "data": data,
}

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUT.write_text(
    yaml.safe_dump_all(
        [service] + jobs,
        sort_keys=False,
    )
)

CM_OUT.write_text(
    yaml.safe_dump(
        configmap,
        sort_keys=False,
    )
)

print(f"PASS: Created {OUT}")
print(f"PASS: Created {CM_OUT}")
print(
    "PASS: Rank 0 PVC=model-files-qwen3-0; "
    "rank 1 PVC=model-files-qwen3-1"
)
print(
    "PASS: Rank 0 owns 0-19; "
    "rank 1 owns 20-39 (runner assertions)"
)
print(f"ConfigMap source bytes: {size}")
