"""Generate an isolated Qwen3 NCCL test Job and source ConfigMap."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(".")
BASE = ROOT / (
    "infrastructure/profiles/v3-2-5/manifests/"
    "v3-gpt2-nccl-concurrent.yaml"
)

assert BASE.is_file(), (
    "Missing verified GPT-2 deployment manifest"
)

docs = list(yaml.safe_load_all(BASE.read_text()))
assert len(docs) == 2

service = deepcopy(next(
    x for x in docs if x["kind"] == "Service"
))
job = deepcopy(next(
    x for x in docs if x["kind"] == "Job"
))

NAME = "v3-qwen3-nccl-concurrent"
CM_NAME = "v3-qwen3-nccl-code"

service["metadata"]["name"] = NAME + "-headless"
service["spec"]["selector"]["app"] = NAME

for port in service["spec"]["ports"]:
    value = (
        29648
        if port["name"] == "rendezvous"
        else 29649
    )
    port["port"] = value
    port["targetPort"] = value

job["metadata"]["name"] = NAME

template = job["spec"]["template"]
template["metadata"]["labels"]["app"] = NAME

spec = template["spec"]
spec["subdomain"] = NAME + "-headless"

anti_affinity = spec["affinity"][
    "podAntiAffinity"
]["requiredDuringSchedulingIgnoredDuringExecution"]

anti_affinity[0]["labelSelector"]["matchLabels"][
    "app"
] = NAME

container = spec["containers"][0]

container["command"] = [
    "python",
    "-u",
    "/workspace/v3_nccl_qwen3_integration.py",
]

for env in container["env"]:
    if env["name"] == "MASTER_ADDR":
        env["value"] = (
            NAME + "-0." + NAME
            + "-headless.llm.svc.cluster.local"
        )
    elif env["name"] == "MASTER_PORT":
        env["value"] = "29648"
    elif env["name"] == "V3_CTRL_PORT":
        env["value"] = "29649"

# Keep the working GPU and shared-memory configuration.
assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"

assert any(
    m["mountPath"] == "/dev/shm"
    for m in container["volumeMounts"]
)

assert any(
    v.get("emptyDir", {}).get("medium") == "Memory"
    for v in spec["volumes"]
)

files = {
    "runner": (
        "scripts/v3_nccl_qwen3_integration.py",
        "v3_nccl_qwen3_integration.py",
    ),
    "provider": (
        "experiments/v3_qwen3_process_factory.py",
        "experiments/v3_qwen3_process_factory.py",
    ),
    "experiments_init": (
        "experiments/__init__.py",
        "experiments/__init__.py",
    ),
    "framework_init": (
        "src/llm_runtime/__init__.py",
        "src/llm_runtime/__init__.py",
    ),
    "factory": (
        "src/llm_runtime/v3_execution_factory.py",
        "src/llm_runtime/v3_execution_factory.py",
    ),
    "scheduler": (
        "src/llm_runtime/v3_pipeline_scheduler.py",
        "src/llm_runtime/v3_pipeline_scheduler.py",
    ),
    "transport": (
        "src/llm_runtime/v3_ordered_rank_transport.py",
        "src/llm_runtime/v3_ordered_rank_transport.py",
    ),
}

data = {}
items = []

for key, (source, target) in files.items():
    path = ROOT / source
    assert path.is_file(), source

    data[key] = path.read_text()
    items.append({
        "key": key,
        "path": target,
    })

total_bytes = sum(
    len(value.encode("utf-8"))
    for value in data.values()
)

assert total_bytes < 800_000, total_bytes

volume = next(
    v for v in spec["volumes"]
    if v["name"] == "test-code"
)

volume["projected"]["sources"] = [
    {
        "configMap": {
            "name": CM_NAME,
            "items": items,
        }
    }
]

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": CM_NAME,
        "namespace": "llm",
    },
    "data": data,
}

destination = ROOT / (
    "infrastructure/profiles/v3-3-3/manifests/"
    "v3-qwen3-nccl-concurrent.yaml"
)

destination.parent.mkdir(
    parents=True, exist_ok=True
)

destination.write_text(
    yaml.safe_dump_all(
        [service, job],
        sort_keys=False,
    )
)

Path(
    "/tmp/v3-qwen3-nccl-code-configmap.yaml"
).write_text(
    yaml.safe_dump(
        configmap,
        sort_keys=False,
    )
)

print("PASS: Qwen3 NCCL deployment prepared")
print("PASS: Generic execution factory packaged")
print("PASS: Existing NCCL image and /dev/shm preserved")
print("Source bytes:", total_bytes)
