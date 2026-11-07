"""Package the current GPT-2 NCCL source into a test-only ConfigMap."""

from pathlib import Path
import yaml

root = Path(".")
smoke = root / (
    "infrastructure/profiles/v3-2-5/manifests/"
    "v3-nccl-ordered-smoke.yaml"
)
assert smoke.exists(), smoke

job_name = "v3-gpt2-nccl-concurrent"
cm_name = "v3-gpt2-nccl-code"

modules = [
    "__init__.py",
    "model_metadata.py",
    "optimization_registry.py",
    "runtime_discovery.py",
    "v3_adapter_registry.py",
    "v3_attention_install.py",
    "v3_attention_plan.py",
    "v3_extension_contracts.py",
    "v3_manifest.py",
    "v3_model_inspector.py",
    "v3_plan_loader.py",
    "v3_plugin_registry.py",
    "v3_qwen3_adapter.py",
    "v3_runtime_plan.py",
    "v3_stage_bridge.py",
    "v3_pipeline_scheduler.py",
    "v3_execution_factory.py",
    "v3_ordered_rank_transport.py",
]

files = {
    "runner": "scripts/v3_nccl_gpt2_integration.py",
    "manifest": "examples/v3_gpt2_extension_nccl.yaml",
    "plugin": "experiments/v3_gpt2_extension.py",
    "stage_provider": "experiments/v3_gpt2_process_factory.py",
    "plugin_init": "experiments/__init__.py",
}

for i, module in enumerate(modules):
    files[f"module_{i:02d}"] = (
        f"src/llm_runtime/{module}"
    )

data = {}
items = []

for key, filename in files.items():
    path = root / filename
    assert path.is_file(), f"Missing source: {path}"

    data[key] = path.read_text()

    target = (
        path.name
        if filename.startswith("scripts/")
        else filename
    )

    items.append({
        "key": key,
        "path": str(target),
    })

source_bytes = sum(
    len(value.encode()) for value in data.values()
)
assert source_bytes < 800_000, (
    f"ConfigMap too large: {source_bytes} bytes"
)

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": cm_name,
        "namespace": "llm",
    },
    "data": data,
}

service, job = list(
    yaml.safe_load_all(smoke.read_text())
)

assert service["kind"] == "Service"
assert job["kind"] == "Job"

service["metadata"]["name"] = (
    f"{job_name}-headless"
)
service["spec"]["selector"]["app"] = job_name

for port in service["spec"]["ports"]:
    port["port"] = (
        29638
        if port["name"] == "rendezvous"
        else 29639
    )
    port["targetPort"] = port["port"]

job["metadata"]["name"] = job_name

pod = job["spec"]["template"]
pod["metadata"]["labels"]["app"] = job_name
pod["spec"]["subdomain"] = (
    f"{job_name}-headless"
)

affinity = pod["spec"]["affinity"]["podAntiAffinity"]
required = affinity[
    "requiredDuringSchedulingIgnoredDuringExecution"
]

required[0]["labelSelector"]["matchLabels"][
    "app"
] = job_name

container = pod["spec"]["containers"][0]

container["command"] = [
    "python",
    "-u",
    "/workspace/v3_nccl_gpt2_integration.py",
]

for env in container["env"]:
    if env["name"] == "MASTER_ADDR":
        env["value"] = (
            f"{job_name}-0."
            f"{job_name}-headless."
            "llm.svc.cluster.local"
        )
    elif env["name"] == "MASTER_PORT":
        env["value"] = "29638"
    elif env["name"] == "V3_CTRL_PORT":
        env["value"] = "29639"

pod["spec"]["volumes"][0]["projected"][
    "sources"
] = [
    {
        "configMap": {
            "name": cm_name,
            "items": items,
        }
    }
]


# Dedicated shared memory for NCCL.
# Applies only to this new GPT-2 test Job.
pod_spec = pod["spec"]

pod_spec["volumes"].append({
    "name": "v3-gpt2-shm",
    "emptyDir": {
        "medium": "Memory",
        "sizeLimit": "1Gi",
    },
})

container.setdefault("volumeMounts", []).append({
    "name": "v3-gpt2-shm",
    "mountPath": "/dev/shm",
})

# Memory-backed emptyDir usage counts toward container memory.
container["resources"].setdefault("requests", {})[
    "memory"
] = "2Gi"

container["resources"].setdefault("limits", {})[
    "memory"
] = "6Gi"

manifest_path = root / (
    "infrastructure/profiles/v3-2-5/manifests/"
    "v3-gpt2-nccl-concurrent.yaml"
)

manifest_path.write_text(
    yaml.safe_dump_all(
        [service, job], sort_keys=False
    )
)

Path(
    "/tmp/v3-gpt2-nccl-code-configmap.yaml"
).write_text(
    yaml.safe_dump(configmap, sort_keys=False)
)

print("PASS: GPT-2 NCCL Job and ConfigMap prepared")
print("Source bytes:", source_bytes)
print("Manifest:", manifest_path)
