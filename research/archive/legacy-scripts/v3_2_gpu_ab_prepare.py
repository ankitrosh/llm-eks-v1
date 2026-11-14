"""Build isolated V3.2 A+B GPU deployment."""
from pathlib import Path
import yaml

base = Path(
    "deploy/manifests/v3_2_qwen3_14b_perf.yaml"
)
runner = Path(
    "scripts/v3_2_gpu_ab_pipeline.py"
)

assert base.exists(), base
assert runner.exists(), runner

data = {}
items = []

source_files = sorted(
    Path("src/llm_runtime").rglob("*.py")
)

for i, path in enumerate(source_files):
    key = f"module_{i:03d}"

    data[key] = path.read_text()

    items.append({
        "key": key,
        "path": path.relative_to("src").as_posix(),
    })

required = (
    "llm_runtime/v3_pipeline_scheduler.py",
    "llm_runtime/v3_1_nrank_transport.py",
    "llm_runtime/v3_2_kv_storage.py",
    "llm_runtime/v3_2_qwen3_stage_prefill.py",
    "llm_runtime/v3_2_qwen3_stage_batch.py",
)

included = {item["path"] for item in items}

for path in required:
    assert path in included, path

data["ab_runner"] = runner.read_text()

items.append({
    "key": "ab_runner",
    "path": "scripts/v3_2_gpu_ab_pipeline.py",
})

size = sum(
    len(content.encode())
    for content in data.values()
)

assert size < 900_000, (
    f"ConfigMap exceeds size budget: {size}"
)

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": "v32-ab-source",
        "namespace": "llm",
    },
    "data": data,
}

Path("/tmp/v32-ab-configmap.yaml").write_text(
    yaml.safe_dump(configmap, sort_keys=False)
)

documents = list(yaml.safe_load_all(
    base.read_text().replace(
        "v32-qwen3-14b-perf",
        "v32-qwen3-14b-ab",
    )
))

assert {
    document["kind"]
    for document in documents
} == {"Service", "Job"}

jobs = [
    document for document in documents
    if document["kind"] == "Job"
]

assert len(jobs) == 2

for job in jobs:
    spec = job["spec"]["template"]["spec"]

    volume = next(
        v for v in spec["volumes"]
        if v["name"] == "v32-source"
    )

    volume["configMap"] = {
        "name": "v32-ab-source",
        "items": items,
    }

    assert all(
        v["name"] != "test-code"
        for v in spec["volumes"]
    )

    container = next(
        c for c in spec["containers"]
        if c["name"] == "runtime"
    )

    container["command"] = [
        "python3",
        "-u",
        "/opt/v32/scripts/v3_2_gpu_ab_pipeline.py",
    ]

    container.pop("args", None)

    assert any(
        m["name"] == "v32-source"
        and m["mountPath"] == "/opt/v32"
        for m in container["volumeMounts"]
    )

    env = {
        e["name"]: e
        for e in container["env"]
    }

    overrides = {
        "V31_CTRL_PORT": "29679",
        "V32_REQUESTS": "4",
        "V32_PROMPT_TOKENS": "128",
        "V32_NEW_TOKENS": "64",
        "AB_WARMUP_TOKENS": "64",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "eth0",
        "PYTHONPATH": "/opt/v32",
    }

    for name, value in overrides.items():
        env[name] = {
            "name": name,
            "value": value,
        }

    container["env"] = list(env.values())

    assert (
        container["resources"]["limits"]["nvidia.com/gpu"]
        == "1"
    )

    print("Validated:", job["metadata"]["name"])

target = Path(
    "deploy/manifests/v3_2_qwen3_14b_ab.yaml"
)

target.write_text(
    yaml.safe_dump_all(
        documents,
        sort_keys=False,
    )
)

print(
    f"CREATED {target}; "
    f"source ConfigMap {size} bytes"
)
