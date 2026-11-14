"""Create isolated V3.2 GPU Jobs from the verified V3 deployment."""

from copy import deepcopy
from pathlib import Path
import yaml

root = Path(".")
base = list(yaml.safe_load_all(
    Path(
        "deploy/manifests/v3-qwen3-14b-benchmark.yaml"
    ).read_text()
))

assert sum(x["kind"] == "Service" for x in base) == 1
assert sum(x["kind"] == "Job" for x in base) == 2

# Mount the current CloudShell source, including untracked V3.2 files.
sources = sorted(Path("src/llm_runtime").rglob("*.py"))

data = {}
items = []

for i, path in enumerate(sources):
    key = f"module_{i:03d}"

    data[key] = path.read_text()
    items.append({
        "key": key,
        "path": path.relative_to("src").as_posix(),
    })

data["runner"] = Path(
    "scripts/v3_2_gpu_qwen3_smoke.py"
).read_text()

items.append({
    "key": "runner",
    "path": "scripts/v3_2_gpu_qwen3_smoke.py",
})

size = sum(len(value.encode()) for value in data.values())
assert size < 900_000, f"ConfigMap too large: {size}"

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": "v32-a10g-source",
        "namespace": "llm",
    },
    "data": data,
}

Path("/tmp/v32-a10g-configmap.yaml").write_text(
    yaml.safe_dump(configmap, sort_keys=False)
)


def rename(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)

    if isinstance(value, list):
        return [rename(x, old, new) for x in value]

    if isinstance(value, dict):
        return {
            k: rename(v, old, new)
            for k, v in value.items()
        }

    return value


for count in (1, 4):
    prefix = f"v32-qwen3-14b-r{count}"
    output = []

    for original in base:
        document = rename(
            deepcopy(original),
            "v3-qwen3-14b-benchmark",
            prefix,
        )

        if document["kind"] == "Job":
            rank = int(document["metadata"]["name"][-1])
            spec = document["spec"]["template"]["spec"]

            container = next(
                c for c in spec["containers"]
                if c["name"] == "runtime"
            )

            container["command"] = [
                "python3",
                "-u",
                "/opt/v32/scripts/v3_2_gpu_qwen3_smoke.py",
            ]
            container.pop("args", None)

            overrides = {
                "RANK": str(rank),
                "WORLD_SIZE": "2",
                "MASTER_ADDR": (
                    f"{prefix}-rank0."
                    f"{prefix}-headless.llm.svc.cluster.local"
                ),
                "PYTHONPATH": "/opt/v32",
                "MODEL_DIR": "/models/qwen3-14b",
                "NCCL_IB_DISABLE": "1",
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_DEBUG": "WARN",
                "V32_REQUESTS": str(count),
                "V32_PROMPT_TOKENS": "16",
                "V32_NEW_TOKENS": "8",
            }

            container["env"] = [
                e for e in container.get("env", [])
                if e["name"] not in overrides
            ] + [
                {"name": name, "value": value}
                for name, value in overrides.items()
            ]

            assert (
                container["resources"]["limits"]["nvidia.com/gpu"]
                == "1"
            )

            # V32_SOURCE_MOUNT_FIX_V1
            # The original V3 Job mounts "test-code".
            # V3.2 uses its own complete source at /opt/v32.
            # Do not retain an obsolete ConfigMap reference.
            old_volumes = [
                v for v in spec["volumes"]
                if v["name"] == "test-code"
            ]
            assert len(old_volumes) == 1, old_volumes

            spec["volumes"] = [
                v for v in spec["volumes"]
                if v["name"] != "test-code"
            ]

            for current in spec["containers"]:
                current["volumeMounts"] = [
                    m for m in current.get("volumeMounts", [])
                    if m["name"] != "test-code"
                ]

            container["volumeMounts"].append({
                "name": "v32-source",
                "mountPath": "/opt/v32",
                "readOnly": True,
            })

            spec["volumes"].append({
                "name": "v32-source",
                "configMap": {
                    "name": "v32-a10g-source",
                    "items": items,
                },
            })

            document["spec"]["activeDeadlineSeconds"] = 900

        output.append(document)

    target = Path(
        f"deploy/manifests/v3_2_qwen3_14b_r{count}.yaml"
    )

    target.write_text(
        yaml.safe_dump_all(output, sort_keys=False)
    )

    print(f"CREATED {target}")

print(f"CREATED V3.2 ConfigMap: {size} bytes")
