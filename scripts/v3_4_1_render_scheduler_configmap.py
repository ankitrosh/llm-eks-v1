"""Render GPU scheduler test manifests without building an image."""

import json
from pathlib import Path

ROOT = Path("deploy/v3_4_1")
SOURCE = ROOT / "rendered"
OUTPUT = ROOT / "scheduler-mount-test"

PLUGIN = Path(
    "examples/v3_4_1/runtime_plugin/src/"
    "user_vllm_ext/scheduler.py"
)

SCHEDULER = "user_vllm_ext.scheduler.UserScheduler"
CONFIGMAP = "v341-user-scheduler"

assert PLUGIN.is_file(), PLUGIN
assert "V341_USER_SCHEDULER_INSTANTIATED" in PLUGIN.read_text()

OUTPUT.mkdir(parents=True, exist_ok=True)

configmap = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {
        "name": CONFIGMAP,
        "namespace": "llm",
    },
    "data": {
        "__init__.py": "",
        "scheduler.py": PLUGIN.read_text(),
    },
}

(OUTPUT / "scheduler-configmap.json").write_text(
    json.dumps(configmap, indent=2) + "\n"
)

for rank in (0, 1):
    name = f"vllm-rank{rank}.json"

    deployment = json.loads(
        (SOURCE / name).read_text()
    )

    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert container["name"] == "vllm"
    assert container["command"][:2] == ["vllm", "serve"]

    command = container["command"]
    assert "--scheduler-cls" not in command

    command.extend([
        "--scheduler-cls",
        SCHEDULER,
    ])

    # Load the Python module through a Kubernetes
    # volume rather than a rebuilt container image.
    pod.setdefault("volumes", []).append({
        "name": "v341-scheduler",
        "configMap": {
            "name": CONFIGMAP,
            "items": [
                {
                    "key": "__init__.py",
                    "path": "user_vllm_ext/__init__.py",
                },
                {
                    "key": "scheduler.py",
                    "path": "user_vllm_ext/scheduler.py",
                },
            ],
        },
    })

    container.setdefault("volumeMounts", []).append({
        "name": "v341-scheduler",
        "mountPath": "/opt/v341",
        "readOnly": True,
    })

    assert not any(
        item["name"] == "PYTHONPATH"
        for item in container.get("env", [])
    )

    container.setdefault("env", []).append({
        "name": "PYTHONPATH",
        "value": "/opt/v341",
    })

    # Keep the existing deployment scaled down.
    assert deployment["spec"]["replicas"] == 0

    (OUTPUT / name).write_text(
        json.dumps(deployment, indent=2) + "\n"
    )

    print(f"PASS: Rank {rank} scheduler mounted")
    print("Selected:", SCHEDULER)
    print("Image:", container["image"])

print("V341_SCHEDULER_MOUNT_RENDER_PASS")
print("No Kubernetes resources modified")
