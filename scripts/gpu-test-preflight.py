#!/usr/bin/env python3

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-results/snapshots"
OUT.mkdir(parents=True, exist_ok=True)

NAMESPACE = "llm"
EXPECTED_CONTEXT = (
    "arn:aws:eks:us-east-2:657173565477:"
    "cluster/llm-pipeline-v1"
)


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def kubectl(kind, name, namespace=NAMESPACE):
    args = ["kubectl", "get", kind, name]
    if namespace:
        args += ["-n", namespace]
    return json.loads(run(*args, "-o", "json"))


def is_ready(pod):
    return any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in pod["status"].get("conditions", [])
    )


print("=== GPU CANDIDATE TEST PREFLIGHT ===")

context = run("kubectl", "config", "current-context")
assert context == EXPECTED_CONTEXT, "Wrong Kubernetes context"

original = (
    ROOT / "infrastructure/model-image.txt"
).read_text().strip()

candidate = (
    ROOT / "infrastructure/rebuilt-image.txt"
).read_text().strip()

assert original != candidate
assert "@sha256:" in original
assert "@sha256:" in candidate

sts = kubectl("statefulset", "qwen3")
assert sts["spec"]["replicas"] == 2
assert sts["spec"]["updateStrategy"]["type"] == "OnDelete"
assert sts["spec"]["podManagementPolicy"] == "Parallel"

retention = sts["spec"].get(
    "persistentVolumeClaimRetentionPolicy", {}
)

assert retention.get("whenScaled", "Retain") == "Retain"
assert retention.get("whenDeleted", "Retain") == "Retain"

runtime = next(
    c for c in sts["spec"]["template"]["spec"]["containers"]
    if c["name"] == "runtime"
)

assert runtime["image"] == original

controller = kubectl("deployment", "qwen3-recovery")
gateway = kubectl("deployment", "qwen3-coordinator")

for name, deployment in [
    ("recovery", controller),
    ("gateway", gateway),
]:
    assert deployment["spec"]["replicas"] == 1, name
    assert deployment["status"].get("availableReplicas") == 1, name

cm = kubectl("configmap", "qwen3-recovery-state")
state = json.loads(cm["data"]["state"])

assert state["mode"] == "RUNNING", state
assert state["phase"] == "SERVING", state

workers = []
nodes = set()

for rank in (0, 1):
    pod = kubectl("pod", f"qwen3-{rank}")

    assert is_ready(pod), f"Rank {rank} is not Ready"

    container = next(
        c for c in pod["spec"]["containers"]
        if c["name"] == "runtime"
    )
    assert container["image"] == original

    node_name = pod["spec"]["nodeName"]
    node = kubectl("node", node_name, namespace=None)

    assert node["metadata"]["labels"].get("pool") == "gpu"
    assert int(
        node["status"]["allocatable"].get("nvidia.com/gpu", "0")
    ) >= 1

    nodes.add(node_name)

    workers.append({
        "rank": rank,
        "pod_uid": pod["metadata"]["uid"],
        "node": node_name,
        "image": container["image"],
    })

assert len(nodes) == 2, "Workers do not occupy two separate GPUs"

storage = {}

for rank in (0, 1):
    name = f"model-files-qwen3-{rank}"
    pvc = kubectl("pvc", name)

    assert pvc["status"]["phase"] == "Bound"
    assert pvc["spec"]["storageClassName"] == "llm-gp3"

    pv_name = pvc["spec"]["volumeName"]
    pv = kubectl("pv", pv_name, namespace=None)

    volume_id = pv["spec"]["csi"]["volumeHandle"]

    storage[name] = {
        "pv": pv_name,
        "ebs_volume": volume_id,
    }

snapshot = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "context": context,
    "original_image": original,
    "candidate_image": candidate,
    "statefulset": sts,
    "controller_deployment": controller,
    "gateway_deployment": gateway,
    "recovery_configmap": cm,
    "recovery_state": state,
    "workers": workers,
    "storage": storage,
}

path = OUT / "gpu-candidate-preflight.json"
path.write_text(json.dumps(snapshot, indent=2) + "\n")

print("PASS: Correct Kubernetes context")
print("PASS: Original image pinned")
print("PASS: Candidate image pinned")
print("PASS: StatefulSet uses OnDelete and Parallel")
print("PASS: EBS PVC retention configured")
print("PASS: Gateway and recovery controller available")
print("PASS: Recovery state RUNNING / SERVING")
print("PASS: Both GPU workers Ready on separate nodes")

print("\n=== PERSISTENT MODEL VOLUMES ===")
for name, info in storage.items():
    print(name, info["ebs_volume"])

print("\nOriginal image:", original)
print("Candidate image:", candidate)
print("\nRollback snapshot:", path)
print("\nPREFLIGHT PASSED — No cluster resources changed")
