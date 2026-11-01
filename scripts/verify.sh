#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p test-results

python3 - <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
NS = "llm"

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "checks": [],
}

def command(args, input_text=None, timeout=300):
    result = subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode:
        raise RuntimeError(
            f"{' '.join(args)} failed: {result.stderr}"
        )

    return result.stdout.strip()

def kubectl(*args):
    return json.loads(command(
        ["kubectl", *args, "-o", "json"]
    ))

def ready(obj):
    return any(
        c.get("type") == "Ready"
        and c.get("status") == "True"
        for c in obj.get("status", {}).get("conditions", [])
    )

def verify(name, fn):
    try:
        details = fn()
        report["checks"].append({
            "name": name,
            "passed": True,
            "details": details,
        })
        print(f"PASS: {name}", flush=True)
        return True
    except Exception as exc:
        report["checks"].append({
            "name": name,
            "passed": False,
            "error": str(exc),
        })
        print(f"FAIL: {name}: {exc}", flush=True)
        return False

def check_context():
    current = command(["kubectl", "config", "current-context"])
    expected = (
        "arn:aws:eks:us-east-2:657173565477:"
        "cluster/llm-pipeline-v1"
    )
    assert current == expected, f"Unexpected context: {current}"
    return {"context": current}

def check_nodes():
    nodes = kubectl("get", "nodes")["items"]
    cpu = []
    gpu = []

    for node in nodes:
        if not ready(node):
            continue

        name = node["metadata"]["name"]
        pool = node["metadata"].get("labels", {}).get("pool")

        if pool == "cpu":
            cpu.append(name)

        if pool == "gpu":
            count = int(
                node["status"]["allocatable"]
                .get("nvidia.com/gpu", "0")
            )
            if count >= 1:
                gpu.append(name)

    assert len(cpu) >= 1, "No Ready CPU node"
    assert len(gpu) >= 2, "Fewer than two Ready GPU nodes"

    return {"cpu_nodes": cpu, "gpu_nodes": gpu}

def check_workers():
    nodes = kubectl("get", "nodes")["items"]

    node_labels = {
        n["metadata"]["name"]:
        n["metadata"].get("labels", {})
        for n in nodes
    }

    results = []

    for rank in range(2):
        pod = kubectl(
            "get", "pod", f"qwen3-{rank}", "-n", NS
        )

        assert ready(pod), f"Rank {rank} not Ready"

        node = pod["spec"]["nodeName"]
        assert node_labels[node].get("pool") == "gpu"

        runtime = next(
            c for c in pod["status"]["containerStatuses"]
            if c["name"] == "runtime"
        )

        results.append({
            "rank": rank,
            "uid": pod["metadata"]["uid"],
            "node": node,
            "restart_count": runtime["restartCount"],
        })

    assert results[0]["node"] != results[1]["node"], (
        "Both ranks placed on same node"
    )

    return results

def check_image():
    expected = (
        ROOT / "infrastructure/model-image.txt"
    ).read_text().strip()

    sts = kubectl(
        "get", "statefulset", "qwen3", "-n", NS
    )

    actual = next(
        c["image"]
        for c in sts["spec"]["template"]["spec"]["containers"]
        if c["name"] == "runtime"
    )

    assert actual == expected, "Runtime image digest differs"

    return {"image": actual}

def check_storage():
    names = [
        "model-files-qwen3-0",
        "model-files-qwen3-1",
    ]

    volumes = {}

    for name in names:
        claim = kubectl("get", "pvc", name, "-n", NS)

        assert claim["status"]["phase"] == "Bound"
        assert claim["spec"]["storageClassName"] == "llm-gp3"

        volumes[name] = claim["spec"]["volumeName"]

    snapshot = (
        ROOT / "test-results/last-lifecycle-v2.json"
    )

    if snapshot.exists():
        previous = json.loads(snapshot.read_text())
        assert volumes == previous["pvc_volumes"], (
            "Model-cache volume identities changed"
        )

    return volumes

def check_recovery():
    cm = kubectl(
        "get", "configmap",
        "qwen3-recovery-state",
        "-n", NS,
    )

    state = json.loads(cm["data"]["state"])

    assert state["mode"] == "RUNNING"
    assert state["phase"] == "SERVING"

    signatures = []

    for rank in range(2):
        pod = kubectl(
            "get", "pod", f"qwen3-{rank}", "-n", NS
        )

        runtime = next(
            c for c in pod["status"]["containerStatuses"]
            if c["name"] == "runtime"
        )

        signatures.append([
            pod["metadata"]["uid"],
            runtime["restartCount"],
        ])

    assert state["baseline"] == signatures, (
        "Persisted NCCL worker identities differ"
    )

    assert state.get("last_error") is None

    return {
        "generation": state["generation"],
        "recoveries": state["recoveries"],
        "last_recovery_seconds": state.get(
            "last_recovery_seconds"
        ),
    }

def check_gateway_and_inference():
    script = r'''
import json
import time
import urllib.request

base = "http://127.0.0.1:8080"

with urllib.request.urlopen(
    base + "/status", timeout=10
) as response:
    health = json.load(response)

assert health["ready"] is True, health
assert all(r["ready"] for r in health["ranks"])
assert health["controller"]["phase"] == "SERVING"

payload = json.dumps({
    "prompt": "What is the capital of France?",
    "max_new_tokens": 32,
}).encode()

request = urllib.request.Request(
    base + "/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

start = time.perf_counter()

with urllib.request.urlopen(
    request, timeout=180
) as response:
    result = json.load(response)

wall_ms = (time.perf_counter() - start) * 1000

assert result["generated_tokens"] > 0
assert "paris" in result["text"].lower()

print(json.dumps({
    "text": result["text"],
    "generated_tokens": result["generated_tokens"],
    "finish_reason": result["finish_reason"],
    "server_e2e_ms": result["metrics"]["e2e_ms"],
    "ttft_ms": result["metrics"]["ttft_ms"],
    "client_wall_ms": round(wall_ms, 3),
}))
'''

    result = command(
        [
            "kubectl", "exec", "-i", "-n", NS,
            "deployment/qwen3-coordinator",
            "--", "python", "-",
        ],
        input_text=script,
        timeout=240,
    )

    return json.loads(result)

print("\n=== V1 END-TO-END VERIFICATION ===")

checks = [
    ("Kubernetes context", check_context),
    ("CPU and GPU nodes", check_nodes),
    ("Distributed inference workers", check_workers),
    ("Pinned container image", check_image),
    ("Persistent model storage", check_storage),
    ("Durable recovery state", check_recovery),
]

all_passed = True

for name, fn in checks:
    all_passed = verify(name, fn) and all_passed

if all_passed:
    all_passed = verify(
        "Gateway and live inference",
        check_gateway_and_inference,
    )
else:
    print("SKIP: Inference test due to failed prerequisites")

report["passed"] = all_passed

timestamp = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)

path = ROOT / "test-results" / (
    f"verification-{timestamp}.json"
)

path.write_text(json.dumps(report, indent=2) + "\n")

print("\n=== RESULTS ===")
print("Overall:", "PASS" if all_passed else "FAIL")
print("Report:", path)

if not all_passed:
    sys.exit(1)
PY
