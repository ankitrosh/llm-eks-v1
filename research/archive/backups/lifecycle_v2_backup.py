#!/usr/bin/env python3
"""Controlled EKS pause/resume for the Qwen3 distributed pipeline."""

import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REGION = "us-east-2"
CLUSTER = "llm-pipeline-v1"
NAMESPACE = "llm"

CPU = "cpu-system"
GPU = "gpu-a10g"

BASE = Path(__file__).resolve().parent
STATE = BASE / ".lifecycle-v2-state.json"
LAST = BASE / "test-results" / "last-lifecycle-v2.json"


def cmd(*argv, timeout=120):
    argv = [str(x) for x in argv]
    print("+", shlex.join(argv), flush=True)

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if proc.returncode:
        raise RuntimeError(
            f"Command failed: {shlex.join(argv)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    return proc.stdout.strip()


def kube(*args):
    return cmd("kubectl", *args)


def kjson(*args):
    return json.loads(kube(*args, "-o", "json"))


def ng_config(name):
    data = json.loads(cmd(
        "aws", "eks", "describe-nodegroup",
        "--region", REGION,
        "--cluster-name", CLUSTER,
        "--nodegroup-name", name,
        "--output", "json",
    ))

    return data["nodegroup"]["scalingConfig"]


def update_ng(name, target):
    current = ng_config(name)
    keys = ("minSize", "maxSize", "desiredSize")

    if all(current[k] == target[k] for k in keys):
        print(f"{name}: scaling already matches", flush=True)
        return

    spec = ",".join(f"{k}={target[k]}" for k in keys)

    response = json.loads(cmd(
        "aws", "eks", "update-nodegroup-config",
        "--region", REGION,
        "--cluster-name", CLUSTER,
        "--nodegroup-name", name,
        "--scaling-config", spec,
        "--output", "json",
    ))

    update_id = response["update"]["id"]
    deadline = time.monotonic() + 900

    while time.monotonic() < deadline:
        data = json.loads(cmd(
            "aws", "eks", "describe-update",
            "--region", REGION,
            "--name", CLUSTER,
            "--nodegroup-name", name,
            "--update-id", update_id,
            "--output", "json",
        ))["update"]

        if data["status"] == "Successful":
            print(f"{name}: scaling update successful")
            return

        if data["status"] in ("Failed", "Cancelled"):
            raise RuntimeError(
                f"{name}: scaling failed: {data}"
            )

        time.sleep(10)

    raise TimeoutError(f"Scaling timed out: {name}")


def replicas(kind, name):
    data = kjson("get", kind, name, "-n", NAMESPACE)
    return int(data["spec"]["replicas"])


def scale(kind, name, count):
    kube(
        "scale", f"{kind}/{name}",
        "-n", NAMESPACE,
        f"--replicas={count}",
    )


def pod_items():
    return kjson("get", "pods", "-n", NAMESPACE)["items"]


def is_ready(obj):
    return any(
        c.get("type") == "Ready"
        and c.get("status") == "True"
        for c in obj.get("status", {}).get("conditions", [])
    )


def wait_pods(predicate, message, timeout=600):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        pods = pod_items()

        if predicate(pods):
            print("OK:", message, flush=True)
            return

        time.sleep(5)

    raise TimeoutError(message)


def wait_absent(prefixes, timeout=300):
    wait_pods(
        lambda pods: not any(
            any(
                p["metadata"]["name"].startswith(prefix)
                for prefix in prefixes
            )
            for p in pods
        ),
        f"Pods absent: {prefixes}",
        timeout,
    )


def wait_ready_names(names, timeout=1200):
    required = set(names)

    wait_pods(
        lambda pods: required.issubset({
            p["metadata"]["name"]
            for p in pods
            if is_ready(p)
        }),
        f"Pods ready: {names}",
        timeout,
    )


def wait_ready_prefix(prefix, timeout=180):
    wait_pods(
        lambda pods: any(
            p["metadata"]["name"].startswith(prefix)
            and is_ready(p)
            for p in pods
        ),
        f"Pod ready: {prefix}",
        timeout,
    )


def wait_nodes(pool, count, timeout=1200):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        nodes = kjson(
            "get", "nodes", "-l", f"pool={pool}"
        )["items"]

        ready = sum(is_ready(node) for node in nodes)

        if ready >= count:
            print(f"{pool}: {ready} Ready nodes")
            return

        time.sleep(10)

    raise TimeoutError(
        f"Waiting for {count} Ready {pool} nodes"
    )


def pvc_volumes():
    claims = kjson(
        "get", "pvc", "-n", NAMESPACE
    )["items"]

    data = {
        c["metadata"]["name"]: c["spec"].get("volumeName")
        for c in claims
        if c.get("status", {}).get("phase") == "Bound"
    }

    names = (
        "model-files-qwen3-0",
        "model-files-qwen3-1",
    )

    if any(not data.get(name) for name in names):
        raise RuntimeError(
            f"Checkpoint PVC missing/unbound: {data}"
        )

    return {name: data[name] for name in names}


def controller_status():
    code = (
        "import json,urllib.request;"
        "print(json.dumps(json.load(urllib.request.urlopen("
        "'http://127.0.0.1:8081/status',timeout=3))))"
    )

    output = kube(
        "exec", "-n", NAMESPACE,
        "deployment/qwen3-recovery",
        "--", "python", "-c", code,
    )

    return json.loads(output)


def save(state):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2) + "\n"
    )
    temporary.replace(STATE)


def pause():
    if STATE.exists():
        raise RuntimeError(
            f"Saved lifecycle state exists: {STATE}. "
            "Resume before starting another pause."
        )

    sts = kjson(
        "get", "statefulset", "qwen3",
        "-n", NAMESPACE,
    )

    retention = sts["spec"].get(
        "persistentVolumeClaimRetentionPolicy", {}
    )

    if retention.get("whenScaled", "Retain") != "Retain":
        raise RuntimeError(
            "PVC retention is not Retain; refusing pause"
        )

    status = controller_status()

    if status.get("phase") != "SERVING":
        raise RuntimeError(
            f"Controller not SERVING: {status.get('phase')}"
        )

    # Do not intentionally interrupt an active inference.
    active = kube(
        "exec", "-n", NAMESPACE,
        "qwen3-0", "-c", "runtime",
        "--", "python", "-c",
        "import json,urllib.request;"
        "print(json.load(urllib.request.urlopen("
        "'http://127.0.0.1:8000/metrics',timeout=3))['active'])",
    )

    if active.strip() != "False":
        raise RuntimeError(
            "Inference may be active; pause while idle"
        )

    counts = {
        "qwen3": replicas("statefulset", "qwen3"),
        "qwen3-coordinator": replicas(
            "deployment", "qwen3-coordinator"
        ),
        "qwen3-recovery": replicas(
            "deployment", "qwen3-recovery"
        ),
    }

    expected = {
        "qwen3": 2,
        "qwen3-coordinator": 1,
        "qwen3-recovery": 1,
    }

    if counts != expected:
        raise RuntimeError(
            f"Unexpected replicas: {counts}"
        )

    # Save original state BEFORE changing anything.
    state = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase": "pausing",
        "replicas": counts,
        "nodegroups": {
            CPU: ng_config(CPU),
            GPU: ng_config(GPU),
        },
        "pvc_volumes": pvc_volumes(),
    }

    save(state)
    print("Original lifecycle state saved")

    # 1. Stop admission of new requests.
    scale("deployment", "qwen3-coordinator", 0)
    wait_absent(["qwen3-coordinator-"])

    # 2. Disable the recovery watchdog.
    scale("deployment", "qwen3-recovery", 0)
    wait_absent(["qwen3-recovery-"])

    # 3. Stop both NCCL workers.
    scale("statefulset", "qwen3", 0)
    wait_absent(["qwen3-0", "qwen3-1"])

    # 4. Confirm model cache volumes were retained.
    if pvc_volumes() != state["pvc_volumes"]:
        raise RuntimeError(
            "Checkpoint PVC identities changed during pause"
        )

    # 5. Scale GPU workers down, then CPU node.
    for name in (GPU, CPU):
        target = dict(state["nodegroups"][name])
        target.update(minSize=0, desiredSize=0)
        update_ng(name, target)

    state["phase"] = "paused"
    save(state)

    print(
        "PAUSED: CPU and GPU node groups desiredSize=0. "
        "EKS control plane and EBS remain billable."
    )


def resume():
    if not STATE.exists():
        raise RuntimeError(
            "No saved pause state found"
        )

    state = json.loads(STATE.read_text())

    if state.get("phase") not in (
        "pausing", "paused", "resuming"
    ):
        raise RuntimeError(
            f"Unexpected lifecycle phase: {state.get('phase')}"
        )

    state["phase"] = "resuming"
    save(state)

    # 1. Restore CPU node and Kubernetes system capacity.
    update_ng(CPU, state["nodegroups"][CPU])

    wait_nodes(
        "cpu",
        state["nodegroups"][CPU]["desiredSize"],
    )

    # 2. Restore both GPU nodes.
    update_ng(GPU, state["nodegroups"][GPU])

    wait_nodes(
        "gpu",
        state["nodegroups"][GPU]["desiredSize"],
    )

    # 3. Start both inference ranks together.
    scale(
        "statefulset", "qwen3",
        state["replicas"]["qwen3"],
    )

    wait_ready_names(
        ["qwen3-0", "qwen3-1"]
    )

    # Make sure checkpoint volumes were preserved.
    if pvc_volumes() != state["pvc_volumes"]:
        raise RuntimeError(
            "Checkpoint PVC identities differ after resume"
        )

    # 4. Restore recovery controller AFTER GPU readiness.
    scale(
        "deployment", "qwen3-recovery",
        state["replicas"]["qwen3-recovery"],
    )

    wait_ready_prefix("qwen3-recovery-")

    deadline = time.monotonic() + 120

    while time.monotonic() < deadline:
        try:
            status = controller_status()

            if (
                status.get("phase") == "SERVING"
                and status.get("auto_recover")
            ):
                break

        except Exception as exc:
            print(
                "Controller not ready yet:",
                exc,
                flush=True,
            )

        time.sleep(5)

    else:
        raise TimeoutError(
            "Recovery controller did not enter SERVING"
        )

    # 5. Allow requests only once recovery is healthy.
    scale(
        "deployment", "qwen3-coordinator",
        state["replicas"]["qwen3-coordinator"],
    )

    wait_ready_prefix("qwen3-coordinator-")

    LAST.parent.mkdir(
        parents=True, exist_ok=True
    )

    LAST.write_text(
        json.dumps(
            {**state, "phase": "resumed"},
            indent=2,
        ) + "\n"
    )

    STATE.unlink()

    print(
        "RESUMED: Both GPU ranks Ready, "
        "recovery controller SERVING, "
        "gateway Ready."
    )


if __name__ == "__main__":
    try:
        if (
            len(sys.argv) != 2
            or sys.argv[1] not in ("pause", "resume")
        ):
            raise SystemExit(
                "Usage: python3 lifecycle_v2.py {pause|resume}"
            )

        if sys.argv[1] == "pause":
            pause()
        else:
            resume()

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            f"Snapshot (if created): {STATE}",
            file=sys.stderr,
        )
        sys.exit(1)
