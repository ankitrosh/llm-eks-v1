#!/usr/bin/env python3
"""Qwen3 maintenance executor: read-only preflight mode."""

import json
import os
import ssl
import urllib.request

NS = "llm"
API = (
    f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:"
    f"{os.environ.get('KUBERNETES_SERVICE_PORT', '443')}"
)
SA = "/var/run/secrets/kubernetes.io/serviceaccount"

with open(f"{SA}/token") as f:
    TOKEN = f.read().strip()

CONTEXT = ssl.create_default_context(cafile=f"{SA}/ca.crt")


def get(path):
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(
        req, context=CONTEXT, timeout=15
    ) as response:
        return json.load(response)


def check(condition, message):
    if not condition:
        raise RuntimeError(f"PREFLIGHT FAILED: {message}")
    print(f"PASS: {message}", flush=True)


def app(name, kind):
    return get(
        f"/apis/apps/v1/namespaces/{NS}/{kind}/{name}"
    )


def pod(rank):
    return get(
        f"/api/v1/namespaces/{NS}/pods/qwen3-{rank}"
    )


def cm(name):
    return get(
        f"/api/v1/namespaces/{NS}/configmaps/{name}"
    )


def runtime_container(spec):
    return next(
        c for c in spec["containers"]
        if c["name"] == "runtime"
    )


def is_ready(p):
    return any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in p.get("status", {}).get("conditions", [])
    )


def main():
    print("=== CPU MAINTENANCE EXECUTOR PREFLIGHT ===",
          flush=True)

    maintenance = json.loads(
        cm("qwen3-maintenance-state")["data"]["state"]
    )

    original = maintenance["original_image"]
    candidate = maintenance["candidate_image"]

    check(
        maintenance["phase"] == "IDLE",
        "No maintenance operation active"
    )
    check(
        original != candidate
        and "@sha256:" in original
        and "@sha256:" in candidate,
        "Original and candidate digests pinned"
    )

    sts = app("qwen3", "statefulsets")
    sts_spec = sts["spec"]

    check(
        sts_spec["replicas"] == 2,
        "StatefulSet has two replicas"
    )
    check(
        sts_spec["updateStrategy"]["type"] == "OnDelete",
        "Explicit OnDelete update strategy"
    )
    check(
        sts_spec["podManagementPolicy"] == "Parallel",
        "Parallel worker creation"
    )

    retention = sts_spec["persistentVolumeClaimRetentionPolicy"]

    check(
        retention["whenDeleted"] == "Retain"
        and retention["whenScaled"] == "Retain",
        "Persistent volume claims retained"
    )

    runtime = runtime_container(
        sts_spec["template"]["spec"]
    )

    check(
        runtime["image"] == original,
        "Original StatefulSet image is active"
    )

    print("\n=== GPU WORKERS ===", flush=True)
    nodes = set()

    for rank in (0, 1):
        p = pod(rank)
        container = runtime_container(p["spec"])
        status = next(
            c for c in p["status"]["containerStatuses"]
            if c["name"] == "runtime"
        )

        check(
            is_ready(p) and status["ready"],
            f"Rank {rank} Ready"
        )
        check(
            container["image"] == original,
            f"Rank {rank} uses original image"
        )

        nodes.add(p["spec"]["nodeName"])

        print(json.dumps({
            "rank": rank,
            "uid": p["metadata"]["uid"],
            "node": p["spec"]["nodeName"],
            "restart_count": status["restartCount"],
            "image": container["image"],
        }, indent=2), flush=True)

    check(
        len(nodes) == 2,
        "Ranks scheduled to distinct nodes"
    )

    print("\n=== CPU SERVICES ===", flush=True)

    for name in ("qwen3-coordinator", "qwen3-recovery"):
        deployment = app(name, "deployments")

        check(
            deployment["spec"]["replicas"] == 1
            and deployment["status"].get(
                "availableReplicas", 0
            ) == 1,
            f"{name} available"
        )

    recovery = json.loads(
        cm("qwen3-recovery-state")["data"]["state"]
    )

    check(
        recovery["mode"] == "RUNNING"
        and recovery["phase"] == "SERVING",
        "Recovery controller RUNNING / SERVING"
    )

    print("\n=== INFERENCE ACTIVITY ===", flush=True)

    url = (
        "http://qwen3-0.qwen3-headless."
        "llm.svc.cluster.local:8000/metrics"
    )

    with urllib.request.urlopen(
        url, timeout=15
    ) as response:
        metrics = json.load(response)

    check(
        metrics.get("active") is False,
        "No active inference request"
    )

    print("\n=== STARTUP PROBE CONFIGURATION ===",
          flush=True)

    print(json.dumps({
        "startupProbe": runtime.get("startupProbe"),
        "livenessProbe": runtime.get("livenessProbe"),
        "readinessProbe": runtime.get("readinessProbe"),
    }, indent=2), flush=True)

    print("\nOriginal:", original, flush=True)
    print("Candidate:", candidate, flush=True)

    print(
        "\nPREFLIGHT PASSED — NO WORKLOAD CHANGES",
        flush=True
    )


if __name__ == "__main__":
    main()
