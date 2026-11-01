#!/usr/bin/env python3
"""Read-only cluster check executed from the CPU node."""

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

SSL_CONTEXT = ssl.create_default_context(
    cafile=f"{SA}/ca.crt"
)


def get(path):
    request = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(
        request, context=SSL_CONTEXT, timeout=10
    ) as response:
        return json.load(response)


def ready(pod):
    return any(
        condition["type"] == "Ready"
        and condition["status"] == "True"
        for condition in pod.get("status", {}).get("conditions", [])
    )


sts = get(f"/apis/apps/v1/namespaces/{NS}/statefulsets/qwen3")

assert sts["spec"]["replicas"] == 2
assert sts["status"].get("readyReplicas") == 2

expected_image = next(
    container["image"]
    for container in sts["spec"]["template"]["spec"]["containers"]
    if container["name"] == "runtime"
)

for rank in (0, 1):
    pod = get(f"/api/v1/namespaces/{NS}/pods/qwen3-{rank}")
    assert ready(pod), f"qwen3-{rank} is not Ready"

    image = next(
        container["image"]
        for container in pod["spec"]["containers"]
        if container["name"] == "runtime"
    )
    assert image == expected_image

    print(f"PASS: qwen3-{rank} Ready", flush=True)


for name in ("qwen3-coordinator", "qwen3-recovery"):
    deployment = get(
        f"/apis/apps/v1/namespaces/{NS}/deployments/{name}"
    )

    assert deployment["spec"]["replicas"] == 1
    assert deployment["status"].get("availableReplicas") == 1

    print(f"PASS: {name} available", flush=True)


cm = get(
    f"/api/v1/namespaces/{NS}/configmaps/qwen3-recovery-state"
)

state = json.loads(cm["data"]["state"])

assert state["mode"] == "RUNNING"
assert state["phase"] == "SERVING"

print("PASS: Recovery state RUNNING / SERVING", flush=True)
print("PASS: CPU-hosted maintenance probe completed", flush=True)
