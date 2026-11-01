#!/usr/bin/env python3
"""Test optimistic concurrency against an isolated ConfigMap."""

import copy
import json
import os
import ssl
import urllib.error
import urllib.request
import uuid

NS = "llm"
NAME = "qwen3-maintenance-cas-test"

API = (
    f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:"
    f"{os.environ.get('KUBERNETES_SERVICE_PORT', '443')}"
)

SA = "/var/run/secrets/kubernetes.io/serviceaccount"

with open(f"{SA}/token") as f:
    TOKEN = f.read().strip()

CONTEXT = ssl.create_default_context(
    cafile=f"{SA}/ca.crt"
)


def request(method, name, obj=None):
    path = f"/api/v1/namespaces/{NS}/configmaps/{name}"

    headers = {"Authorization": f"Bearer {TOKEN}"}
    data = None

    if obj is not None:
        data = json.dumps(obj).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        API + path,
        method=method,
        headers=headers,
        data=data,
    )

    with urllib.request.urlopen(
        req, context=CONTEXT, timeout=15
    ) as response:
        return json.load(response)


def main():
    print("=== MAINTENANCE CAS PROBE ===", flush=True)

    # Two readers observe the same Kubernetes resourceVersion.
    writer_a = request("GET", NAME)
    writer_b = copy.deepcopy(writer_a)

    revision = writer_a["metadata"]["resourceVersion"]

    assert (
        writer_b["metadata"]["resourceVersion"] == revision
    )

    print("PASS: Both writers read the same version")

    marker = uuid.uuid4().hex

    writer_a["data"]["state"] = json.dumps({
        "writer": "A",
        "marker": marker,
    })

    updated = request("PUT", NAME, writer_a)

    assert (
        updated["metadata"]["resourceVersion"] != revision
    )

    print("PASS: Writer A committed successfully")

    # Writer B still carries the old resourceVersion.
    writer_b["data"]["state"] = json.dumps({
        "writer": "B",
        "marker": marker,
    })

    try:
        request("PUT", NAME, writer_b)
    except urllib.error.HTTPError as exc:
        assert exc.code == 409, (
            f"Expected conflict 409, received {exc.code}"
        )
        print("PASS: Stale Writer B rejected with HTTP 409")
    else:
        raise AssertionError(
            "UNSAFE: Kubernetes accepted a stale update"
        )

    final = request("GET", NAME)
    state = json.loads(final["data"]["state"])

    assert state == {
        "writer": "A",
        "marker": marker,
    }

    print("PASS: Writer A's state was preserved")

    # The probe must NOT be able to access production state.
    try:
        request("GET", "qwen3-maintenance-state")
    except urllib.error.HTTPError as exc:
        assert exc.code == 403, (
            f"Expected RBAC denial, received {exc.code}"
        )
        print("PASS: Production maintenance state inaccessible")
    else:
        raise AssertionError(
            "UNSAFE: Probe can access production state"
        )

    print()
    print("CAS PROBE PASSED")
    print("Production maintenance resources unchanged")


if __name__ == "__main__":
    main()
