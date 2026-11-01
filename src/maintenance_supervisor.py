#!/usr/bin/env python3
"""CPU maintenance supervisor — monitoring mode only."""

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

NS = "llm"

API = (
    f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:"
    f"{os.environ.get('KUBERNETES_SERVICE_PORT', '443')}"
)

SA = "/var/run/secrets/kubernetes.io/serviceaccount"

TOKEN = open(f"{SA}/token").read().strip()
CONTEXT = ssl.create_default_context(cafile=f"{SA}/ca.crt")

CM_PATH = (
    f"/api/v1/namespaces/{NS}/configmaps/"
    "qwen3-maintenance-state"
)

STS_PATH = (
    f"/apis/apps/v1/namespaces/{NS}/statefulsets/qwen3"
)

ACTIVE_PHASES = {
    "PREPARING",
    "TESTING",
    "RESTORING",
}


def fetch(path):
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    with urllib.request.urlopen(
        req, context=CONTEXT, timeout=15
    ) as response:
        return json.load(response)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def inspect():
    cm = fetch(CM_PATH)
    state = json.loads(cm["data"]["state"])

    sts = fetch(STS_PATH)

    image = next(
        c["image"]
        for c in sts["spec"]["template"]["spec"]["containers"]
        if c["name"] == "runtime"
    )

    phase = state.get("phase")
    deadline = state.get("deadline_unix")

    result = {
        "observed_at": timestamp(),
        "phase": phase,
        "operation_id": state.get("operation_id"),
        "desired_replicas": sts["spec"].get("replicas", 1),
        "ready_replicas": sts.get("status", {}).get(
            "readyReplicas", 0
        ),
        "actual_image": image,
        "original_image": state.get("original_image"),
        "deadline_unix": deadline,
        "alert": None,
    }

    if phase == "IDLE":
        if image != state.get("original_image"):
            result["alert"] = (
                "IDLE_IMAGE_MISMATCH"
            )

        elif result["desired_replicas"] != 2:
            result["alert"] = (
                "IDLE_REPLICA_MISMATCH"
            )

    elif phase in ACTIVE_PHASES:
        if deadline is None:
            result["alert"] = (
                "ACTIVE_OPERATION_WITHOUT_DEADLINE"
            )

        elif time.time() >= float(deadline):
            result["alert"] = (
                "MAINTENANCE_DEADLINE_EXPIRED"
            )

    elif phase not in {"COMPLETE", "FAILED"}:
        result["alert"] = "UNKNOWN_MAINTENANCE_PHASE"

    return result


def main():
    print(
        "GPU maintenance supervisor started",
        flush=True,
    )
    print(
        "Mode: MONITOR ONLY — no Kubernetes writes",
        flush=True,
    )

    previous = None
    last_log = 0

    while True:
        try:
            current = inspect()

            signature = json.dumps({
                k: v for k, v in current.items()
                if k != "observed_at"
            }, sort_keys=True)

            # Log state changes and periodic heartbeats.
            if (
                signature != previous
                or time.monotonic() - last_log >= 120
            ):
                print(json.dumps(current), flush=True)
                previous = signature
                last_log = time.monotonic()

            if current["alert"]:
                print(
                    "WARNING:",
                    current["alert"],
                    "Automatic restoration is NOT enabled.",
                    flush=True,
                )

        except Exception as exc:
            print(
                "Supervisor error:",
                type(exc).__name__,
                str(exc),
                flush=True,
            )

        time.sleep(10)


if __name__ == "__main__":
    main()
