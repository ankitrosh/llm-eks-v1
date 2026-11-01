#!/usr/bin/env python3
"""Persistent, read-only GPU worker observer."""

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
    "qwen3-gpu-diagnostics"
)

MAX_EVENTS = 120


def now():
    return datetime.now(timezone.utc).isoformat()


def request(method, path, payload=None):
    data = None
    headers = {"Authorization": f"Bearer {TOKEN}"}

    if payload is not None:
        data = json.dumps(payload).encode()
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


def pod_snapshot(rank):
    name = f"qwen3-{rank}"

    try:
        pod = request(
            "GET", f"/api/v1/namespaces/{NS}/pods/{name}"
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"name": name, "exists": False}
        raise

    containers = []

    for status in pod.get("status", {}).get(
        "containerStatuses", []
    ):
        containers.append({
            "name": status.get("name"),
            "ready": status.get("ready"),
            "restartCount": status.get("restartCount"),
            "state": status.get("state"),
            "lastState": status.get("lastState"),
            "containerID": status.get("containerID"),
        })

    return {
        "name": name,
        "exists": True,
        "uid": pod["metadata"]["uid"],
        "phase": pod.get("status", {}).get("phase"),
        "node": pod["spec"].get("nodeName"),
        "images": {
            c["name"]: c["image"]
            for c in pod["spec"].get("containers", [])
        },
        "containers": containers,
    }


def save_event(event):
    # resourceVersion makes updates conditional, preventing lost
    # writes if another writer modifies the ConfigMap.
    for attempt in range(5):
        cm = request("GET", CM_PATH)

        history = json.loads(
            cm.get("data", {}).get("history", "[]")
        )

        history.append(event)
        history = history[-MAX_EVENTS:]

        cm.setdefault("data", {})["history"] = json.dumps(
            history, separators=(",", ":")
        )

        try:
            request("PUT", CM_PATH, cm)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 409:
                raise
            time.sleep(0.2 * (attempt + 1))

    raise RuntimeError("Unable to persist diagnostics")


def main():
    print("GPU observer started", flush=True)
    previous = {}

    while True:
        try:
            for rank in (0, 1):
                snapshot = pod_snapshot(rank)

                # Record changes, not repeated identical samples.
                signature = json.dumps(
                    snapshot, sort_keys=True
                )

                if previous.get(rank) != signature:
                    event = {
                        "observed_at": now(),
                        "rank": rank,
                        "snapshot": snapshot,
                    }

                    save_event(event)
                    previous[rank] = signature

                    print(
                        json.dumps(event),
                        flush=True,
                    )

        except Exception as exc:
            print(
                f"Observer error: {type(exc).__name__}: {exc}",
                flush=True,
            )

        time.sleep(10)


if __name__ == "__main__":
    main()
