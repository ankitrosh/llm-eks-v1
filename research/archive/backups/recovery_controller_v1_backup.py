"""V1 two-rank NCCL recovery watchdog. CPU-only, Python standard library."""

import json
import os
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

NS = "llm"
NAME = "qwen3"
AUTO_RECOVER = os.getenv("AUTO_RECOVER", "false").lower() == "true"

SA = Path("/var/run/secrets/kubernetes.io/serviceaccount")
API = (
    f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:"
    f"{os.getenv('KUBERNETES_SERVICE_PORT_HTTPS', '443')}"
)
TLS = ssl.create_default_context(cafile=str(SA / "ca.crt"))

SCALE = f"/apis/apps/v1/namespaces/{NS}/statefulsets/{NAME}/scale"
URLS = [
    f"http://qwen3-{i}.qwen3-headless.llm.svc.cluster.local:8000/readyz"
    for i in range(2)
]

POLL = 5
STARTUP_GRACE = 360
RECOVERY_TIMEOUT = 480
MAX_ATTEMPTS = 2


def kube(path, method="GET", patch=None, allow_404=False):
    data = None if patch is None else json.dumps(patch).encode()
    headers = {
        "Authorization": "Bearer " + (SA / "token").read_text().strip()
    }

    if patch is not None:
        headers["Content-Type"] = "application/merge-patch+json"

    request = Request(API + path, headers=headers, data=data, method=method)

    try:
        with urlopen(request, timeout=10, context=TLS) as response:
            return json.load(response)
    except HTTPError as exc:
        if allow_404 and exc.code == 404:
            return None
        raise RuntimeError(
            f"Kubernetes API {method} {path}: HTTP {exc.code}"
        ) from exc


def pod(rank):
    name = f"{NAME}-{rank}"
    info = kube(
        f"/api/v1/namespaces/{NS}/pods/{name}",
        allow_404=True,
    )

    if info is None:
        return {
            "name": name,
            "exists": False,
            "ready": False,
            "signature": None,
        }

    containers = info.get("status", {}).get("containerStatuses", [])
    runtime = next(
        (c for c in containers if c["name"] == "runtime"), {}
    )

    return {
        "name": name,
        "exists": True,
        "ready": bool(runtime.get("ready", False)),
        "signature": (
            info["metadata"]["uid"],
            runtime.get("restartCount", 0),
        ),
    }


def inspect():
    desired = kube(SCALE)["spec"]["replicas"]
    workers = [pod(0), pod(1)]

    for i, worker in enumerate(workers):
        worker["http_ready"] = False

        if worker["exists"]:
            try:
                with urlopen(URLS[i], timeout=2) as response:
                    worker["http_ready"] = response.status == 200
            except (HTTPError, URLError, OSError, TimeoutError):
                pass

    return {
        "desired": desired,
        "workers": workers,
        "healthy": (
            desired == 2
            and all(w["ready"] and w["http_ready"] for w in workers)
        ),
        "signature": tuple(w["signature"] for w in workers),
    }


class Watchdog:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "STARTING"
        self.reason = "Monitoring startup"
        self.last = None
        self.baseline = None
        self.failures = 0
        self.started = time.monotonic()
        self.generation = 0
        self.recoveries = 0
        self.last_error = None

    def status(self):
        with self.lock:
            snapshot = self.last or {}

            return {
                "phase": self.phase,
                "reason": self.reason,
                "auto_recover": AUTO_RECOVER,
                "generation": self.generation,
                "recoveries_completed": self.recoveries,
                "failed_checks": self.failures,
                "desired_replicas": snapshot.get("desired"),
                "workers": snapshot.get("workers", []),
                "last_error": self.last_error,
            }

    def run(self):
        while True:
            try:
                snapshot = inspect()
            except Exception as exc:
                with self.lock:
                    self.phase = "DEGRADED"
                    self.reason = "Inspection failed; no restart without API"
                    self.last_error = str(exc)

                print("Inspection failed:", repr(exc), flush=True)
                time.sleep(POLL)
                continue

            trigger = False

            with self.lock:
                self.last = snapshot

                if snapshot["desired"] != 2:
                    self.phase = "PAUSED"
                    self.reason = "Workers scaled away; recovery disabled"
                    self.baseline = None
                    self.failures = 0
                    self.started = time.monotonic()

                elif self.phase == "FAILED":
                    pass  # Manual intervention after exhausted attempts.

                else:
                    if self.phase == "PAUSED":
                        self.phase = "STARTING"
                        self.started = time.monotonic()
                        self.failures = 0

                    same_group = (
                        self.baseline is None
                        or snapshot["signature"] == self.baseline
                    )

                    if snapshot["healthy"] and same_group:
                        self.phase = "SERVING"
                        self.reason = "Both workers ready; same group"
                        self.baseline = snapshot["signature"]
                        self.failures = 0
                        self.last_error = None

                    elif (
                        self.phase == "STARTING"
                        and time.monotonic() - self.started < STARTUP_GRACE
                    ):
                        self.reason = "Waiting for model and NCCL warmup"

                    else:
                        self.phase = "DEGRADED"
                        self.failures += 1
                        self.reason = (
                            "Unhealthy worker or changed pod identity"
                        )
                        trigger = AUTO_RECOVER and self.failures >= 3

            if trigger:
                self.recover()

            time.sleep(POLL)

    def recover(self):
        with self.lock:
            self.phase = "RECOVERING"
            self.generation += 1
            self.reason = "Restarting the complete NCCL group"

        print("Recovery started", flush=True)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                # Do not override an intentional pause.
                if kube(SCALE)["spec"]["replicas"] != 2:
                    with self.lock:
                        self.phase = "PAUSED"
                        self.reason = "External scale change"
                    return

                # Terminate BOTH ranks.
                kube(SCALE, "PATCH", {"spec": {"replicas": 0}})

                deadline = time.monotonic() + 150

                while time.monotonic() < deadline:
                    if not pod(0)["exists"] and not pod(1)["exists"]:
                        break
                    time.sleep(3)
                else:
                    raise TimeoutError(
                        "Old workers did not terminate; refusing overlap"
                    )

                # Start a completely fresh NCCL group.
                kube(SCALE, "PATCH", {"spec": {"replicas": 2}})

                deadline = time.monotonic() + RECOVERY_TIMEOUT

                while time.monotonic() < deadline:
                    current = inspect()

                    if current["desired"] != 2:
                        with self.lock:
                            self.phase = "PAUSED"
                            self.reason = "Operator changed replica count"
                        return

                    if current["healthy"]:
                        with self.lock:
                            self.phase = "SERVING"
                            self.reason = "Fresh NCCL group ready"
                            self.baseline = current["signature"]
                            self.last = current
                            self.failures = 0
                            self.recoveries += 1
                            self.last_error = None

                        print(
                            "Recovery succeeded on attempt",
                            attempt,
                            flush=True,
                        )
                        return

                    time.sleep(5)

                raise TimeoutError("Workers did not recover in time")

            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)

                print(
                    f"Recovery attempt {attempt} failed:",
                    repr(exc),
                    flush=True,
                )

                if attempt < MAX_ATTEMPTS:
                    time.sleep(10)

        with self.lock:
            self.phase = "FAILED"
            self.reason = "Recovery exhausted; operator required"

        print("Recovery exhausted", flush=True)


watchdog = Watchdog()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            status, payload = 200, {"status": "alive"}
        elif self.path == "/status":
            status, payload = 200, watchdog.status()
        else:
            status, payload = 404, {"error": "not found"}

        data = json.dumps(payload).encode()

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


threading.Thread(target=watchdog.run, daemon=True).start()

print(
    f"Recovery watchdog running; AUTO_RECOVER={AUTO_RECOVER}",
    flush=True,
)

ThreadingHTTPServer(("0.0.0.0", 8081), Handler).serve_forever()
