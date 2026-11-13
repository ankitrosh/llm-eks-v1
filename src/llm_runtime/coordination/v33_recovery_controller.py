#!/usr/bin/env python3
"""Durable V3.3 recovery controller for vLLM PP=2."""

import json
import os
import ssl
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Both source files must be mounted into
# the same directory in the V3.3 pod.
from vllm_engine_controller import KubernetesVLLMEngineController

NS = "llm"
AUTO_RECOVER = os.getenv("AUTO_RECOVER", "false").lower() == "true"

SA = Path("/var/run/secrets/kubernetes.io/serviceaccount")
API = (
    f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:"
    f"{os.getenv('KUBERNETES_SERVICE_PORT_HTTPS', '443')}"
)
TLS = ssl.create_default_context(cafile=str(SA / "ca.crt"))

STATE = f"/api/v1/namespaces/{NS}/configmaps/v33-recovery-state"


POLL = 5
STARTUP_GRACE = 360
MAX_ATTEMPTS = 2
DELETE_TIMEOUT = 180
READY_TIMEOUT = 480


def kube(path, method="GET", payload=None, content_type=None,
         missing_ok=False):
    headers = {
        "Authorization": "Bearer " + (SA / "token").read_text().strip()
    }
    body = None

    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = content_type or "application/json"

    request = Request(
        API + path, data=body, headers=headers, method=method
    )

    try:
        with urlopen(request, timeout=12, context=TLS) as response:
            return json.load(response)
    except HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        detail = exc.read(500).decode(errors="replace")
        raise RuntimeError(
            f"Kubernetes API {method} {path}: {exc.code} {detail}"
        ) from exc


def read_state():
    cm = kube(STATE)
    return cm, json.loads(cm["data"]["state"])


def write_state(cm, state):
    # Optimistic concurrency protection.
    patch = [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": cm["metadata"]["resourceVersion"],
        },
        {
            "op": "replace",
            "path": "/data/state",
            "value": json.dumps(state),
        },
    ]
    return kube(
        STATE, "PATCH", patch,
        "application/json-patch+json"
    )


def vllm_api_healthy():
    url = os.getenv(
        "V33_VLLM_HEALTH_URL",
        "http://v33-vllm-rank0.llm.svc.cluster.local:8000/health",
    )

    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200

    except (HTTPError, URLError, TimeoutError, OSError):
        return False


engine = KubernetesVLLMEngineController(
    kube=kube,
    api_healthy=vllm_api_healthy,
    namespace=NS,
    name="v33-vllm",
)


def scale_get():
    return engine.scale_get()


def scale_set(replicas):
    return engine.scale_set(replicas)


def inspect():
    return engine.inspect()


class Watchdog:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "STARTING"
        self.reason = "Waiting for first inspection"
        self.snapshot = {}
        self.state = {}
        self.failures = 0
        self.started = time.monotonic()
        self.error = None

    def display(self, phase, reason, snapshot=None,
                state=None, error=None):
        with self.lock:
            self.phase = phase
            self.reason = reason

            if snapshot is not None:
                self.snapshot = snapshot
            if state is not None:
                self.state = dict(state)

            self.error = error

    def status(self):
        with self.lock:
            state = self.state
            snapshot = self.snapshot

            return {
                "phase": self.phase,
                "reason": self.reason,
                "auto_recover": AUTO_RECOVER,
                "mode": state.get("mode", "UNKNOWN"),
                "generation": state.get("generation"),
                "recoveries_completed": state.get("recoveries", 0),
                "recovery_attempt": state.get("attempt", 0),
                "last_recovery_seconds": state.get(
                    "last_recovery_seconds"
                ),
                "failed_checks": self.failures,
                "desired_replicas": snapshot.get("desired"),
                "workers": snapshot.get("workers", []),
                "last_error": self.error or state.get("last_error"),
            }

    def check_intent(self):
        _, state = read_state()
        if (
            state["mode"] != "RUNNING"
            or state["phase"] != "RECOVERING"
        ):
            raise InterruptedError("Recovery intent changed")
        return state

    def recover(self):
        # Safe to invoke again after controller-process replacement.
        while True:
            cm, state = read_state()

            if (
                state["mode"] != "RUNNING"
                or state["phase"] != "RECOVERING"
            ):
                return

            attempt = int(state.get("attempt", 0))

            if attempt == 0:
                state["attempt"] = 1
                write_state(cm, state)
                attempt = 1

            self.display(
                "RECOVERING",
                f"Restarting vLLM engine, attempt {attempt}",
                state=state,
            )

            try:
                self.check_intent()

                scale_set(0)

                deadline = time.monotonic() + DELETE_TIMEOUT

                while time.monotonic() < deadline:
                    self.check_intent()

                    if engine.pods_absent():
                        break

                    time.sleep(3)
                else:
                    raise TimeoutError(
                        "Old pods not deleted; refusing overlap"
                    )

                self.check_intent()
                scale_set(2)

                deadline = time.monotonic() + READY_TIMEOUT

                while time.monotonic() < deadline:
                    self.check_intent()
                    snapshot = inspect()

                    self.display(
                        "RECOVERING",
                        "Waiting for fresh vLLM engine",
                        snapshot=snapshot,
                    )

                    if snapshot["healthy"]:
                        cm, latest = read_state()

                        if (
                            latest["mode"] != "RUNNING"
                            or latest["phase"] != "RECOVERING"
                        ):
                            raise InterruptedError(
                                "Intent changed before commit"
                            )

                        duration = round(
                            time.time() - latest["started_at"], 2
                        )

                        latest.update(
                            phase="SERVING",
                            baseline=snapshot["signature"],
                            recoveries=latest["recoveries"] + 1,
                            attempt=0,
                            started_at=None,
                            last_error=None,
                            last_recovery_seconds=duration,
                            last_completed_at=datetime.now(
                                timezone.utc
                            ).isoformat(),
                        )

                        write_state(cm, latest)

                        self.failures = 0
                        self.display(
                            "SERVING",
                            "Fresh vLLM engine ready",
                            snapshot=snapshot,
                            state=latest,
                        )

                        print(
                            f"Recovery generation "
                            f"{latest['generation']} succeeded "
                            f"in {duration}s",
                            flush=True,
                        )
                        return

                    time.sleep(5)

                raise TimeoutError(
                    "New vLLM engine did not become ready"
                )

            except InterruptedError as exc:
                print(str(exc), flush=True)
                return

            except Exception as exc:
                print(
                    f"Recovery attempt {attempt} failed: {exc}",
                    flush=True,
                )

                try:
                    cm, latest = read_state()

                    if (
                        latest["mode"] != "RUNNING"
                        or latest["phase"] != "RECOVERING"
                    ):
                        return

                    latest["last_error"] = str(exc)

                    if attempt >= MAX_ATTEMPTS:
                        latest["phase"] = "FAILED"
                        write_state(cm, latest)

                        self.display(
                            "FAILED",
                            "Recovery attempts exhausted",
                            state=latest,
                            error=str(exc),
                        )
                        return

                    latest["attempt"] = attempt + 1
                    write_state(cm, latest)

                except Exception as write_exc:
                    self.display(
                        "DEGRADED",
                        "Unable to persist recovery failure",
                        error=str(write_exc),
                    )
                    return

                time.sleep(10)

    def tick(self):
        cm, state = read_state()

        if state["mode"] == "PAUSED":
            self.failures = 0
            self.display(
                "PAUSED", "Operator pause intent", state=state
            )
            return

        if state["phase"] == "FAILED":
            self.display(
                "FAILED", "Operator intervention required",
                state=state,
            )
            return

        if state["phase"] == "RECOVERING":
            self.display(
                "RECOVERING",
                "Resuming durable recovery",
                state=state,
            )

            if AUTO_RECOVER:
                self.recover()
            return

        snapshot = inspect()
        baseline = state.get("baseline")

        if snapshot["desired"] == 0:
            # No durable recovery intent: treat scale-to-zero as pause.
            state.update(
                mode="PAUSED",
                phase="PAUSED",
                baseline=None,
            )
            write_state(cm, state)

            self.display(
                "PAUSED",
                "Scaled to zero outside recovery",
                snapshot=snapshot,
                state=state,
            )
            return

        same_group = (
            baseline is None
            or baseline == snapshot["signature"]
        )

        if snapshot["healthy"] and same_group:
            self.failures = 0
            self.started = time.monotonic()

            if (
                state["phase"] != "SERVING"
                or baseline is None
            ):
                state.update(
                    phase="SERVING",
                    baseline=snapshot["signature"],
                )
                write_state(cm, state)

            self.display(
                "SERVING",
                "Both ranks ready; identities match",
                snapshot=snapshot,
                state=state,
            )
            return

        if (
            state["phase"] == "STARTING"
            and baseline is None
            and time.monotonic() - self.started < STARTUP_GRACE
        ):
            self.display(
                "STARTING",
                "Waiting for model/NCCL warmup",
                snapshot=snapshot,
                state=state,
            )
            return

        self.failures += 1

        self.display(
            "DEGRADED",
            "Worker unavailable or identity changed",
            snapshot=snapshot,
            state=state,
        )

        if AUTO_RECOVER and self.failures >= 3:
            state.update(
                phase="RECOVERING",
                generation=state["generation"] + 1,
                attempt=0,
                started_at=time.time(),
                last_error=None,
                recovery_reason="Worker failure or identity change",
            )

            # Persist intent BEFORE scaling workers to zero.
            write_state(cm, state)

            print(
                f"Recovery generation {state['generation']} started",
                flush=True,
            )

            self.recover()

    def run(self):
        while True:
            try:
                self.tick()
            except Exception as exc:
                self.display(
                    "DEGRADED",
                    "Controller inspection/API error",
                    error=str(exc),
                )
                print(
                    f"Watchdog error: {exc!r}",
                    flush=True,
                )

            time.sleep(POLL)


watchdog = Watchdog()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            code, payload = 200, {"status": "alive"}
        elif self.path == "/status":
            code, payload = 200, watchdog.status()
        else:
            code, payload = 404, {"error": "not found"}

        data = json.dumps(payload).encode()

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


threading.Thread(
    target=watchdog.run, daemon=True
).start()

print(
    f"Durable watchdog running; AUTO_RECOVER={AUTO_RECOVER}",
    flush=True,
)

ThreadingHTTPServer(
    ("0.0.0.0", 8081), Handler
).serve_forever()
