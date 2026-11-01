#!/usr/bin/env python3
"""Qwen3 restoration worker.

--audit: read-only, safe against healthy production V1
--execute: requires an authorized RESTORING operation and write RBAC
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import maintenance_restore_audit as audit
from maintenance_restore_plan import plan

NS = "llm"
ORIGINAL_TIMEOUT = 1200
OPERATION_ID = os.environ.get("RESTORE_OPERATION_ID")
OWNER_ID = os.environ.get("RESTORE_OWNER_ID")

CM_MAINTENANCE = (
    f"/api/v1/namespaces/{NS}/configmaps/"
    "qwen3-maintenance-state"
)
CM_RECOVERY = (
    f"/api/v1/namespaces/{NS}/configmaps/"
    "qwen3-recovery-state"
)
STS = f"/apis/apps/v1/namespaces/{NS}/statefulsets/qwen3"


def log(*items):
    print(
        datetime.now(timezone.utc).isoformat(),
        *items,
        flush=True,
    )


def api(method, path, obj=None):
    data = json.dumps(obj).encode() if obj is not None else None
    headers = {"Authorization": f"Bearer {audit.TOKEN}"}

    if data is not None:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        audit.API + path,
        method=method,
        headers=headers,
        data=data,
    )

    with urllib.request.urlopen(
        req, context=audit.CONTEXT, timeout=30
    ) as response:
        return json.load(response)


def maintenance():
    return json.loads(
        audit.get(CM_MAINTENANCE)["data"]["state"]
    )


def baseline():
    cm = audit.namespaced(
        "configmaps", "qwen3-maintenance-storage-baseline"
    )
    return json.loads(cm["data"]["baseline"])


def verify_storage():
    matches, details = audit.check_storage(baseline())
    if not matches:
        raise RuntimeError(
            f"Original EBS identity mismatch: {details}"
        )
    return details


def executor_is_finished(state):
    job, pods_present = audit.inspect_job(state)

    if job is None:
        raise RuntimeError(
            "Executor Job missing; ownership cannot be verified"
        )

    if (
        job["name"] != state.get("job_name")
        or job["uid"] != state.get("job_uid")
    ):
        raise RuntimeError("Executor Job identity mismatch")

    if job["active_pods"] or job["terminating_pods"]:
        raise RuntimeError(
            "Executor still active or terminating"
        )

    if not (job["complete"] or job["failed"]):
        raise RuntimeError(
            "Executor Job has no terminal condition"
        )


def guard():
    """Check authorization again before each mutation."""
    state = maintenance()

    if state.get("phase") != "RESTORING":
        raise RuntimeError("Maintenance phase is not RESTORING")

    if (
        not OPERATION_ID
        or not OWNER_ID
        or state.get("operation_id") != OPERATION_ID
        or state.get("restore_owner") != OWNER_ID
    ):
        raise RuntimeError("Restoration ownership mismatch")

    executor_is_finished(state)
    verify_storage()

    return state


def update_object(path, modify, description):
    """GET/PUT with Kubernetes resourceVersion and conflict retry."""
    for attempt in range(6):
        guard()
        obj = audit.get(path)

        changed = modify(obj)
        if not changed:
            return

        guard()

        try:
            api("PUT", path, obj)
            log("UPDATED:", description)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 409:
                raise

            time.sleep(0.3 * (attempt + 1))

    raise RuntimeError(
        f"Too many update conflicts: {description}"
    )


def set_replicas(kind, name, count):
    if kind == "statefulset":
        path = STS
    else:
        path = (
            f"/apis/apps/v1/namespaces/{NS}/deployments/{name}"
        )

    def modify(obj):
        if obj["spec"].get("replicas", 1) == count:
            return False
        obj["spec"]["replicas"] = count
        return True

    update_object(path, modify, f"{name} replicas={count}")


def set_original_image(original):
    def modify(sts):
        runtime = next(
            c for c in sts["spec"]["template"]["spec"]["containers"]
            if c["name"] == "runtime"
        )

        current = runtime["image"]

        if current not in {
            original,
            maintenance()["candidate_image"],
        }:
            raise RuntimeError(
                f"Unrecognized image; refusing overwrite: {current}"
            )

        if current == original:
            return False

        runtime["image"] = original
        return True

    update_object(STS, modify, "restore original image")


def set_recovery_state(fields):
    def modify(cm):
        state = json.loads(cm["data"]["state"])
        updated = dict(state)
        updated.update(fields)

        if updated == state:
            return False

        cm["data"]["state"] = json.dumps(updated)
        return True

    update_object(
        CM_RECOVERY, modify, "update recovery state"
    )


def wait_for(description, predicate, timeout):
    end = time.monotonic() + timeout

    while time.monotonic() < end:
        guard()

        if predicate():
            log("READY:", description)
            return

        time.sleep(5)

    raise TimeoutError(f"Timed out: {description}")


def selected_pods(selector):
    labels = ",".join(
        f"{k}={v}" for k, v in selector.items()
    )
    encoded = urllib.parse.quote(labels, safe="")

    return audit.get(
        f"/api/v1/namespaces/{NS}/pods"
        f"?labelSelector={encoded}"
    )["items"]


def deployment_stopped(name):
    deployment = audit.application("deployments", name)
    pods = selected_pods(
        deployment["spec"]["selector"]["matchLabels"]
    )

    return (
        deployment["spec"].get("replicas", 1) == 0
        and len(pods) == 0
    )


def workers_absent():
    for rank in (0, 1):
        pod = audit.get(
            f"/api/v1/namespaces/{NS}/pods/qwen3-{rank}",
            optional=True,
        )
        if pod is not None:
            return False

    return True


def workers_ready(original):
    sts = audit.application("statefulsets", "qwen3")
    if sts["spec"]["replicas"] != 2:
        return False

    nodes = set()

    for rank in (0, 1):
        pod = audit.get(
            f"/api/v1/namespaces/{NS}/pods/qwen3-{rank}",
            optional=True,
        )

        if pod is None or not audit.ready(pod):
            return False

        if audit.runtime_image(pod) != original:
            return False

        nodes.add(pod["spec"].get("nodeName"))

    return len(nodes) == 2 and None not in nodes


def recovery_serving():
    deployment = audit.deployment_status("qwen3-recovery")
    cm = audit.namespaced(
        "configmaps", "qwen3-recovery-state"
    )
    state = json.loads(cm["data"]["state"])

    return (
        deployment["ready"]
        and state.get("mode") == "RUNNING"
        and state.get("phase") == "SERVING"
    )


def gateway_ready():
    return audit.deployment_status(
        "qwen3-coordinator"
    )["ready"]


def verify_inference():
    url = (
        "http://qwen3-coordinator.llm.svc."
        "cluster.local:8080/generate"
    )

    payload = json.dumps({
        "prompt": "What is the capital of France?",
        "max_new_tokens": 32,
    }).encode()

    for attempt in range(3):
        guard()

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(
                req, timeout=180
            ) as response:
                output = json.load(response)

            if "paris" not in output.get("text", "").lower():
                raise RuntimeError(
                    f"Unexpected inference response: {output}"
                )

            log("PASS: Live gateway inference")
            return
        except Exception as exc:
            log("Inference attempt failed:", exc)
            if attempt == 2:
                raise
            time.sleep(10)


def observe(state):
    original = state["original_image"]

    sts = audit.application("statefulsets", "qwen3")
    image = audit.runtime_image({
        "spec": sts["spec"]["template"]["spec"]
    })

    pods = [
        audit.get(
            f"/api/v1/namespaces/{NS}/pods/qwen3-{rank}",
            optional=True,
        )
        for rank in (0, 1)
    ]

    images = [
        audit.runtime_image(pod) if pod else None
        for pod in pods
    ]

    nodes = [
        pod["spec"].get("nodeName") if pod else None
        for pod in pods
    ]

    gpu_ready = (
        all(audit.ready(pod) for pod in pods)
        and len(set(nodes)) == 2
        and None not in nodes
    )

    coordinator = audit.deployment_status(
        "qwen3-coordinator"
    )
    recovery = audit.deployment_status(
        "qwen3-recovery"
    )

    recovery_cm = audit.namespaced(
        "configmaps", "qwen3-recovery-state"
    )
    recovery_state = json.loads(
        recovery_cm["data"]["state"]
    )

    storage_matches, _ = audit.check_storage(
        baseline()
    )

    healthy = all([
        image == original,
        sts["spec"]["replicas"] == 2,
        gpu_ready,
        images == [original, original],
        storage_matches,
        coordinator["ready"],
        recovery["ready"],
        recovery_state.get("mode") == "RUNNING",
        recovery_state.get("phase") == "SERVING",
    ])

    job, pods_present = audit.inspect_job(state)

    cluster = {
        "sts_image": image,
        "sts_replicas": sts["spec"]["replicas"],
        "workers_ready": gpu_ready,
        "worker_images": images,
        "storage_matches": storage_matches,
        "coordinator_replicas": coordinator["replicas"],
        "coordinator_ready": coordinator["ready"],
        "recovery_replicas": recovery["replicas"],
        "recovery_ready": recovery["ready"],
        "recovery_mode": recovery_state.get("mode"),
        "recovery_phase": recovery_state.get("phase"),
        "executor_pods_present": pods_present,
        "original_healthy": healthy,
    }

    return cluster, job


def mark_complete():
    def modify(cm):
        state = json.loads(cm["data"]["state"])

        if state.get("phase") != "RESTORING":
            raise RuntimeError("Maintenance phase changed")

        if state.get("operation_id") != OPERATION_ID:
            raise RuntimeError("Operation identity changed")

        if state.get("restore_owner") != OWNER_ID:
            raise RuntimeError("Restore ownership changed")

        state["phase"] = "COMPLETE"
        state["restored_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        state["restore_result"] = "PASS"

        cm["data"]["state"] = json.dumps(state)
        return True

    update_object(
        CM_MAINTENANCE, modify, "mark restoration complete"
    )


def run_execute():
    state = maintenance()
    cluster, job = observe(state)

    if state["phase"] == "IDLE" and cluster["original_healthy"]:
        print(
            "NOOP: Original V1 healthy; no writes performed",
            flush=True,
        )
        return

    guard()

    decision = plan(
        state, job, cluster, now=time.time()
    )

    log("Restoration plan:", json.dumps(decision))

    if decision["decision"] != "RESTORE":
        raise RuntimeError(
            f"Restoration blocked: {decision}"
        )

    original = state["original_image"]

    for step in decision["steps"]:
        guard()
        log("STEP:", step)

        if step == "STOP_GATEWAY_AND_WAIT":
            set_replicas(
                "deployment", "qwen3-coordinator", 0
            )
            wait_for(
                "gateway stopped",
                lambda: deployment_stopped(
                    "qwen3-coordinator"
                ),
                300,
            )

        elif step == "STOP_RECOVERY_AND_WAIT":
            set_replicas(
                "deployment", "qwen3-recovery", 0
            )
            wait_for(
                "recovery stopped",
                lambda: deployment_stopped(
                    "qwen3-recovery"
                ),
                300,
            )

        elif step == "PAUSE_RECOVERY_STATE":
            set_recovery_state({
                "mode": "PAUSED",
                "phase": "STARTING",
                "baseline": None,
            })

        elif step == "SCALE_GPU_TO_ZERO_AND_WAIT":
            set_replicas("statefulset", "qwen3", 0)
            wait_for(
                "GPU worker pods removed",
                workers_absent,
                420,
            )

        elif step == "VERIFY_PERSISTENT_STORAGE":
            verify_storage()

        elif step == "SET_ORIGINAL_IMAGE":
            set_original_image(original)

        elif step == "SCALE_GPU_TO_TWO_AND_WAIT_READY":
            set_replicas("statefulset", "qwen3", 2)
            wait_for(
                "original GPU workers Ready",
                lambda: workers_ready(original),
                ORIGINAL_TIMEOUT,
            )

        elif step == "RESET_RECOVERY_STATE_TO_RUNNING_STARTING":
            recovery_cm = audit.namespaced(
                "configmaps", "qwen3-recovery-state"
            )
            recovery_state = json.loads(
                recovery_cm["data"]["state"]
            )

            # Avoid increasing generation again on an
            # already-in-progress, restarted restoration.
            if not (
                recovery_state.get("mode") == "RUNNING"
                and recovery_state.get("phase") == "STARTING"
                and recovery_state.get("baseline") is None
            ):
                set_recovery_state({
                    "mode": "RUNNING",
                    "phase": "STARTING",
                    "baseline": None,
                    "generation": int(
                        recovery_state.get("generation", 0)
                    ) + 1,
                    "attempt": 0,
                    "started_at": None,
                    "last_error": None,
                })

        elif step == "START_RECOVERY_AND_WAIT_SERVING":
            set_replicas(
                "deployment", "qwen3-recovery", 1
            )
            wait_for(
                "recovery RUNNING / SERVING",
                recovery_serving,
                420,
            )

        elif step == "START_GATEWAY_AND_WAIT_READY":
            set_replicas(
                "deployment", "qwen3-coordinator", 1
            )
            wait_for(
                "gateway available",
                gateway_ready,
                240,
            )

        elif step == "VERIFY_LIVE_INFERENCE":
            verify_inference()

        elif step == "MARK_COMPLETE":
            if not workers_ready(original):
                raise RuntimeError(
                    "Original GPU workers no longer healthy"
                )
            verify_storage()
            if not recovery_serving() or not gateway_ready():
                raise RuntimeError(
                    "CPU services not healthy"
                )
            mark_complete()

        else:
            raise RuntimeError(
                f"Unrecognized restoration step: {step}"
            )

    log("RESTORATION COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("audit", "execute"),
        default="audit",
    )
    args = parser.parse_args()

    if args.mode == "audit":
        print(
            "READ-ONLY MODE — Kubernetes writes disabled",
            flush=True,
        )
        audit.main()
    else:
        run_execute()


if __name__ == "__main__":
    main()
