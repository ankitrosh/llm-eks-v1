#!/usr/bin/env python3
"""Read-only validation of the Qwen3 restoration planner."""

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

from maintenance_restore_plan import plan

NS = "llm"

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


def get(path, optional=False):
    request = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    try:
        with urllib.request.urlopen(
            request, context=CONTEXT, timeout=15
        ) as response:
            return json.load(response)

    except urllib.error.HTTPError as exc:
        if optional and exc.code == 404:
            return None
        raise


def namespaced(kind, name):
    return get(f"/api/v1/namespaces/{NS}/{kind}/{name}")


def application(kind, name):
    return get(
        f"/apis/apps/v1/namespaces/{NS}/{kind}/{name}"
    )


def ready(pod):
    if not pod or pod.get("metadata", {}).get("deletionTimestamp"):
        return False

    condition_ready = any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in pod.get("status", {}).get("conditions", [])
    )

    runtime_ready = any(
        c["name"] == "runtime" and c.get("ready") is True
        for c in pod.get("status", {}).get(
            "containerStatuses", []
        )
    )

    return condition_ready and runtime_ready


def runtime_image(obj):
    containers = obj["spec"]["containers"]

    return next(
        c["image"] for c in containers
        if c["name"] == "runtime"
    )


def deployment_status(name):
    deployment = application("deployments", name)

    return {
        "replicas": deployment["spec"].get("replicas", 1),
        "ready": (
            deployment["spec"].get("replicas", 1) == 1
            and deployment.get("status", {}).get(
                "availableReplicas", 0
            ) >= 1
        ),
    }


def check_storage(baseline):
    details = {}
    matches = True

    for claim, expected in baseline.items():
        pvc = namespaced("persistentvolumeclaims", claim)

        pv_name = pvc["spec"].get("volumeName")
        pv = get(f"/api/v1/persistentvolumes/{pv_name}")

        csi = pv.get("spec", {}).get("csi", {})
        volume = csi.get("volumeHandle")

        claim_ref = pv.get("spec", {}).get("claimRef", {})

        correct = (
            pvc["status"].get("phase") == "Bound"
            and pv.get("status", {}).get("phase") == "Bound"
            and pv_name == expected["pv"]
            and volume == expected["ebs_volume"]
            and claim_ref.get("namespace") == NS
            and claim_ref.get("name") == claim
        )

        details[claim] = {
            "expected_ebs": expected["ebs_volume"],
            "actual_ebs": volume,
            "pv": pv_name,
            "matches": correct,
        }

        matches = matches and correct

    return matches, details


def inspect_job(state):
    name = state.get("job_name")

    if not name:
        return None, None

    job = get(
        f"/apis/batch/v1/namespaces/{NS}/jobs/{name}",
        optional=True,
    )

    selector = urllib.parse.quote(
        f"batch.kubernetes.io/job-name={name}",
        safe="",
    )

    pod_list = get(
        f"/api/v1/namespaces/{NS}/pods"
        f"?labelSelector={selector}"
    )["items"]

    pods_present = len(pod_list) > 0

    if job is None:
        return None, pods_present

    uid = job["metadata"]["uid"]

    # Exclude unrelated Pods, but fail closed if ownership differs.
    owner_match = all(
        any(
            ref.get("uid") == uid
            and ref.get("kind") == "Job"
            for ref in pod["metadata"].get(
                "ownerReferences", []
            )
        )
        for pod in pod_list
    )

    if not owner_match:
        raise RuntimeError(
            "Job Pod ownership mismatch"
        )

    active = sum(
        pod.get("status", {}).get("phase")
        not in ("Succeeded", "Failed")
        for pod in pod_list
    )

    terminating = sum(
        bool(pod["metadata"].get("deletionTimestamp"))
        for pod in pod_list
    )

    conditions = {
        c["type"]
        for c in job.get("status", {}).get("conditions", [])
        if c.get("status") == "True"
    }

    return {
        "name": name,
        "uid": uid,
        "active_pods": active,
        "terminating_pods": terminating,
        "complete": "Complete" in conditions,
        "failed": "Failed" in conditions,
    }, pods_present


def main():
    print("=== LIVE RESTORATION AUDIT ===", flush=True)

    maintenance = json.loads(
        namespaced(
            "configmaps", "qwen3-maintenance-state"
        )["data"]["state"]
    )

    baseline = json.loads(
        namespaced(
            "configmaps", "qwen3-maintenance-storage-baseline"
        )["data"]["baseline"]
    )

    sts = application("statefulsets", "qwen3")

    image = runtime_image(
        {"spec": sts["spec"]["template"]["spec"]}
    )

    worker_pods = [
        get(
            f"/api/v1/namespaces/{NS}/pods/qwen3-{rank}",
            optional=True,
        )
        for rank in (0, 1)
    ]

    workers_ready = all(
        ready(pod) for pod in worker_pods
    )

    worker_images = [
        runtime_image(pod) if pod else None
        for pod in worker_pods
    ]

    nodes = [
        pod["spec"].get("nodeName")
        for pod in worker_pods
        if pod
    ]

    different_nodes = (
        len(nodes) == 2
        and all(nodes)
        and nodes[0] != nodes[1]
    )

    coordinator = deployment_status("qwen3-coordinator")
    recovery = deployment_status("qwen3-recovery")

    recovery_state = json.loads(
        namespaced(
            "configmaps", "qwen3-recovery-state"
        )["data"]["state"]
    )

    storage_matches, storage_details = check_storage(
        baseline
    )

    job, executor_pods_present = inspect_job(
        maintenance
    )

    original = maintenance["original_image"]

    original_healthy = all([
        image == original,
        sts["spec"]["replicas"] == 2,
        workers_ready,
        worker_images == [original, original],
        different_nodes,
        storage_matches,
        coordinator["ready"],
        recovery["ready"],
        recovery_state.get("mode") == "RUNNING",
        recovery_state.get("phase") == "SERVING",
    ])

    cluster = {
        "sts_image": image,
        "sts_replicas": sts["spec"]["replicas"],
        "workers_ready": workers_ready and different_nodes,
        "worker_images": worker_images,
        "storage_matches": storage_matches,
        "coordinator_replicas": coordinator["replicas"],
        "coordinator_ready": coordinator["ready"],
        "recovery_replicas": recovery["replicas"],
        "recovery_ready": recovery["ready"],
        "recovery_mode": recovery_state.get("mode"),
        "recovery_phase": recovery_state.get("phase"),
        "executor_pods_present": executor_pods_present,
        "original_healthy": original_healthy,
    }

    import time

    decision = plan(
        state=maintenance,
        job=job,
        cluster=cluster,
        now=time.time(),
    )

    report = {
        "maintenance_phase": maintenance["phase"],
        "statefulset_image": image,
        "statefulset_replicas": sts["spec"]["replicas"],
        "workers_ready": workers_ready,
        "different_gpu_nodes": different_nodes,
        "storage_matches": storage_matches,
        "storage": storage_details,
        "coordinator_ready": coordinator["ready"],
        "recovery_ready": recovery["ready"],
        "recovery_mode": recovery_state.get("mode"),
        "recovery_phase": recovery_state.get("phase"),
        "original_control_plane_healthy": original_healthy,
        "planner": decision,
    }

    print(json.dumps(report, indent=2), flush=True)

    if maintenance["phase"] != "IDLE":
        raise RuntimeError(
            "Expected IDLE maintenance state for this audit"
        )

    if not original_healthy:
        raise RuntimeError(
            "Original deployment health checks failed"
        )

    if not storage_matches:
        raise RuntimeError(
            "Original persistent storage identity mismatch"
        )

    if decision["decision"] != "NOOP":
        raise RuntimeError(
            f"Unexpected restoration decision: {decision}"
        )

    print(
        "\nPASS: Live restoration planner audit",
        flush=True,
    )
    print(
        "NO GPU OR APPLICATION RESOURCES MODIFIED",
        flush=True,
    )


if __name__ == "__main__":
    main()
