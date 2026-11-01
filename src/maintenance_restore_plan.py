"""Read-only planner for Qwen3 maintenance restoration.

This module describes actions. It does NOT execute Kubernetes writes.
"""

ACTIVE = {
    "PREPARING",
    "TESTING",
    "RESTORE_REQUESTED",
    "RESTORING",
}


def result(decision, reason, steps=()):
    return {
        "decision": decision,
        "reason": reason,
        "steps": list(steps),
    }


def plan(state, job, cluster, now):
    phase = state.get("phase")
    original = state.get("original_image")
    candidate = state.get("candidate_image")

    if phase == "IDLE":
        if cluster.get("original_healthy"):
            return result(
                "NOOP", "Healthy original deployment"
            )
        return result(
            "ALERT",
            "Unhealthy outside maintenance; no takeover",
        )

    if phase in {"COMPLETE", "FAILED"}:
        return result(
            "NOOP" if cluster.get("original_healthy") else "ALERT",
            "Terminal maintenance state",
        )

    if phase not in ACTIVE:
        return result("BLOCK", "Unknown maintenance phase")

    if (
        not state.get("operation_id")
        or not state.get("job_uid")
        or not state.get("job_name")
        or not state.get("deadline_unix")
    ):
        return result(
            "BLOCK",
            "Missing operation identity, Job binding, or deadline",
        )

    if not (
        isinstance(original, str)
        and isinstance(candidate, str)
        and "@sha256:" in original
        and "@sha256:" in candidate
        and original != candidate
    ):
        return result("BLOCK", "Invalid image identity")

    if cluster.get("sts_image") not in {original, candidate}:
        return result(
            "BLOCK",
            "Unknown StatefulSet image; manual review required",
        )

    if cluster.get("storage_matches") is not True:
        return result(
            "BLOCK",
            "Original PVC/PV/EBS identity not verified",
        )

    if job is None:
        # A missing Job does not prove its containers are stopped.
        if not (
            phase == "RESTORING"
            and cluster.get("executor_pods_present") is False
        ):
            return result(
                "BLOCK",
                "Executor Job absent; termination unverified",
            )
    else:
        if (
            job.get("uid") != state["job_uid"]
            or job.get("name") != state["job_name"]
        ):
            return result(
                "BLOCK",
                "Executor Job does not match operation",
            )

        if (
            job.get("active_pods") != 0
            or job.get("terminating_pods") != 0
        ):
            return result(
                "WAIT",
                "Executor Pods still running or terminating",
            )

        if not (
            job.get("complete")
            or job.get("failed")
        ):
            if now >= float(state["deadline_unix"]):
                return result(
                    "ALERT",
                    "Deadline passed; Job not terminal",
                )
            return result(
                "WAIT",
                "Awaiting executor Job termination",
            )

    if cluster.get("original_healthy") is True:
        if cluster.get("sts_image") != original:
            return result(
                "BLOCK",
                "Inconsistent original-health observation",
            )
        return result(
            "RESTORE",
            "Already restored; no disruptive changes",
            ["VERIFY_LIVE_INFERENCE", "MARK_COMPLETE"],
        )

    original_workers = (
        cluster.get("sts_image") == original
        and cluster.get("sts_replicas") == 2
        and cluster.get("workers_ready") is True
        and cluster.get("worker_images") == [
            original, original
        ]
    )

    recovery_healthy = (
        cluster.get("recovery_ready") is True
        and cluster.get("recovery_mode") == "RUNNING"
        and cluster.get("recovery_phase") == "SERVING"
    )

    steps = []

    if not original_workers:
        if cluster.get("coordinator_replicas", 0) != 0:
            steps.append("STOP_GATEWAY_AND_WAIT")

        if cluster.get("recovery_replicas", 0) != 0:
            steps.append("STOP_RECOVERY_AND_WAIT")

        if cluster.get("recovery_mode") != "PAUSED":
            steps.append("PAUSE_RECOVERY_STATE")

        steps.extend([
            "SCALE_GPU_TO_ZERO_AND_WAIT",
            "VERIFY_PERSISTENT_STORAGE",
            "SET_ORIGINAL_IMAGE",
            "SCALE_GPU_TO_TWO_AND_WAIT_READY",
            "VERIFY_PERSISTENT_STORAGE",
        ])

        recovery_healthy = False

    if not recovery_healthy:
        steps.extend([
            "RESET_RECOVERY_STATE_TO_RUNNING_STARTING",
            "START_RECOVERY_AND_WAIT_SERVING",
        ])

    if cluster.get("coordinator_ready") is not True:
        steps.append("START_GATEWAY_AND_WAIT_READY")

    steps.extend([
        "VERIFY_LIVE_INFERENCE",
        "MARK_COMPLETE",
    ])

    return result(
        "RESTORE",
        "Original V1 restoration plan",
        steps,
    )
