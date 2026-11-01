"""Pure decision logic for GPU maintenance supervision.

No Kubernetes access or resource mutations in this module.
"""

ACTIVE = {"PREPARING", "TESTING"}
TERMINAL = {"COMPLETE", "FAILED"}


def decide(state, job, now, original_healthy):
    """
    job:
      None, or a dictionary with:
      active: number of active Job pods
      complete: bool
      failed: bool

    Returns a dictionary containing action and reason.
    """
    phase = state.get("phase", "UNKNOWN")
    deadline = state.get("deadline_unix")

    if phase == "IDLE":
        if original_healthy:
            return {
                "action": "NOOP",
                "reason": "Original deployment healthy",
            }
        return {
            "action": "ALERT",
            "reason": "Original deployment unhealthy outside maintenance",
        }

    if phase in TERMINAL:
        return {
            "action": "NOOP" if original_healthy else "ALERT",
            "reason": "Maintenance operation already terminal",
        }

    if phase not in ACTIVE | {"RESTORE_REQUESTED", "RESTORING"}:
        return {
            "action": "ALERT",
            "reason": f"Unknown maintenance phase: {phase}",
        }

    active = int(job.get("active", 0)) if job else 0
    failed = bool(job and job.get("failed"))
    complete = bool(job and job.get("complete"))

    expired = (
        deadline is not None
        and now >= float(deadline)
    )

    if phase == "RESTORING":
        if active:
            return {
                "action": "WAIT",
                "reason": "Executor still active during restoration",
            }
        return {
            "action": "CONTINUE_RESTORE",
            "reason": "Resume idempotent restoration",
        }

    if phase == "RESTORE_REQUESTED":
        if active:
            return {
                "action": "WAIT",
                "reason": "Waiting for executor to exit",
            }
        return {
            "action": "RESTORE",
            "reason": "Executor requested restoration",
        }

    if active:
        return {
            "action": "WAIT",
            "reason": (
                "Deadline expired; awaiting Job termination"
                if expired
                else "Executor running"
            ),
        }

    if failed:
        return {
            "action": "RESTORE",
            "reason": "Executor Job failed",
        }

    if complete:
        return {
            "action": "RESTORE",
            "reason": "Executor finished; restore original V1",
        }

    if expired:
        return {
            "action": "RESTORE",
            "reason": "Maintenance deadline expired",
        }

    return {
        "action": "WAIT",
        "reason": "Maintenance operation awaiting executor",
    }
