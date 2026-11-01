#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src"),
)

from maintenance_restore_plan import plan

ORIGINAL = "ecr/original@sha256:111"
CANDIDATE = "ecr/candidate@sha256:222"

STATE = {
    "phase": "TESTING",
    "original_image": ORIGINAL,
    "candidate_image": CANDIDATE,
    "operation_id": "op1",
    "job_name": "maintenance-op1",
    "job_uid": "uid-1",
    "deadline_unix": 900,
}

FINISHED_JOB = {
    "uid": "uid-1",
    "name": "maintenance-op1",
    "active_pods": 0,
    "terminating_pods": 0,
    "complete": False,
    "failed": True,
}

CANDIDATE_CLUSTER = {
    "sts_image": CANDIDATE,
    "storage_matches": True,
    "sts_replicas": 2,
    "workers_ready": False,
    "worker_images": [CANDIDATE, CANDIDATE],
    "coordinator_ready": False,
    "coordinator_replicas": 0,
    "recovery_replicas": 0,
    "recovery_ready": False,
    "recovery_mode": "PAUSED",
    "recovery_phase": "STARTING",
    "original_healthy": False,
}


def test(
    name, state, job, cluster,
    expected, required=(), forbidden=()
):
    output = plan(state, job, cluster, now=1000)

    assert output["decision"] == expected, (
        name, output
    )

    for step in required:
        assert step in output["steps"], (
            name, output
        )

    for step in forbidden:
        assert step not in output["steps"], (
            name, output
        )

    print(f"PASS: {name} -> {expected}")


test(
    "Healthy original V1",
    dict(STATE, phase="IDLE"),
    None,
    dict(CANDIDATE_CLUSTER, original_healthy=True),
    "NOOP",
)

test(
    "Unhealthy outside maintenance",
    dict(STATE, phase="IDLE"),
    None,
    CANDIDATE_CLUSTER,
    "ALERT",
)

test(
    "Executor still running",
    STATE,
    dict(
        FINISHED_JOB,
        active_pods=1,
        failed=False,
    ),
    CANDIDATE_CLUSTER,
    "WAIT",
)

test(
    "Wrong executor Job UID",
    STATE,
    dict(FINISHED_JOB, uid="wrong"),
    CANDIDATE_CLUSTER,
    "BLOCK",
)

test(
    "Deadline expired without terminal Job",
    STATE,
    dict(FINISHED_JOB, failed=False),
    CANDIDATE_CLUSTER,
    "ALERT",
)

test(
    "Storage identity mismatch",
    STATE,
    FINISHED_JOB,
    dict(
        CANDIDATE_CLUSTER,
        storage_matches=False,
    ),
    "BLOCK",
)

test(
    "Candidate needs GPU rollback",
    STATE,
    FINISHED_JOB,
    CANDIDATE_CLUSTER,
    "RESTORE",
    required=[
        "SCALE_GPU_TO_ZERO_AND_WAIT",
        "SET_ORIGINAL_IMAGE",
    ],
)

test(
    "Already restored: do not restart GPUs",
    STATE,
    FINISHED_JOB,
    dict(
        CANDIDATE_CLUSTER,
        sts_image=ORIGINAL,
        workers_ready=True,
        worker_images=[ORIGINAL, ORIGINAL],
        original_healthy=True,
    ),
    "RESTORE",
    required=["VERIFY_LIVE_INFERENCE"],
    forbidden=["SCALE_GPU_TO_ZERO_AND_WAIT"],
)

test(
    "GPU restored, CPU services still paused",
    dict(STATE, phase="RESTORING"),
    FINISHED_JOB,
    dict(
        CANDIDATE_CLUSTER,
        sts_image=ORIGINAL,
        workers_ready=True,
        worker_images=[ORIGINAL, ORIGINAL],
    ),
    "RESTORE",
    required=[
        "START_RECOVERY_AND_WAIT_SERVING",
        "START_GATEWAY_AND_WAIT_READY",
    ],
    forbidden=["SCALE_GPU_TO_ZERO_AND_WAIT"],
)

test(
    "Supervisor restarts during restoration",
    dict(STATE, phase="RESTORING"),
    None,
    dict(
        CANDIDATE_CLUSTER,
        executor_pods_present=False,
    ),
    "RESTORE",
)

test(
    "Absent executor without termination proof",
    STATE,
    None,
    CANDIDATE_CLUSTER,
    "BLOCK",
)

print("\nALL RESTORE-PLANNER TESTS PASSED")
