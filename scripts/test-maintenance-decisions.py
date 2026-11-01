#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src")
)

from maintenance_decisions import decide


def test(name, phase, job, deadline, healthy, expected):
    state = {
        "phase": phase,
        "deadline_unix": deadline,
    }

    result = decide(
        state=state,
        job=job,
        now=1000,
        original_healthy=healthy,
    )

    assert result["action"] == expected, (
        f"{name}: expected {expected}, got {result}"
    )

    print(f"PASS: {name} -> {result['action']}")


test(
    "Normal original V1",
    "IDLE", None, None, True, "NOOP",
)

test(
    "Original unhealthy outside maintenance",
    "IDLE", None, None, False, "ALERT",
)

test(
    "Executor actively testing",
    "TESTING",
    {"active": 1, "failed": False, "complete": False},
    2000, False, "WAIT",
)

test(
    "Deadline expired but executor still active",
    "TESTING",
    {"active": 1, "failed": False, "complete": False},
    900, False, "WAIT",
)

test(
    "Executor failed",
    "TESTING",
    {"active": 0, "failed": True, "complete": False},
    2000, False, "RESTORE",
)

test(
    "Executor finished successfully",
    "TESTING",
    {"active": 0, "failed": False, "complete": True},
    2000, False, "RESTORE",
)

test(
    "Executor disappeared after deadline",
    "TESTING", None, 900, False, "RESTORE",
)

test(
    "Restoration requested and executor stopped",
    "RESTORE_REQUESTED",
    {"active": 0, "failed": False, "complete": True},
    2000, False, "RESTORE",
)

test(
    "Supervisor interrupted during restoration",
    "RESTORING", None, 900, False, "CONTINUE_RESTORE",
)

test(
    "Completed maintenance with healthy original",
    "COMPLETE", None, None, True, "NOOP",
)

print("\nALL MAINTENANCE DECISION TESTS PASSED")
