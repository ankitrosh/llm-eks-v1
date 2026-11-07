"""Bounded, fail-stop supervisor for a V3.1 distributed worker group.

Never retries a failed generation or reuses its KV caches.
Multi-node Kubernetes group recovery is handled separately.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import uuid


class StopRequested(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Received signal {signum}")


def terminate_group(child, grace_s):
    """Terminate the child process group, escalating if necessary."""

    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass

    # The leader can exit while descendants remain.
    # Attempt group-wide cleanup even after leader exit.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    child.wait(timeout=3)


def run_supervised(command, *, timeout_s, grace_s):
    generation = uuid.uuid4().hex[:12]

    environment = os.environ.copy()
    environment["V31_GENERATION"] = generation

    def on_sigterm(signum, _frame):
        raise StopRequested(signum)

    previous_handler = signal.signal(
        signal.SIGTERM, on_sigterm
    )

    child = None
    completed = False
    status = 1

    try:
        print(
            f"V31_SUPERVISOR_START "
            f"generation={generation}",
            flush=True,
        )

        child = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
        )

        try:
            returncode = child.wait(timeout=timeout_s)

        except subprocess.TimeoutExpired:
            status = 124

            print(
                f"V31_SUPERVISOR_FAILED "
                f"generation={generation} "
                "reason=deadline "
                "generation_invalid=true",
                flush=True,
            )

        else:
            if returncode == 0:
                completed = True
                status = 0

                print(
                    f"V31_SUPERVISOR_COMPLETE "
                    f"generation={generation}",
                    flush=True,
                )
            else:
                status = (
                    returncode
                    if returncode > 0
                    else 128 - returncode
                )

                print(
                    f"V31_SUPERVISOR_FAILED "
                    f"generation={generation} "
                    f"reason=worker_exit "
                    f"code={status} "
                    "generation_invalid=true",
                    flush=True,
                )

    except KeyboardInterrupt:
        status = 130

        print(
            f"V31_SUPERVISOR_FAILED "
            f"generation={generation} "
            "reason=interrupt generation_invalid=true",
            flush=True,
        )

    except StopRequested as exc:
        status = 128 + exc.signum

        print(
            f"V31_SUPERVISOR_FAILED "
            f"generation={generation} "
            f"reason=signal code={status} "
            "generation_invalid=true",
            flush=True,
        )

    except OSError as exc:
        status = 127

        print(
            f"V31_SUPERVISOR_FAILED "
            f"generation={generation} "
            f"reason=launch_error "
            f"error={type(exc).__name__} "
            "generation_invalid=true",
            flush=True,
        )

    finally:
        # Avoid re-entering the handler during cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        try:
            if child is not None and not completed:
                terminate_group(child, grace_s)
        finally:
            signal.signal(
                signal.SIGTERM, previous_handler
            )

    return status


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timeout-s", required=True, type=float
    )
    parser.add_argument(
        "--grace-s", default=2.0, type=float
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    if (
        not math.isfinite(args.timeout_s)
        or not math.isfinite(args.grace_s)
        or not 0 < args.timeout_s <= 86400
        or not 0 < args.grace_s <= 60
    ):
        parser.error("Invalid supervisor time limits")

    command = args.command

    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("Provide a command after --")

    return run_supervised(
        command,
        timeout_s=args.timeout_s,
        grace_s=args.grace_s,
    )


if __name__ == "__main__":
    sys.exit(main())
