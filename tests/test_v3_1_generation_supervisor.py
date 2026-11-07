import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v3_1_generation_supervisor.py"
)


def launch(code, *, timeout=3, grace=0.25):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--timeout-s",
            str(timeout),
            "--grace-s",
            str(grace),
            "--",
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=12,
    )


def test_success():
    result = launch(
        "print('CHILD_OK', flush=True)"
    )

    assert result.returncode == 0
    assert "CHILD_OK" in result.stdout
    assert "V31_SUPERVISOR_COMPLETE" in result.stdout


def test_failure_is_not_retried():
    result = launch(
        "import sys; sys.exit(7)"
    )

    assert result.returncode == 7
    assert "reason=worker_exit" in result.stdout
    assert "generation_invalid=true" in result.stdout


def test_deadline_is_bounded():
    result = launch(
        "import time; time.sleep(30)",
        timeout=0.5,
    )

    assert result.returncode == 124
    assert "reason=deadline" in result.stdout


def test_sigkill_escalation():
    result = launch(
        "import signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "time.sleep(30)",
        timeout=0.5,
    )

    assert result.returncode == 124
    assert "generation_invalid=true" in result.stdout


def test_parent_sigterm_cleans_up():
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--timeout-s",
            "30",
            "--grace-s",
            "0.25",
            "--",
            sys.executable,
            "-c",
            "import time;time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        first = process.stdout.readline()

        assert "V31_SUPERVISOR_START" in first

        time.sleep(0.2)
        os.kill(process.pid, signal.SIGTERM)

        out, err = process.communicate(timeout=7)

        assert process.returncode == 143, (
            process.returncode, out, err
        )
        assert "reason=signal" in out

    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
