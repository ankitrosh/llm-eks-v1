import os
import signal
from pathlib import Path
from datetime import datetime, timezone

targets = []

for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue

    pid = int(entry.name)

    if pid == os.getpid():
        continue

    try:
        raw = (entry / "cmdline").read_bytes()
        args = [
            part.decode(errors="replace")
            for part in raw.split(b"\0")
            if part
        ]
    except OSError:
        continue

    # Select the actual model worker, not its torchrun launcher.
    if (
        "/app/qwen3_serve.py" in args
        and not any("torchrun" in arg for arg in args)
    ):
        targets.append(pid)

if len(targets) != 1:
    raise SystemExit(
        f"Expected exactly one model worker; found {targets}. "
        "No processes were killed."
    )

pid = targets[0]

print(
    f"Sending SIGKILL to Rank 1 worker PID={pid}",
    flush=True,
)

os.kill(pid, signal.SIGKILL)

print(
    "SIGKILL sent at",
    datetime.now(timezone.utc).isoformat(),
    flush=True,
)
