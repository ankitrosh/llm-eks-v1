"""CPU-only integration tests for the V3.3 watchdog."""

import ast
import copy
import threading
import time
import unittest

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "src/llm_runtime/coordination"
    / "v33_recovery_controller.py"
)


class FakeLifecycle:

    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.replicas = 2
        self.up = True
        self.events = []

    def read_state(self):
        return (
            {"metadata": {"resourceVersion": "1"}},
            copy.deepcopy(self.state),
        )

    def write_state(self, _cm, state):
        self.state = copy.deepcopy(state)
        self.events.append(
            ("persist", state["phase"])
        )

    def scale_set(self, replicas):
        self.replicas = replicas
        self.events.append(
            ("scale", replicas)
        )

    def inspect(self):
        return {
            "desired": self.replicas,
            "healthy": self.replicas == 2 and self.up,
            "signature": [
                ["pod0", 0],
                ["pod1", 0],
            ],
            "workers": [],
        }

    def pods_absent(self):
        return self.replicas == 0


def make_watchdog(state, auto_recover=False):
    fake = FakeLifecycle(state)

    tree = ast.parse(SOURCE.read_text())

    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == "Watchdog"
    )

    isolated = ast.Module(
        body=[node],
        type_ignores=[],
    )

    fake_time = SimpleNamespace(
        monotonic=time.monotonic,
        time=time.time,
        sleep=lambda _: None,
    )

    namespace = {
        "threading": threading,
        "time": fake_time,
        "datetime": datetime,
        "timezone": timezone,
        "AUTO_RECOVER": auto_recover,
        "POLL": 5,
        "STARTUP_GRACE": 360,
        "MAX_ATTEMPTS": 2,
        "DELETE_TIMEOUT": 180,
        "READY_TIMEOUT": 480,
        "read_state": fake.read_state,
        "write_state": fake.write_state,
        "scale_set": fake.scale_set,
        "inspect": fake.inspect,
        "engine": SimpleNamespace(
            pods_absent=fake.pods_absent
        ),
    }

    exec(
        compile(isolated, str(SOURCE), "exec"),
        namespace,
    )

    return fake, namespace["Watchdog"]()


def state(mode="RUNNING", phase="SERVING"):
    return {
        "mode": mode,
        "phase": phase,
        "generation": 1,
        "attempt": 0,
        "recoveries": 0,
        "baseline": None,
        "started_at": time.time(),
        "last_error": None,
    }


class TestV33Recovery(unittest.TestCase):

    def test_legacy_resources_not_targeted(self):
        source = SOURCE.read_text()

        self.assertIn(
            "configmaps/v33-recovery-state",
            source,
        )
        self.assertIn(
            "engine.pods_absent()", source
        )
        self.assertNotIn(
            "statefulsets/{NAME}", source
        )
        self.assertNotIn(
            "qwen3-headless", source
        )

    def test_full_recovery_sequence(self):
        fake, watchdog = make_watchdog(
            state(phase="RECOVERING")
        )

        watchdog.recover()

        self.assertEqual(
            fake.state["phase"], "SERVING"
        )
        self.assertEqual(
            fake.state["recoveries"], 1
        )
        self.assertEqual(
            fake.state["attempt"], 0
        )
        self.assertEqual(
            fake.state["baseline"],
            [["pod0", 0], ["pod1", 0]],
        )

        scaling = [
            event
            for event in fake.events
            if event[0] == "scale"
        ]

        self.assertEqual(
            scaling,
            [("scale", 0), ("scale", 2)],
        )

    def test_paused_does_not_restart(self):
        fake, watchdog = make_watchdog(
            state(
                mode="PAUSED",
                phase="PAUSED",
            )
        )

        watchdog.tick()

        self.assertEqual(
            watchdog.phase, "PAUSED"
        )
        self.assertFalse(
            any(e[0] == "scale" for e in fake.events)
        )

    def test_scale_zero_becomes_pause(self):
        fake, watchdog = make_watchdog(state())
        fake.replicas = 0

        watchdog.tick()

        self.assertEqual(
            fake.state["mode"], "PAUSED"
        )
        self.assertEqual(
            fake.state["phase"], "PAUSED"
        )
        self.assertFalse(
            any(e[0] == "scale" for e in fake.events)
        )

    def test_auto_recovery_disabled(self):
        fake, watchdog = make_watchdog(
            state(),
            auto_recover=False,
        )
        fake.up = False

        watchdog.tick()

        self.assertEqual(
            watchdog.phase, "DEGRADED"
        )
        self.assertFalse(
            any(e[0] == "scale" for e in fake.events)
        )


if __name__ == "__main__":
    unittest.main()
