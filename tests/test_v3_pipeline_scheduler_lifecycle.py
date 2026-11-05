"""Lifecycle regression tests for V3.2."""
import time
import threading
import unittest
from concurrent.futures import CancelledError

from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    SchedulerBusy,
    StageResult,
)


class LifecycleTests(unittest.TestCase):

    def test_cancel_queued_frees_slot(self):
        entered = threading.Event()
        release = threading.Event()

        def stage(task, cache):
            if task.request_id == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [stage], lambda x: x,
                max_active_requests=2
            ) as s:
                a = s.submit(
                    1, prompt_length=1, max_new_tokens=1
                )
                self.assertTrue(entered.wait(3))

                b = s.submit(
                    2, prompt_length=1, max_new_tokens=1
                )
                self.assertTrue(b.cancel())

                with self.assertRaises(CancelledError):
                    b.result()

                c = s.submit(
                    3, prompt_length=1, max_new_tokens=1
                )

                release.set()

                self.assertEqual(
                    a.result(timeout=3).tokens, (1,)
                )
                self.assertEqual(
                    c.result(timeout=3).tokens, (3,)
                )
        finally:
            release.set()

    def test_cancel_running_keeps_cache_until_finish(self):
        entered = threading.Event()
        release = threading.Event()
        seen = []

        def stage0(task, cache):
            seen.append((0, task.request_id))
            if task.request_id == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            return StageResult(task.payload, object())

        def stage1(task, cache):
            seen.append((1, task.request_id))
            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [stage0, stage1],
                lambda x: x,
                max_active_requests=1,
            ) as s:
                a = s.submit(
                    1, prompt_length=1, max_new_tokens=3
                )
                self.assertTrue(entered.wait(3))
                self.assertTrue(a.cancel())

                with self.assertRaises(SchedulerBusy):
                    s.submit(
                        2, prompt_length=1,
                        max_new_tokens=1
                    )

                release.set()

                with self.assertRaises(CancelledError):
                    a.result()

                until = time.monotonic() + 3
                while (
                    s.snapshot()["active_requests"]
                    and time.monotonic() < until
                ):
                    time.sleep(0.01)

                self.assertEqual(
                    s.snapshot()["active_requests"], 0
                )
                self.assertEqual(
                    s.snapshot()["cache_entries_by_stage"],
                    (0, 0),
                )
                self.assertNotIn((1, 1), seen)

                b = s.submit(
                    5, prompt_length=1, max_new_tokens=1
                )
                self.assertEqual(
                    b.result(timeout=3).tokens, (5,)
                )
        finally:
            release.set()

    def test_queued_deadline_expires(self):
        entered = threading.Event()
        release = threading.Event()

        def stage(task, cache):
            if task.request_id == 1:
                entered.set()
                self.assertTrue(release.wait(5))
            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [stage], lambda x: x,
                max_active_requests=2
            ) as s:
                a = s.submit(
                    1, prompt_length=1, max_new_tokens=1
                )
                self.assertTrue(entered.wait(3))

                b = s.submit(
                    2, prompt_length=1,
                    max_new_tokens=1,
                    timeout_s=0.05,
                )

                with self.assertRaisesRegex(
                    TimeoutError, "deadline"
                ):
                    b.result(timeout=3)

                self.assertEqual(
                    s.snapshot()["active_requests"], 1
                )

                release.set()
                self.assertEqual(
                    a.result(timeout=3).tokens, (1,)
                )
        finally:
            release.set()

    def test_running_deadline_keeps_capacity(self):
        entered = threading.Event()
        release = threading.Event()

        def stage(task, cache):
            entered.set()
            self.assertTrue(release.wait(5))
            return StageResult(task.payload, object())

        try:
            with PipelineScheduler(
                [stage], lambda x: x,
                max_active_requests=1
            ) as s:
                a = s.submit(
                    1, prompt_length=1,
                    max_new_tokens=1,
                    timeout_s=0.05,
                )

                self.assertTrue(entered.wait(3))

                with self.assertRaisesRegex(
                    TimeoutError, "deadline"
                ):
                    a.result(timeout=3)

                with self.assertRaises(SchedulerBusy):
                    s.submit(
                        2, prompt_length=1,
                        max_new_tokens=1
                    )

                release.set()
        finally:
            release.set()

    def test_sampler_failure_isolated(self):
        def sample(x):
            if x == -1:
                raise ValueError("sampler failed")
            return x

        with PipelineScheduler(
            [
                lambda task, cache:
                StageResult(task.payload)
            ],
            sample,
            max_active_requests=2,
        ) as s:
            bad = s.submit(
                -1, prompt_length=1, max_new_tokens=1
            )
            good = s.submit(
                4, prompt_length=1, max_new_tokens=1
            )

            with self.assertRaisesRegex(
                ValueError, "sampler failed"
            ):
                bad.result(timeout=3)

            self.assertEqual(
                good.result(timeout=3).tokens, (4,)
            )


if __name__ == "__main__":
    unittest.main()
