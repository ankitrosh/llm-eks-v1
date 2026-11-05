import threading
import time
import unittest

from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError,
    PipelineScheduler,
    PipelineUnhealthyError,
    StageResult,
)


class SchedulerReliabilityTests(unittest.TestCase):

    def test_fatal_stage_fails_all_requests(self):
        entered = threading.Event()
        release = threading.Event()

        def stage0(task, cache):
            if task.payload == "kill":
                raise FatalStageError("lost NCCL peer")
            return StageResult(task.payload)

        def stage1(task, cache):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Stage blocked")
            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [stage0, stage1],
                lambda x: x,
                max_active_requests=3,
            ) as scheduler:

                a = scheduler.submit(
                    "ok", prompt_length=1, max_new_tokens=1
                )
                self.assertTrue(entered.wait(3))

                b = scheduler.submit(
                    "kill", prompt_length=1, max_new_tokens=1
                )

                for future in (a, b):
                    with self.assertRaises(PipelineUnhealthyError):
                        future.result(timeout=3)

                self.assertFalse(scheduler.metrics()["healthy"])
                self.assertEqual(
                    scheduler.metrics()["failed_requests"], 2
                )

                with self.assertRaises(PipelineUnhealthyError):
                    scheduler.submit(
                        "new", prompt_length=1, max_new_tokens=1
                    )

                release.set()
        finally:
            release.set()

    def test_request_error_does_not_kill_pipeline(self):
        def stage(task, cache):
            if task.payload == -1:
                raise ValueError("bad request")
            return StageResult(task.payload)

        with PipelineScheduler(
            [stage], lambda x: x
        ) as scheduler:

            bad = scheduler.submit(
                -1, prompt_length=1, max_new_tokens=1
            )

            with self.assertRaises(ValueError):
                bad.result(timeout=3)

            good = scheduler.submit(
                7, prompt_length=1, max_new_tokens=1
            )

            self.assertEqual(
                good.result(timeout=3).tokens, (7,)
            )

            metrics = scheduler.metrics()
            self.assertTrue(metrics["healthy"])
            self.assertEqual(metrics["failed_requests"], 1)
            self.assertEqual(metrics["completed_requests"], 1)

    def test_scheduler_metrics(self):
        def stage(task, cache):
            time.sleep(0.002)
            return StageResult(
                task.payload, (cache or 0) + 1
            )

        with PipelineScheduler(
            [stage, stage], lambda x: x
        ) as scheduler:

            futures = [
                scheduler.submit(
                    token,
                    prompt_length=1,
                    max_new_tokens=2,
                )
                for token in (8, 9)
            ]

            for future in futures:
                future.result(timeout=3)

            metrics = scheduler.metrics()

            self.assertEqual(metrics["submitted_requests"], 2)
            self.assertEqual(metrics["completed_requests"], 2)
            self.assertEqual(metrics["generated_tokens"], 4)
            self.assertEqual(metrics["first_tokens"], 2)

            self.assertGreater(metrics["tokens_per_second"], 0)
            self.assertGreater(metrics["avg_latency_ms"], 0)
            self.assertGreater(metrics["avg_ttft_ms"], 0)

            self.assertEqual(
                tuple(s["tasks"] for s in metrics["stages"]),
                (4, 4),
            )

    def test_fifo_prefill_decode_fairness(self):
        starts = []
        entered = threading.Event()
        release = threading.Event()

        def stage(task, cache):
            starts.append((task.request_id, task.step))

            if task.request_id == 1 and task.step == 0:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Stage blocked")

            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [stage], lambda x: x
            ) as scheduler:

                a = scheduler.submit(
                    "a", prompt_length=1, max_new_tokens=2
                )
                self.assertTrue(entered.wait(3))

                b = scheduler.submit(
                    "b", prompt_length=1, max_new_tokens=1
                )

                release.set()
                a.result(timeout=3)
                b.result(timeout=3)

                self.assertEqual(
                    starts,
                    [(1, 0), (2, 0), (1, 1)],
                )
        finally:
            release.set()

    def test_cancellation_and_timeout_metrics(self):
        entered = threading.Event()
        release = threading.Event()

        def stage(task, cache):
            if task.request_id == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Stage blocked")
            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [stage], lambda x: x
            ) as scheduler:

                a = scheduler.submit(
                    1, prompt_length=1, max_new_tokens=1
                )
                self.assertTrue(entered.wait(3))

                b = scheduler.submit(
                    2, prompt_length=1, max_new_tokens=1
                )
                self.assertTrue(b.cancel())

                c = scheduler.submit(
                    3,
                    prompt_length=1,
                    max_new_tokens=1,
                    timeout_s=0.02,
                )

                with self.assertRaises(TimeoutError):
                    c.result(timeout=3)

                metrics = scheduler.metrics()

                self.assertEqual(
                    metrics["cancelled_requests"], 1
                )
                self.assertEqual(
                    metrics["timed_out_requests"], 1
                )

                release.set()
                a.result(timeout=3)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
