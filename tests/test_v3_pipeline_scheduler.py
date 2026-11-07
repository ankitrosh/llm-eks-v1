import threading
import unittest

from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    SchedulerBusy,
    StageResult,
)


class PipelineSchedulerTests(unittest.TestCase):

    def test_generation_cache_isolation_and_positions(self):
        seen = {}

        def make_stage(rank):
            def run(task, cache):
                key = (rank, task.request_id)

                seen.setdefault(key, []).append(
                    (task.step, task.position, cache)
                )

                self.assertEqual(
                    cache,
                    task.step if task.step else None,
                )

                x = (
                    task.payload[-1]
                    if isinstance(task.payload, tuple)
                    else task.payload
                )

                value = (
                    x * 2 if rank == 0
                    else x + 1
                )

                return StageResult(
                    value,
                    task.step + 1,
                )

            return run

        with PipelineScheduler(
            [make_stage(0), make_stage(1)],
            lambda x: x,
            max_active_requests=2,
        ) as scheduler:

            a = scheduler.submit(
                (1, 2),
                prompt_length=2,
                max_new_tokens=3,
            )

            b = scheduler.submit(
                (3, 4),
                prompt_length=2,
                max_new_tokens=2,
            )

            ra = a.result(timeout=3)
            rb = b.result(timeout=3)

            self.assertEqual(
                ra.tokens, (5, 11, 23)
            )

            self.assertEqual(
                rb.tokens, (9, 19)
            )

            self.assertEqual(
                scheduler.snapshot()["active_requests"],
                0,
            )

            self.assertEqual(
                scheduler.snapshot()[
                    "cache_entries_by_stage"
                ],
                (0, 0),
            )

            for rank in (0, 1):
                self.assertEqual(
                    [
                        p for _, p, _ in seen[
                            (rank, ra.request_id)
                        ]
                    ],
                    [0, 2, 3],
                )

                self.assertEqual(
                    [
                        p for _, p, _ in seen[
                            (rank, rb.request_id)
                        ]
                    ],
                    [0, 2],
                )

    def test_stage_overlap(self):
        stage1_started = threading.Event()
        release_stage1 = threading.Event()
        stage0_second = threading.Event()

        def first(task, cache):
            if task.request_id == 2:
                stage0_second.set()

            return StageResult(task.payload)

        def second(task, cache):
            if task.request_id == 1:
                stage1_started.set()

                if not release_stage1.wait(timeout=3):
                    raise TimeoutError(
                        "Stage 1 was not released"
                    )

            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [first, second],
                lambda x: x,
                max_active_requests=2,
            ) as scheduler:

                a = scheduler.submit(
                    7,
                    prompt_length=1,
                    max_new_tokens=1,
                )

                self.assertTrue(
                    stage1_started.wait(timeout=3)
                )

                b = scheduler.submit(
                    9,
                    prompt_length=1,
                    max_new_tokens=1,
                )

                self.assertTrue(
                    stage0_second.wait(timeout=3),
                    "Stages did not overlap",
                )

                release_stage1.set()

                self.assertEqual(
                    a.result(timeout=3).tokens,
                    (7,),
                )

                self.assertEqual(
                    b.result(timeout=3).tokens,
                    (9,),
                )

        finally:
            release_stage1.set()

    def test_admission_limit(self):
        release = threading.Event()
        entered = threading.Event()

        def block(task, cache):
            entered.set()

            if not release.wait(timeout=3):
                raise TimeoutError("Release missing")

            return StageResult(task.payload)

        try:
            with PipelineScheduler(
                [block],
                lambda x: x,
                max_active_requests=1,
            ) as scheduler:

                a = scheduler.submit(
                    1,
                    prompt_length=1,
                    max_new_tokens=1,
                )

                self.assertTrue(
                    entered.wait(timeout=3)
                )

                with self.assertRaises(SchedulerBusy):
                    scheduler.submit(
                        2,
                        prompt_length=1,
                        max_new_tokens=1,
                    )

                release.set()

                self.assertEqual(
                    a.result(timeout=3).tokens,
                    (1,),
                )

        finally:
            release.set()

    def test_failure_isolated(self):
        def run(task, cache):
            if task.payload == -1:
                raise ValueError("deliberate failure")

            return StageResult(task.payload)

        with PipelineScheduler(
            [run, run],
            lambda x: x,
            max_active_requests=2,
        ) as scheduler:

            bad = scheduler.submit(
                -1,
                prompt_length=1,
                max_new_tokens=1,
            )

            good = scheduler.submit(
                3,
                prompt_length=1,
                max_new_tokens=1,
            )

            with self.assertRaisesRegex(
                ValueError, "deliberate"
            ):
                bad.result(timeout=3)

            self.assertEqual(
                good.result(timeout=3).tokens,
                (3,),
            )

            self.assertEqual(
                scheduler.snapshot()["active_requests"],
                0,
            )

    def test_three_stages_and_early_stop(self):
        def fn(task, cache):
            return StageResult(task.payload + 1)

        with PipelineScheduler(
            [fn, fn, fn],
            lambda x: x,
            stop_fn=lambda x: x >= 9,
        ) as scheduler:

            result = scheduler.submit(
                0,
                prompt_length=1,
                max_new_tokens=10,
            ).result(timeout=3)

            self.assertEqual(
                result.tokens,
                (3, 6, 9),
            )


if __name__ == "__main__":
    unittest.main()
