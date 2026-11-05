import os
import time
import unittest
from concurrent.futures import CancelledError

from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError,
    PipelineScheduler,
    PipelineUnhealthyError,
    StageResult,
)
from llm_runtime.v3_process_transport import (
    ProcessStageTransport,
    RemoteStageError,
)


class ArithmeticFactory:
    """Picklable factory; stage creation happens in child."""

    def __init__(self, delay_s=0.025):
        self.delay_s = delay_s

    def __call__(self, rank):

        def run(task, cache):
            time.sleep(self.delay_s)

            if task.payload == -777:
                os._exit(17)

            if task.payload == -999:
                raise FatalStageError(
                    "Simulated NCCL disconnect"
                )

            if task.payload == -1:
                raise ValueError("Bad prompt")

            if task.step == 0:
                assert cache is None
            else:
                assert cache == (
                    rank,
                    task.request_id,
                    task.step - 1,
                )

            value = (
                task.payload[-1]
                if isinstance(task.payload, tuple)
                else task.payload
            )

            value = (
                value * 2
                if rank == 0
                else value + 1
            )

            return StageResult(
                value,
                (rank, task.request_id, task.step),
            )

        return run


class ProcessTransportTests(unittest.TestCase):

    def test_requests_caches_and_overlap(self):
        with ProcessStageTransport(
            ArithmeticFactory()
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                lambda x: x,
                max_active_requests=3,
                on_request_end=transport.release_request,
            ) as scheduler:

                futures = [
                    scheduler.submit(
                        (1, n),
                        prompt_length=2,
                        max_new_tokens=3,
                    )
                    for n in (2, 4, 6)
                ]

                results = [
                    f.result(timeout=8)
                    for f in futures
                ]

                self.assertEqual(
                    [r.tokens for r in results],
                    [
                        (5, 11, 23),
                        (9, 19, 39),
                        (13, 27, 55),
                    ],
                )

                events = scheduler.snapshot()["events"]

                starts = {
                    (e.stage, e.request_id, e.step): e.at
                    for e in events
                    if e.event == "start"
                }

                finishes = {
                    (e.stage, e.request_id, e.step): e.at
                    for e in events
                    if e.event == "finish"
                }

                self.assertLess(
                    starts[(0, 2, 0)],
                    finishes[(1, 1, 0)],
                )

                self.assertLess(
                    starts[(1, 1, 0)],
                    finishes[(0, 2, 0)],
                )

            stats = transport.stats()

            self.assertEqual(
                tuple(s["cache_entries"] for s in stats),
                (0, 0),
            )

            self.assertEqual(
                tuple(s["runs"] for s in stats),
                (9, 9),
            )

            self.assertNotEqual(
                stats[0]["pid"],
                stats[1]["pid"],
            )

            for rank in (0, 1):
                self.assertNotEqual(
                    stats[rank]["pid"],
                    os.getpid(),
                )

                for rid in (1, 2, 3):
                    calls = [
                        row
                        for row in stats[rank]["trace"]
                        if row["request_id"] == rid
                    ]

                    self.assertEqual(
                        [row["step"] for row in calls],
                        [0, 1, 2],
                    )

                    self.assertEqual(
                        [row["cache_present"] for row in calls],
                        [False, True, True],
                    )

    def test_cancellation_releases_caches(self):
        with ProcessStageTransport(
            ArithmeticFactory(delay_s=0.08)
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                lambda x: x,
                on_request_end=transport.release_request,
            ) as scheduler:

                f = scheduler.submit(
                    (1, 2),
                    prompt_length=2,
                    max_new_tokens=10,
                )

                deadline = time.monotonic() + 3

                while not any(
                    scheduler.snapshot()["busy_by_stage"]
                ):
                    self.assertLess(
                        time.monotonic(),
                        deadline,
                    )
                    time.sleep(0.005)

                self.assertTrue(f.cancel())

                with self.assertRaises(CancelledError):
                    f.result()

            self.assertEqual(
                tuple(
                    s["cache_entries"]
                    for s in transport.stats()
                ),
                (0, 0),
            )

    def test_request_error_isolated(self):
        with ProcessStageTransport(
            ArithmeticFactory(delay_s=0)
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                lambda x: x,
                on_request_end=transport.release_request,
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

                with self.assertRaises(RemoteStageError):
                    bad.result(timeout=5)

                self.assertEqual(
                    good.result(timeout=5).tokens,
                    (7,),
                )

                self.assertTrue(
                    scheduler.metrics()["healthy"]
                )

    def test_fatal_error_stops_pipeline(self):
        with ProcessStageTransport(
            ArithmeticFactory(delay_s=0)
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                lambda x: x,
                on_request_end=transport.release_request,
            ) as scheduler:

                bad = scheduler.submit(
                    -999,
                    prompt_length=1,
                    max_new_tokens=1,
                )

                with self.assertRaises(PipelineUnhealthyError):
                    bad.result(timeout=5)

                self.assertFalse(
                    scheduler.metrics()["healthy"]
                )

    def test_process_crash_is_fatal(self):
        with ProcessStageTransport(
            ArithmeticFactory(delay_s=0)
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                lambda x: x,
                on_request_end=transport.release_request,
            ) as scheduler:

                bad = scheduler.submit(
                    -777,
                    prompt_length=1,
                    max_new_tokens=1,
                )

                with self.assertRaises(PipelineUnhealthyError):
                    bad.result(timeout=5)

                self.assertFalse(
                    scheduler.metrics()["healthy"]
                )


if __name__ == "__main__":
    unittest.main()
