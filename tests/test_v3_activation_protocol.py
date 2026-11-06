import threading
import time
import unittest

from dataclasses import replace

from llm_runtime.v3_activation_protocol import (
    OrderedActivationLane,
    ActivationProtocolError,
    ActivationBackpressure,
)
from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    StageResult,
)


class ActivationProtocolTests(unittest.TestCase):

    def test_interleaved_requests_and_overlap(self):
        with OrderedActivationLane(max_pending=4) as lane:

            def stage0(task, cache):
                time.sleep(0.02)

                x = (
                    task.payload[-1]
                    if isinstance(task.payload, tuple)
                    else task.payload
                )

                ticket = lane.publish(
                    request_id=task.request_id,
                    step=task.step,
                    position=task.position,
                    payload=x * 2,
                    shape=(1, 1, 64),
                    dtype="float32",
                )

                return StageResult(
                    ticket, (cache or 0) + 1
                )

            def stage1(task, cache):
                value = lane.receive(
                    task.payload,
                    request_id=task.request_id,
                    step=task.step,
                    position=task.position,
                )

                time.sleep(0.02)

                return StageResult(
                    value + 1, (cache or 0) + 1
                )

            with PipelineScheduler(
                [stage0, stage1],
                lambda x: x,
                max_active_requests=3,
                on_request_end=lane.discard_request,
            ) as scheduler:

                futures = [
                    scheduler.submit(
                        (1, n),
                        prompt_length=2,
                        max_new_tokens=3,
                    )
                    for n in (2, 4, 6)
                ]

                actual = [
                    future.result(timeout=4).tokens
                    for future in futures
                ]

                self.assertEqual(
                    actual,
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

            stats = lane.stats()

            self.assertEqual(stats["published"], 9)
            self.assertEqual(stats["transferred"], 9)
            self.assertEqual(stats["pending"], 0)
            self.assertEqual(stats["unconsumed"], 0)

    def test_cancellation_skips_staged_tensor(self):
        stage1_blocked = threading.Event()
        release = threading.Event()

        with OrderedActivationLane(max_pending=3) as lane:

            def stage0(task, cache):
                ticket = lane.publish(
                    request_id=task.request_id,
                    step=task.step,
                    position=task.position,
                    payload=task.payload,
                    shape=(1, 1, 8),
                    dtype="float32",
                )
                return StageResult(ticket)

            def stage1(task, cache):
                value = lane.receive(
                    task.payload,
                    request_id=task.request_id,
                    step=task.step,
                    position=task.position,
                )

                if task.request_id == 1:
                    stage1_blocked.set()

                    if not release.wait(timeout=3):
                        raise TimeoutError("Release missing")

                return StageResult(value)

            try:
                with PipelineScheduler(
                    [stage0, stage1],
                    lambda x: x,
                    max_active_requests=3,
                    on_request_end=lane.discard_request,
                ) as scheduler:

                    a = scheduler.submit(
                        7, prompt_length=1,
                        max_new_tokens=1,
                    )

                    self.assertTrue(
                        stage1_blocked.wait(3)
                    )

                    b = scheduler.submit(
                        8, prompt_length=1,
                        max_new_tokens=1,
                    )

                    deadline = time.monotonic() + 3

                    while lane.stats()["published"] < 2:
                        self.assertLess(
                            time.monotonic(), deadline
                        )
                        time.sleep(0.005)

                    self.assertTrue(b.cancel())

                    self.assertEqual(
                        lane.stats()["discarded"], 1
                    )

                    c = scheduler.submit(
                        9, prompt_length=1,
                        max_new_tokens=1,
                    )

                    release.set()

                    self.assertEqual(
                        a.result(timeout=3).tokens, (7,)
                    )
                    self.assertEqual(
                        c.result(timeout=3).tokens, (9,)
                    )

                self.assertEqual(
                    lane.stats()["pending"], 0
                )
                self.assertEqual(
                    lane.stats()["transferred"], 2
                )

            finally:
                release.set()

    def test_out_of_order_receive_is_fatal(self):
        with OrderedActivationLane() as lane:
            t1 = lane.publish(
                request_id=1, step=0, position=0,
                payload=1, shape=(1, 1, 2),
                dtype="float32",
            )
            t2 = lane.publish(
                request_id=2, step=0, position=0,
                payload=2, shape=(1, 1, 2),
                dtype="float32",
            )

            with self.assertRaisesRegex(
                ActivationProtocolError, "order"
            ):
                lane.receive(
                    t2, request_id=2,
                    step=0, position=0,
                )

            with self.assertRaises(
                ActivationProtocolError
            ):
                lane.receive(
                    t1, request_id=1,
                    step=0, position=0,
                )

    def test_metadata_mismatch_is_fatal(self):
        with OrderedActivationLane() as lane:
            ticket = lane.publish(
                request_id=1, step=0, position=0,
                payload=1, shape=(1, 1, 2),
                dtype="float32",
            )

            with self.assertRaises(
                ActivationProtocolError
            ):
                lane.receive(
                    replace(ticket, step=1),
                    request_id=1,
                    step=1,
                    position=0,
                )

    def test_capacity_then_discard(self):
        with OrderedActivationLane(max_pending=1) as lane:
            lane.publish(
                request_id=1, step=0, position=0,
                payload=1, shape=(1, 1, 1),
                dtype="float32",
            )

            with self.assertRaises(
                ActivationBackpressure
            ):
                lane.publish(
                    request_id=2, step=0, position=0,
                    payload=2, shape=(1, 1, 1),
                    dtype="float32",
                )

            lane.discard_request(1)

            ticket = lane.publish(
                request_id=2, step=0, position=0,
                payload=2, shape=(1, 1, 1),
                dtype="float32",
            )

            self.assertEqual(
                lane.receive(
                    ticket, request_id=2,
                    step=0, position=0,
                ),
                2,
            )

            self.assertEqual(
                lane.stats()["discarded"], 1
            )


if __name__ == "__main__":
    unittest.main()
