"""Verify the frozen V3 scheduler can run 3+ local stages."""

import threading
import unittest
from collections import defaultdict

from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    StageResult,
)


class MultiStageSchedulerTests(unittest.TestCase):

    def run_pipeline(self, num_stages):
        history = defaultdict(list)
        lock = threading.Lock()

        def make_stage(rank):
            def execute(task, cache):
                expected_step = (
                    0 if cache is None else cache
                )

                if expected_step != task.step:
                    raise ValueError(
                        f"Rank {rank}: wrong cache step "
                        f"{expected_step} != {task.step}"
                    )

                with lock:
                    history[rank].append(
                        (task.request_id, task.step)
                    )

                return StageResult(
                    payload=rank,
                    cache_state=task.step + 1,
                )

            return execute

        stages = [
            make_stage(rank)
            for rank in range(num_stages)
        ]

        with PipelineScheduler(
            stages,
            lambda output: 7,
            max_active_requests=4,
        ) as scheduler:
            futures = [
                scheduler.submit(
                    [10 + request_id, 20],
                    prompt_length=2,
                    max_new_tokens=4,
                    timeout_s=20,
                )
                for request_id in range(4)
            ]

            results = [
                future.result(timeout=25)
                for future in futures
            ]

            metrics = scheduler.metrics()

        self.assertEqual(
            metrics["completed_requests"], 4
        )
        self.assertEqual(
            metrics["generated_tokens"], 16
        )
        self.assertEqual(
            metrics["failed_requests"], 0
        )
        self.assertTrue(metrics["healthy"])

        for result in results:
            self.assertEqual(
                tuple(result.tokens),
                (7, 7, 7, 7),
            )

        for rank in range(num_stages):
            records = history[rank]

            self.assertEqual(len(records), 16)

            request_ids = {
                rid for rid, _ in records
            }
            self.assertEqual(len(request_ids), 4)

            for rid in request_ids:
                steps = [
                    step for request, step in records
                    if request == rid
                ]
                self.assertEqual(
                    steps, [0, 1, 2, 3]
                )

        return metrics

    def test_three_stage_pipeline(self):
        metrics = self.run_pipeline(3)
        self.assertEqual(len(metrics["stages"]), 3)

    def test_four_stage_pipeline(self):
        metrics = self.run_pipeline(4)
        self.assertEqual(len(metrics["stages"]), 4)


if __name__ == "__main__":
    unittest.main()
