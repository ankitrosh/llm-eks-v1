"""CPU-only tests for the vLLM engine controller."""

import unittest
from urllib.parse import parse_qs, urlparse

from llm_runtime.coordination.vllm_engine_controller import (
    KubernetesVLLMEngineController,
)


class FakeCluster:
    def __init__(self):
        self.sizes = [1, 1]
        self.pods = [
            [self._pod(0)],
            [self._pod(1)],
        ]
        self.calls = []
        self.api_calls = 0
        self.up = True

    @staticmethod
    def _pod(rank):
        return {
            "metadata": {
                "uid": f"pod-uid-{rank}"
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{
                    "name": "vllm",
                    "ready": True,
                    "restartCount": 0,
                }],
            },
        }

    def kube(
        self,
        path,
        method="GET",
        payload=None,
        content_type=None,
    ):
        self.calls.append(
            (method, path, payload)
        )

        if path.endswith("/scale"):
            rank = int(
                path.split("-rank")[1].split("/")[0]
            )

            if method == "PATCH":
                self.sizes[rank] = (
                    payload["spec"]["replicas"]
                )

            return {
                "spec": {
                    "replicas": self.sizes[rank]
                }
            }

        parsed = urlparse(path)

        selector = parse_qs(
            parsed.query
        )["labelSelector"][0]

        rank = int(
            selector.split("rank=")[1]
        )

        return {
            "items": self.pods[rank]
        }

    def health(self):
        self.api_calls += 1
        return self.up


class TestEngineAdapter(unittest.TestCase):

    def setUp(self):
        self.cluster = FakeCluster()

        self.adapter = (
            KubernetesVLLMEngineController(
                self.cluster.kube,
                self.cluster.health,
            )
        )

    def test_both_workers_and_api_healthy(self):
        state = self.adapter.inspect()

        self.assertTrue(state["healthy"])
        self.assertEqual(state["desired"], 2)
        self.assertEqual(
            self.cluster.api_calls, 1
        )

    def test_rank1_is_headless(self):
        self.adapter.inspect()

        self.assertEqual(
            self.cluster.api_calls, 1
        )

        self.assertEqual(
            len(self.adapter.inspect()["workers"]),
            2,
        )

    def test_missing_worker_fails_closed(self):
        self.cluster.pods[1] = []

        self.assertFalse(
            self.adapter.inspect()["healthy"]
        )

        self.assertEqual(
            self.cluster.api_calls, 0
        )

    def test_api_failure_fails_closed(self):
        self.cluster.up = False

        self.assertFalse(
            self.adapter.inspect()["healthy"]
        )

    def test_two_deployments_scaled_as_unit(self):
        self.adapter.stop()

        self.assertEqual(
            self.cluster.sizes, [0, 0]
        )

        self.assertEqual(
            self.adapter.scale_get(), 0
        )

        self.cluster.pods = [[], []]

        self.assertTrue(
            self.adapter.pods_absent()
        )

        self.adapter.start()

        self.assertEqual(
            self.cluster.sizes, [1, 1]
        )

        self.assertEqual(
            self.adapter.scale_get(), 2
        )

    def test_mixed_replica_counts_fail_closed(self):
        self.cluster.sizes = [1, 0]

        self.assertIsNone(
            self.adapter.scale_get()
        )

        self.assertFalse(
            self.adapter.inspect()["healthy"]
        )

    def test_invalid_replica_count_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.scale_set(1)

    def test_replacing_pod_changes_signature(self):
        previous = (
            self.adapter.inspect()["signature"]
        )

        self.cluster.pods[1][0][
            "metadata"
        ]["uid"] = "fresh-uid"

        self.assertNotEqual(
            previous,
            self.adapter.inspect()["signature"],
        )


if __name__ == "__main__":
    unittest.main()
