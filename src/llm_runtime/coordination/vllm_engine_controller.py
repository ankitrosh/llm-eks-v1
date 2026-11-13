"""Kubernetes lifecycle adapter for a two-node vLLM engine."""

from typing import Protocol
from urllib.parse import urlencode


class EngineController(Protocol):
    def inspect(self) -> dict: ...
    def stop(self) -> None: ...
    def start(self) -> None: ...
    def pods_absent(self) -> bool: ...


class KubernetesVLLMEngineController:
    """Manage two Deployments as one logical engine.

    kube: existing Kubernetes API helper.
    api_healthy: callable checking rank-0 vLLM /health.
    """

    def __init__(
        self,
        kube,
        api_healthy,
        namespace="llm",
        name="v33-vllm",
    ):
        self.kube = kube
        self.api_healthy = api_healthy
        self.namespace = namespace
        self.name = name

    def _deployment_path(self, rank):
        return (
            f"/apis/apps/v1/namespaces/{self.namespace}"
            f"/deployments/{self.name}-rank{rank}/scale"
        )

    def _pods_path(self, rank):
        selector = f"app={self.name},rank={rank}"
        return (
            f"/api/v1/namespaces/{self.namespace}/pods?"
            + urlencode({"labelSelector": selector})
        )

    def _desired_replicas(self):
        return tuple(
            int(
                self.kube(
                    self._deployment_path(rank)
                )["spec"]["replicas"]
            )
            for rank in (0, 1)
        )

    def scale_get(self):
        sizes = self._desired_replicas()

        if sizes == (0, 0):
            return 0

        if sizes == (1, 1):
            return 2

        # An inconsistent state is NOT an operator pause.
        return None

    def scale_set(self, total_replicas):
        if (
            type(total_replicas) is not int
            or total_replicas not in (0, 2)
        ):
            raise ValueError(
                "vLLM PP=2 expects total replicas 0 or 2"
            )

        per_rank = total_replicas // 2

        for rank in (0, 1):
            self.kube(
                self._deployment_path(rank),
                method="PATCH",
                payload={
                    "spec": {"replicas": per_rank}
                },
                content_type=(
                    "application/merge-patch+json"
                ),
            )

    def stop(self):
        self.scale_set(0)

    def start(self):
        self.scale_set(2)

    def _worker(self, rank):
        result = self.kube(
            self._pods_path(rank)
        )

        items = result.get("items", [])

        if len(items) != 1:
            return {
                "rank": rank,
                "exists": bool(items),
                "ready": False,
                "signature": None,
                "pod_count": len(items),
            }

        pod = items[0]
        metadata = pod.get("metadata", {})
        status = pod.get("status", {})

        containers = status.get(
            "containerStatuses", []
        )

        runtime = next(
            (
                c for c in containers
                if c.get("name") == "vllm"
            ),
            {},
        )

        ready = (
            status.get("phase") == "Running"
            and not metadata.get("deletionTimestamp")
            and bool(runtime.get("ready"))
        )

        return {
            "rank": rank,
            "exists": True,
            "ready": ready,
            "pod_count": 1,
            "signature": [
                metadata.get("uid"),
                runtime.get("restartCount", 0),
            ],
        }

    def pods_absent(self):
        return all(
            not self.kube(
                self._pods_path(rank)
            ).get("items")
            for rank in (0, 1)
        )

    def inspect(self):
        sizes = self._desired_replicas()

        workers = [
            self._worker(rank)
            for rank in (0, 1)
        ]

        k8s_ready = (
            sizes == (1, 1)
            and all(w["ready"] for w in workers)
        )

        # Rank 1 is headless. Only rank 0 has
        # an HTTP inference/health endpoint.
        api_ready = (
            bool(self.api_healthy())
            if k8s_ready
            else False
        )

        return {
            "desired": (
                2 if sizes == (1, 1)
                else 0 if sizes == (0, 0)
                else None
            ),
            "desired_replicas": sizes,
            "workers": workers,
            "signature": [
                w["signature"]
                for w in workers
            ],
            "api_ready": api_ready,
            "healthy": k8s_ready and api_ready,
        }
