"""Minimal user-defined V3 model provider.

CPU-only example showing the stage factory and cache contract.
Not intended for the NCCL GPU runner.
"""

from llm_runtime.v3_pipeline_scheduler import StageResult


class ToyStage:
    def __init__(self, rank, offset):
        self.rank = rank
        self.offset = offset

    def __call__(self, task, cache):
        if type(task.payload) is not int:
            raise TypeError("ToyStage expects an integer payload")

        if cache is None:
            cache = {
                "request_id": task.request_id,
                "calls": 0,
            }

        if cache["request_id"] != task.request_id:
            raise ValueError("Cache belongs to another request")

        updated_cache = {
            "request_id": task.request_id,
            "calls": cache["calls"] + 1,
        }

        return StageResult(
            task.payload + self.offset,
            updated_cache,
        )


class ToyStageFactory:
    def __init__(self, offsets=(1, 10)):
        if (
            not isinstance(offsets, (list, tuple))
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError("Expected two integer offsets")

        self.offsets = tuple(offsets)

    def __call__(self, rank):
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError("ToyStage supports ranks 0 and 1")

        return ToyStage(rank, self.offsets[rank])
