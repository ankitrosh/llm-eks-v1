"""Model-independent stage factory for V3.3.

The framework loads an adapter-backed stage factory from
configuration. Model construction happens within the worker.

A stage callable accepts (StageTask, opaque_cache) and returns
StageResult. The framework never inspects cache internals.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from .v3_pipeline_scheduler import StageResult, StageTask


class StageFactoryError(RuntimeError):
    pass


def resolve_factory(path: str) -> Callable:
    if not isinstance(path, str) or ":" not in path:
        raise StageFactoryError(
            "Factory must be in module:object format"
        )

    module_name, object_name = path.split(":", 1)

    if not module_name or not object_name:
        raise StageFactoryError("Invalid factory path")

    try:
        obj = importlib.import_module(module_name)

        for part in object_name.split("."):
            obj = getattr(obj, part)

    except (ImportError, AttributeError) as exc:
        raise StageFactoryError(
            f"Cannot load factory {path}: {exc}"
        ) from exc

    if not callable(obj):
        raise StageFactoryError(
            f"Factory {path} is not callable"
        )

    return obj


class ValidatedStage:
    """Enforces the common forward contract."""

    def __init__(self, stage: Callable):
        if not callable(stage):
            raise StageFactoryError(
                "Stage executor must be callable"
            )

        self._stage = stage
        self.forward_calls = 0

    def __call__(
        self,
        task: StageTask,
        cache: Any,
    ) -> StageResult:
        if not isinstance(task, StageTask):
            raise TypeError("Expected StageTask")

        result = self._stage(task, cache)

        if not isinstance(result, StageResult):
            raise TypeError(
                "Stage executor must return StageResult"
            )

        self.forward_calls += 1
        return result


@dataclass(frozen=True)
class GenericStageFactory:
    """Serializable configuration for a stage worker.

    factory_path points to an existing model-specific factory
    with signature:

        provider = Factory(**kwargs)
        executor = provider(rank)

    The executor implements:
        executor(StageTask, cache) -> StageResult
    """

    factory_path: str
    kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    def __call__(self, rank: int) -> ValidatedStage:
        if type(rank) is not int or rank < 0:
            raise ValueError("Invalid stage rank")

        constructor = resolve_factory(
            self.factory_path
        )

        provider = constructor(**dict(self.kwargs))
        executor = provider(rank)

        return ValidatedStage(executor)
