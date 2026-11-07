import unittest

from llm_runtime.v3_execution_factory import (
    GenericStageFactory,
    StageFactoryError,
    ValidatedStage,
    resolve_factory,
)
from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    StageResult,
    StageTask,
)


class MultiplyProvider:
    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, rank):
        def execute(task, cache):
            expected = (
                None if task.step == 0
                else (rank, task.request_id, task.step - 1)
            )
            assert cache == expected

            return StageResult(
                task.payload * self.factor,
                (rank, task.request_id, task.step),
            )

        return execute


class AddProvider:
    def __init__(self, amount=3):
        self.amount = amount

    def __call__(self, rank):
        def execute(task, cache):
            return StageResult(
                task.payload + self.amount + rank,
                {"request": task.request_id, "step": task.step},
            )

        return execute


class InvalidProvider:
    def __call__(self, rank):
        return lambda task, cache: task.payload


class ExecutionFactoryTests(unittest.TestCase):

    def make_factory(self, name, **kwargs):
        return GenericStageFactory(
            f"{__name__}:{name}",
            kwargs=kwargs,
        )

    def test_factory_loading(self):
        provider = resolve_factory(
            f"{__name__}:MultiplyProvider"
        )
        self.assertIs(provider, MultiplyProvider)

    def test_invalid_factory(self):
        with self.assertRaises(StageFactoryError):
            resolve_factory("missing.module:NotFound")

        with self.assertRaises(StageFactoryError):
            resolve_factory("invalid-path")

    def test_cache_remains_opaque(self):
        stage = self.make_factory(
            "MultiplyProvider", factor=2
        )(0)

        a = stage(
            StageTask(7, 0, 0, 4), None
        )
        b = stage(
            StageTask(7, 1, 1, a.payload),
            a.cache_state,
        )

        self.assertEqual(b.payload, 16)
        self.assertEqual(stage.forward_calls, 2)
        self.assertEqual(
            b.cache_state, (0, 7, 1)
        )

    def test_stage_result_validation(self):
        stage = self.make_factory(
            "InvalidProvider"
        )(0)

        with self.assertRaises(TypeError):
            stage(StageTask(1, 0, 0, 5), None)

    def test_model_swap_without_scheduler_changes(self):
        multiply = self.make_factory(
            "MultiplyProvider", factor=2
        )
        add = self.make_factory(
            "AddProvider", amount=3
        )

        for factory, expected in (
            (multiply, (4, 8, 16)),
            (add, (5, 8, 11)),
        ):
            with self.subTest(factory=factory.factory_path):
                with PipelineScheduler(
                    [factory(0)],
                    lambda x: x,
                ) as scheduler:
                    result = scheduler.submit(
                        2,
                        prompt_length=1,
                        max_new_tokens=3,
                    ).result(timeout=3)

                    self.assertEqual(
                        result.tokens, expected
                    )


if __name__ == "__main__":
    unittest.main()
