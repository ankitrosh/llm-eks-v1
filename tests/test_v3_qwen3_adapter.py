import unittest

from llm_runtime.v3_adapter_registry import (
    UnsupportedAdapterError,
    create_adapter,
    list_adapters,
)
from llm_runtime.v3_extension_contracts import (
    StageInputs,
    StageSpec,
)


class FakeQwen3Stage:
    calls = []

    def __init__(self, model_dir, rank, split=20):
        self.rank = rank
        self.split = split
        self.start = 0 if rank == 0 else split
        self.stop = split if rank == 0 else 40

        self.calls.append(
            ("construct", rank, split, model_dir)
        )

    def load(self):
        self.calls.append(("load", self.rank))
        return self

    def make_cache(self):
        return {"rank": self.rank}

    def forward_local(self, data, start_position, cache):
        self.calls.append(
            ("forward", self.rank, start_position)
        )
        return (data, start_position, cache)


def make_specs(split=20):
    return [
        StageSpec(
            rank=0,
            start_layer=0,
            end_layer=split,
            first_stage=True,
            final_stage=False,
        ),
        StageSpec(
            rank=1,
            start_layer=split,
            end_layer=40,
            first_stage=False,
            final_stage=True,
        ),
    ]


class Qwen3AdapterTests(unittest.TestCase):

    def setUp(self):
        FakeQwen3Stage.calls.clear()

        self.adapter = create_adapter(
            "qwen3",
            num_layers=40,
            stage_factory=FakeQwen3Stage,
        )

    def test_adapter_registry(self):
        catalog = list_adapters()
        self.assertIn("qwen3", catalog)
        self.assertNotIn("gpt2", catalog)

    def test_unsupported_model(self):
        with self.assertRaises(UnsupportedAdapterError):
            create_adapter("gpt2", num_layers=6)

    def test_equal_partition(self):
        stages = self.adapter.validate_partition(
            make_specs(20)
        )
        self.assertEqual(stages[0].end_layer, 20)
        self.assertEqual(stages[1].start_layer, 20)

    def test_unequal_partition(self):
        stages = self.adapter.validate_partition(
            make_specs(23)
        )
        self.assertEqual(stages[0].end_layer, 23)
        self.assertEqual(stages[1].start_layer, 23)

    def test_more_than_two_ranks_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.validate_partition(
                make_specs() + [make_specs()[0]]
            )

    def test_gap_rejected(self):
        stages = make_specs()
        stages[1] = StageSpec(
            rank=1,
            start_layer=21,
            end_layer=40,
            first_stage=False,
            final_stage=True,
        )
        with self.assertRaises(ValueError):
            self.adapter.validate_partition(stages)

    def test_stage_creation_and_range(self):
        specs = make_specs(23)
        self.adapter.validate_partition(specs)

        stage = self.adapter.create_stage(
            specs[1],
            checkpoint="/mock/model",
            device="cuda",
            dtype="bfloat16",
        )

        self.assertEqual(stage.module.start, 23)
        self.assertEqual(stage.module.stop, 40)

        self.assertIn(
            ("construct", 1, 23, "/mock/model"),
            FakeQwen3Stage.calls,
        )

    def test_cpu_not_supported(self):
        specs = make_specs()
        self.adapter.validate_partition(specs)

        with self.assertRaises(ValueError):
            self.adapter.create_stage(
                specs[0],
                checkpoint="/mock/model",
                device="cpu",
                dtype="bfloat16",
            )

    def test_forward_and_cache(self):
        specs = make_specs()
        self.adapter.validate_partition(specs)

        first = self.adapter.create_stage(
            specs[0],
            checkpoint="/mock/model",
            device="cuda",
            dtype="bfloat16",
        )

        cache = self.adapter.create_cache(first)

        result = self.adapter.forward_stage(
            first,
            StageInputs(
                payload="input_tokens",
                positions=5,
                cache_state=cache,
            ),
        )

        self.assertEqual(
            result.hidden_states,
            ("input_tokens", 5, cache),
        )
        self.assertIsNone(result.logits)
        self.assertIs(result.cache_state, cache)

    def test_final_stage_returns_logits(self):
        specs = make_specs()
        self.adapter.validate_partition(specs)

        last = self.adapter.create_stage(
            specs[1],
            checkpoint="/mock/model",
            device="cuda",
            dtype="bfloat16",
        )

        result = self.adapter.forward_stage(
            last,
            StageInputs(
                payload="hidden_states",
                positions=0,
                cache_state={},
            ),
        )

        self.assertIsNone(result.hidden_states)
        self.assertIsNotNone(result.logits)

    def test_attention_override_not_silently_applied(self):
        specs = make_specs()
        self.adapter.validate_partition(specs)

        with self.assertRaises(NotImplementedError):
            self.adapter.replace_attention(
                None,
                global_layer_idx=5,
                module=None,
            )


if __name__ == "__main__":
    unittest.main()
