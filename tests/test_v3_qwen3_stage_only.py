import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    Qwen3StageOnlyFactory,
)


class Qwen3StageOnlyTests(unittest.TestCase):

    def check_loading(self, sharded):
        torch.manual_seed(42)

        config = Qwen3Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            use_cache=True,
        )

        config._attn_implementation = "eager"

        original = Qwen3ForCausalLM(config).eval()

        with tempfile.TemporaryDirectory() as directory:
            kwargs = (
                {"max_shard_size": "100KB"}
                if sharded else {}
            )

            original.save_pretrained(
                directory, **kwargs
            )

            index = Path(directory) / (
                "model.safetensors.index.json"
            )

            self.assertEqual(index.exists(), sharded)

            factory = Qwen3StageOnlyFactory(
                checkpoint_dir=directory,
                device="cpu",
            )

            first = factory(0)
            second = factory(1)

            self.assertEqual(
                first.model.owned_layers, (0, 1)
            )
            self.assertEqual(
                second.model.owned_layers, (2, 3)
            )

            for idx in (2, 3):
                self.assertIsInstance(
                    first.model.model.layers[idx],
                    nn.Identity,
                )

            for idx in (0, 1):
                self.assertIsInstance(
                    second.model.model.layers[idx],
                    nn.Identity,
                )

            original_params = dict(
                original.named_parameters()
            )
            first_params = dict(
                first.model.named_parameters()
            )
            second_params = dict(
                second.model.named_parameters()
            )

            self.assertFalse(
                set(first_params) & set(second_params)
            )

            self.assertEqual(
                set(first_params) | set(second_params),
                set(original_params),
            )

            for name, tensor in (
                first_params | second_params
            ).items():
                self.assertFalse(tensor.is_meta)

                torch.testing.assert_close(
                    tensor,
                    original_params[name],
                    rtol=0,
                    atol=0,
                )

            full_count = sum(
                p.numel()
                for p in original.parameters()
            )

            counts = [
                sum(
                    p.numel()
                    for p in stage.model.parameters()
                )
                for stage in (first, second)
            ]

            self.assertEqual(
                sum(counts), full_count
            )

            self.assertTrue(
                all(count < full_count for count in counts)
            )

            print(
                "PASS: Stage-only parameter counts "
                f"{counts} vs full {full_count}",
                flush=True,
            )

    def test_single_file_stage_loading(self):
        self.check_loading(sharded=False)

    def test_sharded_stage_loading(self):
        self.check_loading(sharded=True)


if __name__ == "__main__":
    unittest.main()
