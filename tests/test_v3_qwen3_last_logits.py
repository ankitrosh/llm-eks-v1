import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    Qwen3StageOnlyFactory,
)
from llm_runtime.v3_pipeline_scheduler import StageTask


class LastLogitsTests(unittest.TestCase):

    def test_prefill_and_decode_logits(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

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

        full = Qwen3ForCausalLM(config).eval()

        with tempfile.TemporaryDirectory() as directory:
            full.save_pretrained(directory)

            factory = Qwen3StageOnlyFactory(
                checkpoint_dir=directory,
                device="cpu",
                split_at=1,
            )

            first = factory(0)
            second = factory(1)

            prompt = torch.tensor(
                [[4, 12, 15, 31, 8, 26]],
                dtype=torch.long,
            )

            first_result = first(
                StageTask(1, 0, 0, prompt), None
            )

            self.assertEqual(
                tuple(first_result.payload.shape),
                (1, 6, 128),
            )

            second_result = second(
                StageTask(
                    1, 0, 0,
                    first_result.payload,
                ),
                None,
            )

            self.assertEqual(
                tuple(second_result.payload.shape),
                (1, 1, 128),
            )

            with torch.inference_mode():
                reference = full(
                    prompt, use_cache=False
                ).logits[:, -1:, :]

            torch.testing.assert_close(
                second_result.payload,
                reference,
                rtol=1e-3,
                atol=1e-4,
            )

            next_token = int(
                reference.argmax(dim=-1).item()
            )

            step0 = first(
                StageTask(
                    1, 1, prompt.shape[1],
                    next_token,
                ),
                first_result.cache_state,
            )

            step1 = second(
                StageTask(
                    1, 1, prompt.shape[1],
                    step0.payload,
                ),
                second_result.cache_state,
            )

            extended = torch.cat(
                [
                    prompt,
                    torch.tensor([[next_token]]),
                ],
                dim=1,
            )

            with torch.inference_mode():
                reference_next = full(
                    extended, use_cache=False
                ).logits[:, -1:, :]

            self.assertEqual(
                tuple(step1.payload.shape),
                (1, 1, 128),
            )

            torch.testing.assert_close(
                step1.payload,
                reference_next,
                rtol=1e-3,
                atol=1e-4,
            )

        print(
            "PASS: Qwen3 last-token logits match "
            "reference for prefill and decode",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
