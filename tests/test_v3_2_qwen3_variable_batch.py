"""Real Qwen3 variable-length batching correctness."""

import pytest
import torch

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_qwen3_padding import (
    prepare_prefill,
    prepare_decode,
)


@pytest.mark.parametrize("prompts", [
    ((2, 5, 8), (7, 4, 12, 9, 3)),
    ((2, 5, 8, 11, 13), (7, 4)),
])
def test_variable_length_batched_prefill_decode(prompts):
    torch.manual_seed(42)
    torch.set_num_threads(1)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )
    config._attn_implementation = "eager"

    model = Qwen3ForCausalLM(config).eval()

    with torch.inference_mode():
        # Independent inference for each request.
        references = [
            model(
                torch.tensor([prompt]),
                use_cache=True,
            )
            for prompt in prompts
        ]

        # Joint prefill with explicit padding metadata.
        batch = prepare_prefill(prompts)

        batched = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=True,
        )

        reference_logits = torch.cat([
            output.logits[:, -1, :]
            for output in references
        ])

        batch_logits = batched.logits[:, -1, :]

        torch.testing.assert_close(
            batch_logits,
            reference_logits,
            rtol=1e-4,
            atol=1e-4,
        )

        # Decode the first generated token.
        tokens = [
            int(logits.argmax())
            for logits in reference_logits
        ]

        decode = prepare_decode(
            batch.attention_mask,
            tokens,
        )

        assert decode.position_ids.flatten().tolist() == [
            len(prompt) for prompt in prompts
        ]

        batch_next = model(
            input_ids=decode.input_ids,
            attention_mask=decode.attention_mask,
            position_ids=decode.position_ids,
            past_key_values=batched.past_key_values,
            use_cache=True,
        )

        reference_next = [
            model(
                input_ids=torch.tensor([[token]]),
                past_key_values=reference.past_key_values,
                use_cache=True,
            )
            for token, reference in zip(tokens, references)
        ]

        expected = torch.cat([
            output.logits[:, -1, :]
            for output in reference_next
        ])

        torch.testing.assert_close(
            batch_next.logits[:, -1, :],
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

    print(
        "PASS: Variable-length Qwen3 prefill and decode",
        [len(prompt) for prompt in prompts],
    )


def test_invalid_variable_length_input():
    with pytest.raises(ValueError):
        prepare_prefill([])

    with pytest.raises(ValueError):
        prepare_prefill([[1, 2], []])

    with pytest.raises(ValueError):
        prepare_decode(
            torch.ones((2, 3), dtype=torch.long),
            [1],
        )
