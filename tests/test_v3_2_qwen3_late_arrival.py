"""Real Qwen3 late-arrival KV batching correctness."""

import torch

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_qwen3_cache_batch import (
    batched_decode_step,
    merge_caches,
)


def make_model():
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

    return Qwen3ForCausalLM(config).eval()


def test_late_arrival_merges_unequal_caches():
    model = make_model()

    with torch.inference_mode():
        # Request A starts first.
        a = model(
            torch.tensor([[2, 5, 8]]),
            use_cache=True,
        )

        first_a = int(
            a.logits[0, -1].argmax()
        )

        # A has already entered decoding.
        a = model(
            torch.tensor([[first_a]]),
            past_key_values=a.past_key_values,
            use_cache=True,
        )

        next_a = int(
            a.logits[0, -1].argmax()
        )

        # Request B arrives later with a longer prompt.
        b = model(
            torch.tensor([[7, 4, 12, 9, 3, 6]]),
            use_cache=True,
        )

        next_b = int(
            b.logits[0, -1].argmax()
        )

        assert a.past_key_values.get_seq_length() == 4
        assert b.past_key_values.get_seq_length() == 6

        merged = merge_caches(
            (a.past_key_values, b.past_key_values),
            model.config,
        )

        assert merged.lengths == (4, 6)
        assert merged.attention_mask.tolist() == [
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]

        # Count actual model forwards.
        calls = [0]

        def count_forward(module, inputs, output):
            calls[0] += 1

        hook = model.register_forward_hook(count_forward)

        try:
            actual, caches = batched_decode_step(
                model,
                (a.past_key_values, b.past_key_values),
                (next_a, next_b),
            )
        finally:
            hook.remove()

        assert calls[0] == 1
        assert [c.get_seq_length() for c in caches] == [
            5, 7
        ]

        # Independent reference forwards.
        ref_a = model(
            torch.tensor([[next_a]]),
            past_key_values=a.past_key_values,
            use_cache=True,
        )

        ref_b = model(
            torch.tensor([[next_b]]),
            past_key_values=b.past_key_values,
            use_cache=True,
        )

        expected = torch.cat([
            ref_a.logits[:, -1, :],
            ref_b.logits[:, -1, :],
        ])

        torch.testing.assert_close(
            actual, expected,
            rtol=1e-4, atol=1e-4,
        )

        # Compare the actual per-request KV tensors.
        for individual, reference in zip(
            caches,
            (
                ref_a.past_key_values,
                ref_b.past_key_values,
            ),
        ):
            for left, right in zip(
                individual.layers,
                reference.layers,
            ):
                torch.testing.assert_close(
                    left.keys, right.keys,
                    rtol=1e-4, atol=1e-4,
                )
                torch.testing.assert_close(
                    left.values, right.values,
                    rtol=1e-4, atol=1e-4,
                )

        # Both requests continue decoding together.
        next_tokens = actual.argmax(dim=-1).tolist()

        actual2, caches2 = batched_decode_step(
            model, caches, next_tokens
        )

        ref_a2 = model(
            torch.tensor([[next_tokens[0]]]),
            past_key_values=ref_a.past_key_values,
            use_cache=True,
        )
        ref_b2 = model(
            torch.tensor([[next_tokens[1]]]),
            past_key_values=ref_b.past_key_values,
            use_cache=True,
        )

        expected2 = torch.cat([
            ref_a2.logits[:, -1, :],
            ref_b2.logits[:, -1, :],
        ])

        torch.testing.assert_close(
            actual2, expected2,
            rtol=1e-4, atol=1e-4,
        )

        assert [c.get_seq_length() for c in caches2] == [
            6, 8
        ]

    print("PASS: Late request joined active decoding")
    print("PASS: One physical forward for two requests")
    print("PASS: Independent KV caches restored")
    print("PASS: Subsequent batched decode correct")
