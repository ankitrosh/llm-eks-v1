"""V3.3.3a: Qwen3 full-forward vs interleaved KV-cache reference."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache


PROMPTS = [
    [4, 12, 15, 31, 8, 26],
    [9, 18, 7, 21],
    [3, 16, 27, 5, 11],
    [20, 6, 13, 17, 2, 10],
]

STEPS = 4


def main():
    torch.set_num_threads(2)
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
        attention_dropout=0.0,
        use_cache=True,
    )

    config._attn_implementation = "eager"

    model = Qwen3ForCausalLM(config).eval()

    assert len(model.model.layers) == 4

    references = {}

    with torch.inference_mode():
        # Unsplit full-context reference for every request.
        for rid, prompt in enumerate(PROMPTS, 1):
            sequence = list(prompt)
            tokens = []
            logits_history = []

            for step in range(STEPS):
                ids = torch.tensor(
                    [sequence], dtype=torch.long
                )

                output = model(
                    input_ids=ids,
                    use_cache=False,
                )

                logits = output.logits[:, -1, :]
                logits_history.append(logits.clone())

                token = int(logits.argmax(dim=-1).item())
                tokens.append(token)
                sequence.append(token)

            references[rid] = (tuple(tokens), logits_history)

        # Simulate multiple requests sharing model weights,
        # but keeping separate cache objects.
        caches = {
            rid: DynamicCache(config=config)
            for rid in references
        }

        generated = {
            rid: [] for rid in references
        }

        for step in range(STEPS):
            for rid, prompt in enumerate(PROMPTS, 1):

                if step == 0:
                    input_tokens = prompt
                    position = 0
                else:
                    input_tokens = [generated[rid][-1]]
                    position = len(prompt) + step - 1

                ids = torch.tensor(
                    [input_tokens], dtype=torch.long
                )

                cache_position = torch.arange(
                    position,
                    position + len(input_tokens),
                    dtype=torch.long,
                )

                output = model(
                    input_ids=ids,
                    past_key_values=caches[rid],
                    cache_position=cache_position,
                    use_cache=True,
                )

                assert output.past_key_values is caches[rid]

                expected_tokens, expected_logits = references[rid]

                actual_logits = output.logits[:, -1, :]

                torch.testing.assert_close(
                    actual_logits,
                    expected_logits[step],
                    rtol=1e-3,
                    atol=1e-4,
                )

                token = int(
                    actual_logits.argmax(dim=-1).item()
                )

                assert token == expected_tokens[step]

                generated[rid].append(token)

                expected_cache_length = (
                    len(prompt) + step
                )

                assert caches[rid].get_seq_length() == (
                    expected_cache_length
                )

        for rid, (expected, _) in references.items():
            actual = tuple(generated[rid])

            assert actual == expected, (rid, actual, expected)

            print(
                f"PASS: Qwen3 request {rid} cached tokens "
                f"match full reference: {actual}",
                flush=True,
            )

    assert len({id(cache) for cache in caches.values()}) == 4

    print(
        "PASS: Four independent Qwen3 KV caches",
        flush=True,
    )
    print(
        "PASS: Qwen3 GQA, RoPE and cache positions",
        flush=True,
    )
    print(
        "PASS: V3.3.3a Qwen3 architecture reference",
        flush=True,
    )


if __name__ == "__main__":
    main()
