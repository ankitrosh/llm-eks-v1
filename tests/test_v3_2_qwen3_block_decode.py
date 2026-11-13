"""Full Qwen3 decoder parity with block-backed KV."""

import pytest
import torch

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_qwen3_block_decode import (
    qwen3_block_decode_step,
)


def make_model():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    cfg = Qwen3Config(
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
    cfg._attn_implementation = "eager"

    return Qwen3ForCausalLM(cfg).eval()


def load_prefills(model, cache, prompts):
    references = [
        model(torch.tensor([prompt]), use_cache=True)
        for prompt in prompts
    ]

    receipt = cache.prepare_batch({
        index + 1: len(prompt)
        for index, prompt in enumerate(prompts)
    })

    for rid, result in enumerate(references, start=1):
        for index, layer in enumerate(
            result.past_key_values.layers
        ):
            cache.stage_write(
                rid, index, layer.keys, layer.values
            )

    cache.commit_batch(receipt)
    return references


@pytest.mark.parametrize("prompts", [
    ((2, 5, 8), (7, 4, 12, 9, 3, 6)),
    ((2, 5, 8, 11, 13), (7, 4)),
])
def test_block_decoder_matches_dynamic_cache(prompts):
    model = make_model()

    cache = BlockKVCache(
        layers=2,
        capacity_blocks=24,
        block_size=2,
        kv_heads=2,
        head_dim=16,
        max_context_tokens=32,
    )

    with torch.inference_mode():
        references = load_prefills(
            model, cache, prompts
        )
        ids = (1, 2)

        for _ in range(2):
            tokens = [
                int(r.logits[0, -1, :].argmax())
                for r in references
            ]

            before_bytes = cache.snapshot()[
                "physical_storage_bytes"
            ]

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "Full model forward or KV materialization"
                )

            original_forward = model.forward
            original_read = cache.read_request

            model.forward = forbidden
            cache.read_request = forbidden

            try:
                actual = qwen3_block_decode_step(
                    model, cache, ids, tokens
                )
            finally:
                model.forward = original_forward
                cache.read_request = original_read

            references = [
                model(
                    torch.tensor([[token]]),
                    past_key_values=r.past_key_values,
                    use_cache=True,
                )
                for token, r in zip(tokens, references)
            ]

            expected = torch.cat([
                r.logits[:, -1, :]
                for r in references
            ])

            torch.testing.assert_close(
                actual, expected,
                rtol=1e-4,
                atol=1e-4,
            )

            assert cache.snapshot()[
                "physical_storage_bytes"
            ] == before_bytes

            for rid, reference in zip(ids, references):
                stored = cache.read_request(rid)

                for (k, v), layer in zip(
                    stored,
                    reference.past_key_values.layers,
                ):
                    torch.testing.assert_close(
                        k, layer.keys,
                        rtol=1e-4, atol=1e-4,
                    )
                    torch.testing.assert_close(
                        v, layer.values,
                        rtol=1e-4, atol=1e-4,
                    )

        assert not cache.snapshot()["pending_batch"]

        assert cache.release_request(1)
        assert cache.release_request(2)

        assert cache.snapshot()["allocated_blocks"] == 0


def test_uninitialized_cache_rejected():
    model = make_model()

    cache = BlockKVCache(
        layers=2,
        capacity_blocks=8,
        block_size=2,
        kv_heads=2,
        head_dim=16,
        max_context_tokens=16,
    )

    with pytest.raises(ValueError, match="committed KV"):
        qwen3_block_decode_step(
            model, cache, (1,), (2,)
        )
