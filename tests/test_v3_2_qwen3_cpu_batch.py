"""Native Qwen3 batched execution correctness baseline."""

import torch

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_batch_planner import (
    ContinuousBatchPlanner,
)


def test_batched_prefill_decode_matches_individual():
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

    planner_config = {
        "pipeline": {
            "memory": {
                "max_active_requests": 2,
                "max_total_tokens_per_request": 16,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "policy": "fcfs",
            "max_requests_per_batch": 2,
            "max_batch_tokens": 8,
            "max_queue_requests": 4,
        },
    }

    planner = ContinuousBatchPlanner(planner_config)

    planner.submit(1, [2, 5, 8], 2)
    planner.submit(2, [7, 4, 12], 2)

    with torch.inference_mode():
        # Independent reference runs.
        prompts = [
            torch.tensor([[2, 5, 8]]),
            torch.tensor([[7, 4, 12]]),
        ]

        references = [
            model(ids, use_cache=True)
            for ids in prompts
        ]

        # One physical model forward for both requests.
        plan = planner.next_batch()

        assert plan.phase == "prefill"
        assert len(plan.items) == 2

        batch_ids = torch.tensor([
            item.tokens for item in plan.items
        ])

        batched = model(batch_ids, use_cache=True)

        expected = torch.cat([
            output.logits[:, -1, :]
            for output in references
        ])

        actual = batched.logits[:, -1, :]

        torch.testing.assert_close(
            actual, expected,
            rtol=1e-4,
            atol=1e-4,
        )

        first_tokens = actual.argmax(dim=-1)

        planner.complete_batch({
            item.request_id: int(first_tokens[i])
            for i, item in enumerate(plan.items)
        })

        # Decode both requests in one forward, reusing KV.
        plan = planner.next_batch()

        assert plan.phase == "decode"
        assert len(plan.items) == 2

        decode_ids = torch.tensor([
            item.tokens for item in plan.items
        ])

        batched_decode = model(
            decode_ids,
            past_key_values=batched.past_key_values,
            use_cache=True,
        )

        reference_decode = [
            model(
                torch.tensor([[int(first_tokens[i])]]),
                past_key_values=references[i].past_key_values,
                use_cache=True,
            )
            for i in range(2)
        ]

        expected_decode = torch.cat([
            output.logits[:, -1, :]
            for output in reference_decode
        ])

        actual_decode = batched_decode.logits[:, -1, :]

        torch.testing.assert_close(
            actual_decode,
            expected_decode,
            rtol=1e-4,
            atol=1e-4,
        )

        second_tokens = actual_decode.argmax(dim=-1)

        result = planner.complete_batch({
            item.request_id: int(second_tokens[i])
            for i, item in enumerate(plan.items)
        })

    assert set(result.finished) == {1, 2}
    assert set(planner.drain_releases()) == {1, 2}
    assert planner.snapshot()["reserved_kv_tokens"] == 0

    print("PASS: Native Qwen3 batched prefill")
    print("PASS: Native Qwen3 batched decode")
    print("PASS: Batched outputs match individual inference")
    print("PASS: KV-cache reservations released")
