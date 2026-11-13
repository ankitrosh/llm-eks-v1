"""Planner-driven asynchronous Qwen3 across Gloo ranks."""

from datetime import timedelta
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_async_gloo_qwen3 import (
    run_async_gloo_qwen3_rank,
)


LAYOUTS = {
    2: (0, 3, 6),
    3: (0, 1, 3, 6),
    4: (0, 1, 2, 4, 6),
}

PROMPTS = {
    1: (2, 5, 8),
    2: (7, 4, 12, 9, 3),
    3: (3, 9, 11, 17),
}

MAX_NEW_TOKENS = 3


def make_model():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 6,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )

    config._attn_implementation = "eager"

    return Qwen3ForCausalLM(config).eval()


def independent_reference(model):
    expected_logits = {}
    expected_tokens = {}
    final_states = {}

    with torch.inference_mode():
        for rid, prompt in PROMPTS.items():
            state = model(
                torch.tensor([prompt]),
                use_cache=True,
            )

            logits = [
                state.logits[0, -1].detach().clone()
            ]

            tokens = []

            for step in range(MAX_NEW_TOKENS):
                token = int(logits[-1].argmax())
                tokens.append(token)

                # The last generated token has not yet
                # entered the KV cache, as in the planner.
                if step == MAX_NEW_TOKENS - 1:
                    break

                state = model(
                    torch.tensor([[token]]),
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )

                logits.append(
                    state.logits[0, -1]
                    .detach()
                    .clone()
                )

            expected_logits[rid] = tuple(logits)
            expected_tokens[rid] = tuple(tokens)
            final_states[rid] = state

    return (
        expected_logits,
        expected_tokens,
        final_states,
    )


def worker(
    rank,
    world_size,
    rendezvous,
    checkpoint,
    boundaries,
    output_dir,
    batch_tokens,
    markers,
):
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )

    try:
        run_async_gloo_qwen3_rank(
            rank=rank,
            checkpoint=checkpoint,
            boundaries=boundaries,
            prompts=PROMPTS,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_tokens=batch_tokens,
            output_dir=output_dir,
            markers=markers,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3, 4])
@pytest.mark.parametrize(
    "batch_tokens",
    [16, 2],
    ids=["full-prefill", "chunked-prefill"],
)
def test_async_qwen3_planner_and_kv_parity(
    tmp_path, ranks, batch_tokens
):
    model = make_model()

    checkpoint = tmp_path / "checkpoint"
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    model.save_pretrained(
        checkpoint,
        safe_serialization=True,
    )

    (
        expected_logits,
        expected_tokens,
        final_states,
    ) = independent_reference(model)

    context = get_context("spawn")

    markers = tuple(
        context.Event() for _ in range(4)
    )

    mp.spawn(
        worker,
        args=(
            ranks,
            str(tmp_path / "gloo.init"),
            str(checkpoint),
            LAYOUTS[ranks],
            str(output_dir),
            batch_tokens,
            markers,
        ),
        nprocs=ranks,
        join=True,
    )

    result = torch.load(
        output_dir / "async_results.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert result["peak_pending"] == 2
    assert result["finished_request_ids"] == tuple(
        sorted(PROMPTS)
    )
    assert result["release_acknowledged"] is True
    assert result["released_request_ids"] == tuple(
        sorted(PROMPTS)
    )

    assert "prefill" in result["phases"]
    assert "decode" in result["phases"]

    prefill_count = sum(
        (len(prompt) + batch_tokens - 1)
        // batch_tokens
        for prompt in PROMPTS.values()
    )

    expected_batches = (
        prefill_count
        + len(PROMPTS) * (MAX_NEW_TOKENS - 1)
    )

    assert (
        result["completed_sequences"]
        == expected_batches
    )

    for rid in PROMPTS:
        assert result["generated_tokens"][rid] == (
            expected_tokens[rid]
        )

        actual_logits = result["emitted_logits"][rid]

        assert len(actual_logits) == MAX_NEW_TOKENS

        for actual, expected in zip(
            actual_logits,
            expected_logits[rid],
        ):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )

    # Verify actual distributed Qwen3 execution overlap.
    assert markers[0].is_set()
    assert markers[1].is_set()

    if ranks >= 3:
        assert markers[2].is_set()
        assert markers[3].is_set()

    owned_layers = []

    for rank in range(ranks):
        state = torch.load(
            output_dir / f"rank_{rank}_kv.pt",
            map_location="cpu",
            weights_only=True,
        )

        expected_owned = tuple(range(
            LAYOUTS[ranks][rank],
            LAYOUTS[ranks][rank + 1],
        ))

        assert state["owned_layers"] == expected_owned
        owned_layers.extend(expected_owned)

        before = state["before_release"]
        after = state["after_release"]

        assert before["requests"] == len(PROMPTS)
        assert before["allocated_blocks"] > 0
        assert not before["pending_batch"]

        assert after["requests"] == 0
        assert after["allocated_blocks"] == 0
        assert not after["pending_batch"]

        for rid in PROMPTS:
            actual_kv = state["kv"][rid]

            for local, global_layer in enumerate(
                expected_owned
            ):
                actual_k, actual_v = actual_kv[local]

                reference_layer = (
                    final_states[rid]
                    .past_key_values
                    .layers[global_layer]
                )

                torch.testing.assert_close(
                    actual_k,
                    reference_layer.keys,
                    rtol=1e-4,
                    atol=1e-4,
                )

                torch.testing.assert_close(
                    actual_v,
                    reference_layer.values,
                    rtol=1e-4,
                    atol=1e-4,
                )

    assert sorted(owned_layers) == list(range(6))
