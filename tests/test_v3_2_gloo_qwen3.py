"""Real multi-process Qwen3 parity via CPU/Gloo."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_gloo_qwen3_smoke import (
    run_gloo_qwen3_rank,
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

DECODE_STEPS = 2


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


def reference_inference(model):
    prefill = {}
    decode = {rid: [] for rid in PROMPTS}
    states = {}

    with torch.inference_mode():
        for rid, prompt in PROMPTS.items():
            result = model(
                torch.tensor([prompt]),
                use_cache=True,
            )

            prefill[rid] = (
                result.logits[0, -1].detach().clone()
            )

            states[rid] = result

        for _ in range(DECODE_STEPS):
            next_states = {}

            for rid, result in states.items():
                token = int(
                    result.logits[0, -1].argmax()
                )

                next_result = model(
                    torch.tensor([[token]]),
                    past_key_values=result.past_key_values,
                    use_cache=True,
                )

                decode[rid].append(
                    next_result.logits[0, -1]
                    .detach().clone()
                )

                next_states[rid] = next_result

            states = next_states

    return prefill, decode, states


def gloo_worker(
    rank,
    world_size,
    rendezvous,
    checkpoint,
    boundaries,
    output_dir,
    chunk_size,
):
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    try:
        run_gloo_qwen3_rank(
            rank=rank,
            checkpoint=checkpoint,
            boundaries=boundaries,
            prompts=PROMPTS,
            chunk_size=chunk_size,
            decode_steps=DECODE_STEPS,
            output_dir=output_dir,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ranks", [2, 3, 4])
@pytest.mark.parametrize(
    "chunk_size", [None, 2],
    ids=["full-prefill", "chunked-prefill"],
)
def test_multiprocess_qwen3_parity(
    tmp_path, ranks, chunk_size
):
    model = make_model()

    checkpoint = tmp_path / "checkpoint"
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    model.save_pretrained(
        checkpoint,
        safe_serialization=True,
    )

    expected_prefill, expected_decode, states = (
        reference_inference(model)
    )

    mp.spawn(
        gloo_worker,
        args=(
            ranks,
            str(tmp_path / "gloo.init"),
            str(checkpoint),
            LAYOUTS[ranks],
            str(output_dir),
            chunk_size,
        ),
        nprocs=ranks,
        join=True,
    )

    results = torch.load(
        output_dir / "rank_0_results.pt",
        map_location="cpu",
        weights_only=True,
    )

    for rid in PROMPTS:
        torch.testing.assert_close(
            results["prefill_logits"][rid],
            expected_prefill[rid],
            rtol=1e-4,
            atol=1e-4,
        )

        actual_steps = results["decode_logits"][rid]

        assert len(actual_steps) == DECODE_STEPS

        for actual, expected in zip(
            actual_steps,
            expected_decode[rid],
        ):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )

    # Compare every rank's physical K/V blocks with
    # the independent full-model DynamicCache.
    all_owned_layers = []

    for rank in range(ranks):
        stage_state = torch.load(
            output_dir / f"rank_{rank}_kv.pt",
            map_location="cpu",
            weights_only=True,
        )

        expected_layers = tuple(range(
            LAYOUTS[ranks][rank],
            LAYOUTS[ranks][rank + 1],
        ))

        assert stage_state["rank"] == rank
        assert (
            stage_state["owned_layers"]
            == expected_layers
        )

        all_owned_layers.extend(expected_layers)

        before = stage_state["before_release"]
        after = stage_state["after_release"]

        assert before["requests"] == len(PROMPTS)
        assert before["allocated_blocks"] > 0
        assert not before["pending_batch"]

        assert after["requests"] == 0
        assert after["allocated_blocks"] == 0
        assert after["free_blocks"] == after[
            "capacity_blocks"
        ]
        assert not after["pending_batch"]

        for rid, final_reference in states.items():
            actual_layers = stage_state["kv"][rid]

            assert len(actual_layers) == len(
                expected_layers
            )

            for local, global_index in enumerate(
                expected_layers
            ):
                actual_k, actual_v = actual_layers[local]

                reference_layer = (
                    final_reference.past_key_values
                    .layers[global_index]
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

    assert sorted(all_owned_layers) == list(range(6))

    # Each rank executed identical microbatch sequence
    # metadata. The full prefill uses one batch; chunked
    # prefill uses multiple rounds.
    expected_prefill_rounds = (
        1 if chunk_size is None else 3
    )

    assert results["completed_sequences"] == (
        expected_prefill_rounds + DECODE_STEPS
    )
