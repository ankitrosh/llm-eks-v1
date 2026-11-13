"""Validate N-rank Qwen3 logits against a full model.

Uses only small, locally generated model weights.
No AWS, network requests, or pretrained downloads.
"""

import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_execution_factory import (
    GenericStageFactory,
)
from llm_runtime.v3_pipeline_scheduler import StageTask


def validate(checkpoint, reference, boundaries):
    factory = GenericStageFactory(
        factory_path=(
            "llm_runtime.providers."
            "v3_1_qwen3_nrank_factory:"
            "Qwen3NRankStageOnlyFactory"
        ),
        kwargs={
            "checkpoint_dir": checkpoint,
            "layer_boundaries": list(boundaries),
            "device": "cpu",
            "dtype": "float32",
        },
    )

    world_size = len(boundaries) - 1

    stages = [
        factory(rank)
        for rank in range(world_size)
    ]

    caches = [None] * world_size

    # Check that the stages collectively own the
    # expected parameter count, without duplicating
    # non-owned transformer layers.
    stage_parameters = sum(
        parameter.numel()
        for stage in stages
        for parameter in stage._stage.model.parameters()
    )

    reference_parameters = sum(
        parameter.numel()
        for parameter in reference.parameters()
    )

    assert stage_parameters == reference_parameters, (
        stage_parameters,
        reference_parameters,
    )

    for rank, stage in enumerate(stages):
        partial = stage._stage.model

        assert partial.owned_layers == tuple(
            range(
                boundaries[rank],
                boundaries[rank + 1],
            )
        )

        assert hasattr(
            partial.model, "embed_tokens"
        ) == (rank == 0)

        assert hasattr(
            partial, "lm_head"
        ) == (rank == world_size - 1)

    history = [2, 5, 8, 11]
    generated = []

    for step in range(3):
        if step == 0:
            payload = torch.tensor(
                [history],
                dtype=torch.long,
            )
            position = 0

        else:
            payload = torch.tensor(
                [[generated[-1]]],
                dtype=torch.long,
            )
            position = len(history) - 1

        for rank, stage in enumerate(stages):
            result = stage(
                StageTask(
                    request_id=1,
                    step=step,
                    position=position,
                    payload=payload,
                ),
                caches[rank],
            )

            payload = result.payload
            caches[rank] = result.cache_state

        with torch.inference_mode():
            expected_logits = reference(
                torch.tensor(
                    [history],
                    dtype=torch.long,
                ),
                use_cache=False,
            ).logits[:, -1:, :]

        torch.testing.assert_close(
            payload,
            expected_logits,
            rtol=1e-4,
            atol=1e-4,
        )

        token = int(
            payload[0, -1, :].argmax().item()
        )

        reference_token = int(
            expected_logits[0, -1, :].argmax().item()
        )

        assert token == reference_token

        generated.append(token)
        history.append(token)

    print(
        f"PASS: {world_size}-rank Qwen3 parity, "
        f"boundaries={list(boundaries)}, "
        f"generated={generated}",
        flush=True,
    )


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)

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

    reference = Qwen3ForCausalLM(
        config
    ).eval()

    with tempfile.TemporaryDirectory(
        prefix="v31-qwen3-"
    ) as checkpoint:
        reference.save_pretrained(
            checkpoint,
            safe_serialization=True,
        )

        validate(
            checkpoint,
            reference,
            (0, 1, 3, 6),
        )

        validate(
            checkpoint,
            reference,
            (0, 1, 2, 4, 6),
        )

    print(
        "PASS: All tiny Qwen3 N-rank CPU "
        "reference-model parity checks",
        flush=True,
    )


if __name__ == "__main__":
    main()
