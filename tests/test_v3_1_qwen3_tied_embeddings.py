"""Validate tied embeddings across Qwen3 pipeline stages."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)


@pytest.mark.parametrize(
    "boundaries",
    [(0, 3, 6), (0, 2, 4, 6)],
)
def test_tied_embeddings(tmp_path, boundaries):
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
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )

    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).eval()

    tensors = {
        name: value.detach().cpu().clone().contiguous()
        for name, value in reference.state_dict().items()
        if name != "lm_head.weight"
    }

    assert "model.embed_tokens.weight" in tensors
    assert "lm_head.weight" not in tensors

    config.save_pretrained(str(tmp_path))
    save_file(tensors, str(tmp_path / "model.safetensors"))

    factory = Qwen3NRankStageOnlyFactory(
        checkpoint_dir=str(tmp_path),
        layer_boundaries=boundaries,
        device="cpu",
        dtype="float32",
    )

    stages = [
        factory(rank)
        for rank in range(len(boundaries) - 1)
    ]

    torch.testing.assert_close(
        stages[-1].model.lm_head.weight,
        reference.model.embed_tokens.weight,
        rtol=0,
        atol=0,
    )

    prompt = torch.tensor([[4, 7, 9]])
    caches = [None] * len(stages)

    with torch.inference_mode():
        for step, position, initial_payload in (
            (0, 0, prompt),
            (1, 3, 11),
        ):
            payload = initial_payload

            for rank, stage in enumerate(stages):
                result = stage(
                    SimpleNamespace(
                        step=step,
                        position=position,
                        payload=payload,
                    ),
                    caches[rank],
                )

                payload = result.payload
                caches[rank] = result.cache_state

            reference_ids = (
                prompt
                if step == 0
                else torch.tensor([[4, 7, 9, 11]])
            )

            expected = reference(
                input_ids=reference_ids,
                use_cache=False,
            ).logits[:, -1:, :]

            torch.testing.assert_close(
                payload[:, -1:, :],
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
