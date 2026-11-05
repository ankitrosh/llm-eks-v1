"""Verify separate per-stage caches through the V3 adapter."""

import tempfile

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

from llm_runtime.v3_extension_contracts import StageSpec, StageInputs
from llm_runtime.v3_plugin_registry import load_plugin
from llm_runtime.v3_attention_install import install_attention_extensions


torch.set_num_threads(2)
torch.manual_seed(42)

config = GPT2Config(
    vocab_size=128,
    n_positions=64,
    n_ctx=64,
    n_embd=64,
    n_layer=4,
    n_head=4,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_cache=True,
)

specs = [
    StageSpec(0, 0, 2, True, False),
    StageSpec(1, 2, 4, False, True),
]

with tempfile.TemporaryDirectory() as tmp:
    GPT2LMHeadModel(config).eval().save_pretrained(tmp)

    reference = GPT2LMHeadModel.from_pretrained(
        tmp, attn_implementation="eager"
    ).eval()

    adapter = load_plugin(
        "experiments.v3_gpt2_extension:GPT2Adapter",
        "model",
    )
    adapter.validate_partition(specs)

    stages = [
        adapter.create_stage(
            spec, checkpoint=tmp,
            device="cpu", dtype="float32"
        )
        for spec in specs
    ]

    plan = {
        "stages": [
            {
                "rank": 0,
                "attention": [
                    {
                        "start_layer": 1,
                        "end_layer": 2,
                        "implementation": "custom",
                        "plugin": (
                            "experiments.v3_gpt2_extension:"
                            "IdentityAttention"
                        ),
                        "options": {},
                    }
                ],
            },
            {
                "rank": 1,
                "attention": [
                    {
                        "start_layer": 2,
                        "end_layer": 3,
                        "implementation": "custom",
                        "plugin": (
                            "experiments.v3_gpt2_extension:"
                            "IdentityAttention"
                        ),
                        "options": {},
                    }
                ],
            },
        ]
    }

    for stage in stages:
        install_attention_extensions(
            adapter, stage, plan, config.to_dict()
        )

    def new_caches():
        # Separate cache objects, as on different GPU ranks.
        return [
            DynamicCache(config=config),
            DynamicCache(config=config),
        ]

    def forward(ids, position, caches):
        with torch.inference_mode():
            first = adapter.forward_stage(
                stages[0],
                StageInputs(
                    payload=ids,
                    positions=position,
                    cache_state=caches[0],
                ),
            )
            second = adapter.forward_stage(
                stages[1],
                StageInputs(
                    payload=first.hidden_states,
                    positions=position,
                    cache_state=caches[1],
                ),
            )

        assert first.cache_state is caches[0]
        assert second.cache_state is caches[1]
        return second.logits

    def expected(ids):
        with torch.inference_mode():
            return reference(
                ids, use_cache=False
            ).logits

    a = torch.tensor([[4, 12, 15, 31, 8, 26]])
    b = torch.tensor([[9, 18, 7, 21, 14, 3]])

    ca = new_caches()
    cb = new_caches()

    # Interleave two requests to detect accidental
    # cross-request cache sharing.
    forward(a[:, :-1], 0, ca)
    forward(b[:, :-1], 0, cb)

    a_last = forward(a[:, -1:], 5, ca)
    b_last = forward(b[:, -1:], 5, cb)

    torch.testing.assert_close(
        a_last[:, -1, :],
        expected(a)[:, -1, :],
        rtol=1e-4, atol=1e-5,
    )
    torch.testing.assert_close(
        b_last[:, -1, :],
        expected(b)[:, -1, :],
        rtol=1e-4, atol=1e-5,
    )

    print("PASS: Two-stage cached decoding", flush=True)
    print("PASS: Interleaved request cache isolation", flush=True)

    # Generate additional tokens, comparing each decode
    # step against a complete unsplit reference forward.
    sequence = a.clone()
    next_token = a_last[:, -1, :].argmax(
        dim=-1, keepdim=True
    )

    for _ in range(3):
        sequence = torch.cat(
            [sequence, next_token], dim=1
        )

        actual = forward(
            next_token,
            sequence.shape[1] - 1,
            ca,
        )

        torch.testing.assert_close(
            actual[:, -1, :],
            expected(sequence)[:, -1, :],
            rtol=1e-4, atol=1e-5,
        )

        next_token = actual[:, -1, :].argmax(
            dim=-1, keepdim=True
        )

    print(
        "PASS: Multi-token incremental generation",
        flush=True,
    )

    assert stages[0].blocks["1"].attn.calls >= 5
    assert stages[1].blocks["2"].attn.calls >= 5

    print(
        "PASS: Attention plugins executed during decoding",
        flush=True,
    )

print("PASS: V3 partitioned KV-cache integration", flush=True)
