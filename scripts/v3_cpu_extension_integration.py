"""Validate external model adapters and attention installation."""

import copy
import tempfile

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from llm_runtime.v3_extension_contracts import (
    StageInputs,
    StageSpec,
)
from llm_runtime.v3_attention_install import (
    install_attention_extensions,
)
from llm_runtime.v3_plugin_registry import (
    PluginError,
    collect_attention_plugins,
    load_plugin,
)


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
    use_cache=False,
)

specs = [
    StageSpec(
        rank=0,
        start_layer=0,
        end_layer=2,
        first_stage=True,
        final_stage=False,
    ),
    StageSpec(
        rank=1,
        start_layer=2,
        end_layer=4,
        first_stage=False,
        final_stage=True,
    ),
]

tokens = torch.tensor(
    [[4, 12, 15, 31, 8, 26]],
    dtype=torch.long,
)

with tempfile.TemporaryDirectory() as tmp:
    original = GPT2LMHeadModel(config).eval()
    original.save_pretrained(
        tmp, safe_serialization=True
    )

    reference = GPT2LMHeadModel.from_pretrained(
        tmp, attn_implementation="eager"
    ).eval()

    with torch.inference_mode():
        expected = reference(
            tokens, use_cache=False
        ).logits

    adapter = load_plugin(
        "experiments.v3_gpt2_extension:GPT2Adapter",
        "model",
    )
    adapter.validate_partition(specs)

    left, right = [
        adapter.create_stage(
            spec,
            checkpoint=tmp,
            device="cpu",
            dtype="float32",
        )
        for spec in specs
    ]

    print(
        "PASS: External model adapter loaded and partitioned",
        flush=True,
    )

    def run():
        with torch.inference_mode():
            first = adapter.forward_stage(
                left,
                StageInputs(
                    payload=tokens, positions=0
                ),
            )

            second = adapter.forward_stage(
                right,
                StageInputs(
                    payload=first.hidden_states,
                    positions=0,
                ),
            )

        return second.logits

    actual = run()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    assert torch.equal(
        actual.argmax(-1),
        expected.argmax(-1),
    )

    print(
        "PASS: Two-stage logits match unsplit GPT-2",
        flush=True,
    )

    untouched0 = left.blocks["0"].attn
    untouched3 = right.blocks["3"].attn

    plan = {
        "stages": [
            {
                "rank": 0,
                "attention": [
                    {
                        "start_layer": 0,
                        "end_layer": 1,
                        "implementation": "model_native",
                    },
                    {
                        "start_layer": 1,
                        "end_layer": 2,
                        "implementation": "custom",
                        "plugin": (
                            "experiments.v3_gpt2_extension:"
                            "IdentityAttention"
                        ),
                        "options": {},
                    },
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
                    },
                    {
                        "start_layer": 3,
                        "end_layer": 4,
                        "implementation": "model_native",
                    },
                ],
            },
        ],
    }

    assert len(collect_attention_plugins(plan)) == 2

    assert install_attention_extensions(
        adapter, left, plan, config.to_dict()
    ) == [1]

    assert install_attention_extensions(
        adapter, right, plan, config.to_dict()
    ) == [2]

    identity = run()

    torch.testing.assert_close(
        identity,
        expected,
        rtol=1e-4,
        atol=1e-5,
    )

    assert left.blocks["1"].attn.calls > 0
    assert right.blocks["2"].attn.calls > 0

    assert left.blocks["0"].attn is untouched0
    assert right.blocks["3"].attn is untouched3

    print(
        "PASS: Per-layer plugins executed with logit parity",
        flush=True,
    )

    modified_plan = copy.deepcopy(plan)

    modified_plan["stages"][1]["attention"][0].update(
        plugin=(
            "experiments.v3_gpt2_extension:"
            "PerturbAttention"
        ),
        options={"delta": 0.5},
    )

    assert install_attention_extensions(
        adapter,
        right,
        modified_plan,
        config.to_dict(),
    ) == [2]

    changed = run()

    difference = (
        changed - identity
    ).abs().max().item()

    assert difference > 1e-5, difference

    print(
        "PASS: Controlled attention modification "
        f"changes logits ({difference:.6f})",
        flush=True,
    )

    bad_plan = copy.deepcopy(plan)
    bad_plan["stages"][0]["attention"][1]["plugin"] = (
        "experiments.v3_gpt2_extension:MissingPlugin"
    )

    try:
        install_attention_extensions(
            adapter,
            left,
            bad_plan,
            config.to_dict(),
        )
    except PluginError:
        print(
            "PASS: Missing plugin rejected",
            flush=True,
        )
    else:
        raise AssertionError(
            "Missing plugin unexpectedly accepted"
        )

print(
    "PASS: V3 external adapter and attention integration",
    flush=True,
)
