"""GPU validation of the V3 external adapter and attention plugins."""

import tempfile
from pathlib import Path

import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

from llm_runtime.v3_attention_plan import resolve_attention_plan
from llm_runtime.v3_attention_install import install_attention_extensions
from llm_runtime.v3_extension_contracts import StageSpec, StageInputs
from llm_runtime.v3_plugin_registry import load_plugin

assert torch.cuda.is_available(), "CUDA is unavailable"

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.set_device(0)

device = "cuda:0"

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("PyTorch:", torch.__version__, flush=True)

manifest = yaml.safe_load(
    Path("/workspace/v3_gpt2_extension.yaml").read_text()
)

plan = resolve_attention_plan(manifest)

specs = [
    StageSpec(
        rank=s["rank"],
        start_layer=s["start_layer"],
        end_layer=s["end_layer"],
        first_stage=(i == 0),
        final_stage=(i == len(plan["stages"]) - 1),
    )
    for i, s in enumerate(plan["stages"])
]

custom_layers = [
    (s["rank"], layer)
    for s in plan["stages"]
    for segment in s["attention"]
    if segment["implementation"] == "custom"
    for layer in range(
        segment["start_layer"], segment["end_layer"]
    )
]

assert custom_layers == [(0, 1), (1, 2)]
print("PASS: Manifest selected attention layers", flush=True)

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

tokens = torch.tensor(
    [[4, 12, 15, 31, 8, 26]],
    dtype=torch.long,
    device=device,
)

with tempfile.TemporaryDirectory() as directory:
    GPT2LMHeadModel(config).eval().save_pretrained(directory)

    reference = GPT2LMHeadModel.from_pretrained(
        directory,
        attn_implementation="eager",
    ).to(device).eval()

    adapter = load_plugin(
        "experiments.v3_gpt2_extension:GPT2Adapter",
        "model",
    )
    adapter.validate_partition(specs)

    stages = [
        adapter.create_stage(
            spec,
            checkpoint=directory,
            device=device,
            dtype="float32",
        )
        for spec in specs
    ]

    assert all(
        p.device.type == "cuda"
        for stage in stages
        for p in stage.parameters()
    )

    print("PASS: External adapter loaded on CUDA", flush=True)

    def forward(ids, offset=0, caches=None):
        with torch.inference_mode():
            first = adapter.forward_stage(
                stages[0],
                StageInputs(
                    payload=ids,
                    positions=offset,
                    cache_state=(
                        caches[0] if caches else None
                    ),
                ),
            )
            second = adapter.forward_stage(
                stages[1],
                StageInputs(
                    payload=first.hidden_states,
                    positions=offset,
                    cache_state=(
                        caches[1] if caches else None
                    ),
                ),
            )
        return second.logits

    with torch.inference_mode():
        expected = reference(
            tokens, use_cache=False
        ).logits

    native = forward(tokens)

    torch.testing.assert_close(
        native, expected,
        rtol=1e-4, atol=1e-4,
    )

    print("PASS: CUDA native-stage logit parity", flush=True)

    for stage in stages:
        install_attention_extensions(
            adapter, stage, plan, config.to_dict()
        )

    identity = forward(tokens)

    torch.testing.assert_close(
        identity, expected,
        rtol=1e-4, atol=1e-4,
    )

    assert stages[0].blocks["1"].attn.calls > 0
    assert stages[1].blocks["2"].attn.calls > 0

    print("PASS: CUDA custom attention logit parity", flush=True)

    caches = [
        DynamicCache(config=config),
        DynamicCache(config=config),
    ]

    # Prefill first, then decode one token.
    forward(tokens[:, :-1], 0, caches)

    last = forward(
        tokens[:, -1:],
        tokens.shape[1] - 1,
        caches,
    )

    torch.testing.assert_close(
        last[:, -1, :],
        expected[:, -1, :],
        rtol=1e-4, atol=1e-4,
    )

    print("PASS: CUDA partitioned KV-cache decoding", flush=True)

    sequence = tokens.clone()
    next_token = last[:, -1, :].argmax(
        dim=-1, keepdim=True
    )

    for _ in range(3):
        sequence = torch.cat(
            (sequence, next_token), dim=1
        )
        actual = forward(
            next_token,
            sequence.shape[1] - 1,
            caches,
        )

        with torch.inference_mode():
            expected_step = reference(
                sequence, use_cache=False
            ).logits[:, -1, :]

        torch.testing.assert_close(
            actual[:, -1, :],
            expected_step,
            rtol=1e-4, atol=1e-4,
        )

        next_token = actual[:, -1, :].argmax(
            dim=-1, keepdim=True
        )

    print("PASS: CUDA multi-token generation", flush=True)

    torch.cuda.synchronize()

    peak_mib = (
        torch.cuda.max_memory_allocated() / 1024**2
    )

    print(f"Peak allocated: {peak_mib:.2f} MiB", flush=True)

print("PASS: V3.1.6 single-GPU extension validation", flush=True)
