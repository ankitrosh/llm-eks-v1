"""V3.2.2: Real GPT-2 inference through the pipeline scheduler."""

import tempfile
from pathlib import Path

import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

from llm_runtime.v3_attention_plan import resolve_attention_plan
from llm_runtime.v3_attention_install import install_attention_extensions
from llm_runtime.v3_extension_contracts import StageSpec, StageInputs
from llm_runtime.v3_pipeline_scheduler import (
    PipelineScheduler,
    StageResult,
)
from llm_runtime.v3_plugin_registry import load_plugin


torch.set_num_threads(2)
torch.manual_seed(42)

manifest = yaml.safe_load(
    Path("examples/v3_gpt2_extension.yaml").read_text()
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

assert len(specs) == 2

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

prompts = [
    [4, 12, 15, 31, 8, 26],
    [9, 18, 7, 21],
    [3, 16, 27, 5, 11],
]

max_new_tokens = 4

with tempfile.TemporaryDirectory() as checkpoint:
    GPT2LMHeadModel(config).eval().save_pretrained(
        checkpoint
    )

    reference = GPT2LMHeadModel.from_pretrained(
        checkpoint,
        attn_implementation="eager",
    ).eval()

    adapter = load_plugin(
        "experiments.v3_gpt2_extension:GPT2Adapter",
        "model",
    )

    adapter.validate_partition(specs)

    stages = [
        adapter.create_stage(
            spec,
            checkpoint=checkpoint,
            device="cpu",
            dtype="float32",
        )
        for spec in specs
    ]

    for stage in stages:
        install_attention_extensions(
            adapter, stage, plan, config.to_dict()
        )

    print(
        "PASS: Manifest-driven GPT-2 stages initialized",
        flush=True,
    )

    # Retain references so cache object identities can
    # be checked even after requests complete.
    created_caches = {}

    def make_stage_executor(rank):
        stage = stages[rank]

        def execute(task, cache):
            key = (rank, task.request_id)

            if cache is None:
                assert key not in created_caches

                cache = DynamicCache(config=config)
                created_caches[key] = cache
            else:
                assert created_caches[key] is cache

            payload = task.payload

            # Initial request payload: [1, prompt_length]
            # Subsequent payload: one sampled token.
            if rank == 0 and isinstance(payload, int):
                payload = torch.tensor(
                    [[payload]], dtype=torch.long
                )

            with torch.inference_mode():
                output = adapter.forward_stage(
                    stage,
                    StageInputs(
                        payload=payload,
                        positions=task.position,
                        cache_state=cache,
                        request_id=str(task.request_id),
                    ),
                )

            result_payload = (
                output.logits
                if output.logits is not None
                else output.hidden_states
            )

            return StageResult(
                payload=result_payload,
                cache_state=output.cache_state,
            )

        return execute

    def sample(logits):
        return int(
            logits[0, -1, :].argmax().item()
        )

    @torch.inference_mode()
    def reference_generate(prompt):
        sequence = torch.tensor(
            [prompt], dtype=torch.long
        )

        generated = []

        for _ in range(max_new_tokens):
            logits = reference(
                sequence,
                use_cache=False,
            ).logits

            token = int(
                logits[0, -1, :].argmax().item()
            )

            generated.append(token)

            sequence = torch.cat(
                [
                    sequence,
                    torch.tensor(
                        [[token]], dtype=torch.long
                    ),
                ],
                dim=1,
            )

        return tuple(generated)

    expected = [
        reference_generate(p) for p in prompts
    ]

    with PipelineScheduler(
        [
            make_stage_executor(0),
            make_stage_executor(1),
        ],
        sample_fn=sample,
        max_active_requests=3,
    ) as scheduler:

        futures = [
            scheduler.submit(
                torch.tensor(
                    [prompt],
                    dtype=torch.long,
                ),
                prompt_length=len(prompt),
                max_new_tokens=max_new_tokens,
            )
            for prompt in prompts
        ]

        results = [
            future.result(timeout=30)
            for future in futures
        ]

        snapshot = scheduler.snapshot()

    for i, (actual, target) in enumerate(
        zip(results, expected)
    ):
        assert actual.tokens == target, (
            f"Request {i}: "
            f"actual={actual.tokens}, "
            f"expected={target}"
        )

        print(
            f"PASS: Request {i + 1} tokens "
            f"match unsplit GPT-2: {actual.tokens}",
            flush=True,
        )

    # Three requests x two stages = six distinct caches.
    assert len(created_caches) == 6

    assert len({
        id(cache)
        for cache in created_caches.values()
    }) == 6

    assert snapshot["active_requests"] == 0
    assert snapshot["cache_entries_by_stage"] == (0, 0)

    print(
        "PASS: Six independent stage-local KV caches",
        flush=True,
    )

    expected_calls = len(prompts) * max_new_tokens

    assert (
        stages[0].blocks["1"].attn.calls
        == expected_calls
    )
    assert (
        stages[1].blocks["2"].attn.calls
        == expected_calls
    )

    print(
        "PASS: Attention plugins executed "
        "on all scheduled inference steps",
        flush=True,
    )

    events = snapshot["events"]

    print("\nFirst scheduler events:")
    for event in list(events)[:12]:
        print(
            f"  stage={event.stage}, "
            f"request={event.request_id}, "
            f"step={event.step}, "
            f"event={event.event}"
        )

print(
    "\nPASS: V3.2.2 real-model scheduler integration",
    flush=True,
)
