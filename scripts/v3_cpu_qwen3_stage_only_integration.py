"""Real Qwen3 through generic factory, scheduler and CPU processes."""

import os
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_execution_factory import GenericStageFactory
from llm_runtime.v3_pipeline_scheduler import PipelineScheduler
from llm_runtime.v3_process_transport import ProcessStageTransport


PROMPTS = [
    [4, 12, 15, 31, 8, 26],
    [9, 18, 7, 21],
    [3, 16, 27, 5, 11],
    [20, 6, 13, 17, 2, 10],
]

STEPS = 4


def reference_generate(model, prompt):
    sequence = torch.tensor(
        [prompt], dtype=torch.long
    )

    tokens = []

    with torch.inference_mode():
        for _ in range(STEPS):
            logits = model(
                sequence, use_cache=False
            ).logits

            token = int(
                logits[0, -1, :].argmax().item()
            )

            tokens.append(token)

            sequence = torch.cat(
                [
                    sequence,
                    torch.tensor(
                        [[token]], dtype=torch.long
                    ),
                ],
                dim=1,
            )

    return tuple(tokens)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=True,
    )

    config._attn_implementation = "eager"

    with tempfile.TemporaryDirectory() as checkpoint:
        Qwen3ForCausalLM(config).eval().save_pretrained(
            checkpoint
        )

        reference = Qwen3ForCausalLM.from_pretrained(
            checkpoint,
            attn_implementation="eager",
        ).eval()

        expected = [
            reference_generate(reference, prompt)
            for prompt in PROMPTS
        ]

        del reference

        factory = GenericStageFactory(
            factory_path=(
                "experiments.v3_qwen3_stage_only_factory:"
                "Qwen3StageOnlyFactory"
            ),
            kwargs={
                "checkpoint_dir": checkpoint,
                "device": "cpu",
            },
        )

        with ProcessStageTransport(
            factory,
            num_stages=2,
            timeout_s=45,
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                lambda logits: int(
                    logits[0, -1, :].argmax().item()
                ),
                max_active_requests=4,
                on_request_end=transport.release_request,
            ) as scheduler:

                futures = [
                    scheduler.submit(
                        torch.tensor(
                            [prompt], dtype=torch.long
                        ),
                        prompt_length=len(prompt),
                        max_new_tokens=STEPS,
                        timeout_s=60,
                    )
                    for prompt in PROMPTS
                ]

                results = [
                    future.result(timeout=65)
                    for future in futures
                ]

                metrics = scheduler.metrics()

            stats = transport.stats()

        for i, (actual, target) in enumerate(
            zip(results, expected), 1
        ):
            assert actual.tokens == target, (
                i, actual.tokens, target
            )

            print(
                f"PASS: Partitioned Qwen3 request {i} "
                f"matches reference: {actual.tokens}",
                flush=True,
            )

        assert metrics["healthy"]
        assert metrics["completed_requests"] == 4
        assert metrics["generated_tokens"] == 16

        assert len(stats) == 2
        assert stats[0]["pid"] != stats[1]["pid"]
        assert all(
            item["pid"] != os.getpid()
            for item in stats
        )

        for rank, item in enumerate(stats):
            assert item["rank"] == rank
            assert item["runs"] == 16
            assert item["cache_entries"] == 0
            assert item["releases"] == 4

            for rid in range(1, 5):
                trace = [
                    entry
                    for entry in item["trace"]
                    if entry["request_id"] == rid
                ]

                assert [
                    entry["step"] for entry in trace
                ] == [0, 1, 2, 3]

                assert [
                    entry["cache_present"]
                    for entry in trace
                ] == [False, True, True, True]

        print(
            "PASS: Qwen3 stages executed in separate processes",
            flush=True,
        )

        print(
            "PASS: Eight independent stage-local KV caches",
            flush=True,
        )

        print(
            "PASS: Four request caches released on both stages",
            flush=True,
        )

    print(
        "PASS: V3.3.4 Qwen3 stage-only CPU inference",
        flush=True,
    )


if __name__ == "__main__":
    main()
