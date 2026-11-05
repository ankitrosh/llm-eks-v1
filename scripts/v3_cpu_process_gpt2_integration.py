"""V3.2.5a: Actual GPT-2 inference over CPU process transport."""

import os
import tempfile
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.v3_gpt2_process_factory import GPT2ProcessFactory
from llm_runtime.v3_pipeline_scheduler import PipelineScheduler
from llm_runtime.v3_process_transport import ProcessStageTransport


def reference_generate(model, prompt, count):
    sequence = torch.tensor([prompt], dtype=torch.long)
    generated = []

    with torch.inference_mode():
        for _ in range(count):
            logits = model(
                sequence, use_cache=False
            ).logits

            token = int(
                logits[0, -1, :].argmax().item()
            )

            generated.append(token)

            sequence = torch.cat(
                (
                    sequence,
                    torch.tensor(
                        [[token]], dtype=torch.long
                    ),
                ),
                dim=1,
            )

    return tuple(generated)


def main():
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

    prompts = [
        [4, 12, 15, 31, 8, 26],
        [9, 18, 7, 21],
        [3, 16, 27, 5, 11],
        [20, 6, 13, 17, 2, 10],
    ]

    output_tokens = 4

    with tempfile.TemporaryDirectory() as checkpoint:
        GPT2LMHeadModel(config).eval().save_pretrained(
            checkpoint
        )

        reference = GPT2LMHeadModel.from_pretrained(
            checkpoint,
            attn_implementation="eager",
        ).eval()

        expected = [
            reference_generate(
                reference, prompt, output_tokens
            )
            for prompt in prompts
        ]

        factory = GPT2ProcessFactory(
            checkpoint_dir=checkpoint,
            manifest_path=str(
                Path(
                    "examples/v3_gpt2_extension.yaml"
                ).resolve()
            ),
        )

        with ProcessStageTransport(
            factory,
            num_stages=2,
            timeout_s=45,
        ) as transport:

            with PipelineScheduler(
                transport.stage_functions(),
                sample_fn=lambda logits: int(
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
                        max_new_tokens=output_tokens,
                        timeout_s=60,
                    )
                    for prompt in prompts
                ]

                results = [
                    future.result(timeout=65)
                    for future in futures
                ]

                metrics = scheduler.metrics()
                events = scheduler.snapshot()["events"]

            worker_stats = transport.stats()

        for i, (actual, target) in enumerate(
            zip(results, expected)
        ):
            assert actual.tokens == target, (
                i, actual.tokens, target
            )

            print(
                f"PASS: Request {i + 1} generated "
                f"reference-matching tokens: {actual.tokens}",
                flush=True,
            )

        assert metrics["healthy"]
        assert metrics["completed_requests"] == 4
        assert metrics["generated_tokens"] == 16
        assert metrics["active"] == 0

        print(
            "PASS: Four concurrent requests completed",
            flush=True,
        )

        # Cache storage remains inside the worker processes.
        assert len(worker_stats) == 2

        pids = [s["pid"] for s in worker_stats]

        assert len(set(pids)) == 2
        assert all(pid != os.getpid() for pid in pids)

        for rank, stats in enumerate(worker_stats):
            assert stats["rank"] == rank
            assert stats["runs"] == 16, stats
            assert stats["cache_entries"] == 0, stats
            assert stats["releases"] == 4, stats

            for rid in range(1, 5):
                trace = [
                    row for row in stats["trace"]
                    if row["request_id"] == rid
                ]

                assert [t["step"] for t in trace] == [
                    0, 1, 2, 3
                ], trace

                assert [
                    t["cache_present"] for t in trace
                ] == [
                    False, True, True, True
                ], trace

        print(
            "PASS: Separate model processes and "
            "request-local KV caches",
            flush=True,
        )

        starts = {
            (e.stage, e.request_id, e.step): e.at
            for e in events
            if e.event == "start"
        }

        finishes = {
            (e.stage, e.request_id, e.step): e.at
            for e in events
            if e.event == "finish"
        }

        assert len(starts) == 32
        assert len(finishes) == 32

        # Validate event ordering, rather than assuming
        # a particular relative execution speed.
        for rid in range(1, 5):
            for step in range(output_tokens):
                assert (
                    finishes[(0, rid, step)]
                    <= starts[(1, rid, step)]
                )

        print(
            "PASS: Request and generation-step "
            "routing preserved",
            flush=True,
        )

        print(
            "Scheduler metrics:",
            {
                "completed": metrics["completed_requests"],
                "tokens": metrics["generated_tokens"],
                "tokens_per_second": round(
                    metrics["tokens_per_second"], 2
                ),
                "avg_ttft_ms": round(
                    metrics["avg_ttft_ms"], 2
                ),
                "avg_latency_ms": round(
                    metrics["avg_latency_ms"], 2
                ),
            },
            flush=True,
        )

    print(
        "PASS: V3.2.5a real-model process transport",
        flush=True,
    )


if __name__ == "__main__":
    main()
