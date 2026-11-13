"""V3.3.3c: Qwen3 inference through the generic two-rank NCCL pipeline."""

import hashlib
import os
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_execution_factory import GenericStageFactory
from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError,
    PipelineScheduler,
    StageResult,
    StageTask,
)
from llm_runtime.v3_ordered_rank_transport import (
    OrderedRank0Transport,
    TransportFailure,
    open_control,
    run_rank1,
)


PROMPTS = [
    [4, 12, 15, 31, 8, 26],
    [9, 18, 7, 21],
    [3, 16, 27, 5, 11],
    [20, 6, 13, 17, 2, 10],
]
STEPS = 4


def fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(
            tensor.detach().contiguous().cpu().numpy().tobytes()
        )
    return h.digest()


def reference_generate(model, prompt):
    sequence = torch.tensor(
        [prompt], dtype=torch.long, device="cuda:0"
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
                        [[token]], dtype=torch.long,
                        device="cuda:0"
                    ),
                ],
                dim=1,
            )

    return tuple(tokens)


def main():
    rank = int(os.environ["RANK"])
    assert rank in (0, 1)
    assert int(os.environ["WORLD_SIZE"]) == 2
    assert torch.cuda.is_available()

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=180),
    )

    try:
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
            model = Qwen3ForCausalLM(config).eval()

            fp = torch.tensor(
                list(fingerprint(model)),
                dtype=torch.uint8,
                device=device,
            )

            peer = [
                torch.empty_like(fp)
                for _ in range(2)
            ]

            dist.all_gather(peer, fp)

            assert torch.equal(
                peer[0], peer[1]
            ), "Rank model weights differ"

            print(
                f"PASS: Rank {rank} Qwen3 fingerprint "
                f"{bytes(fp.cpu().tolist()).hex()[:16]}",
                flush=True,
            )

            model.save_pretrained(checkpoint)
            del model

            stage = GenericStageFactory(
                factory_path=(
                    "experiments.v3_qwen3_process_factory:"
                    "Qwen3ProcessFactory"
                ),
                kwargs={
                    "checkpoint_dir": checkpoint,
                    "device": "cuda:0",
                },
            )(rank)

            expected = None

            if rank == 0:
                reference = Qwen3ForCausalLM.from_pretrained(
                    checkpoint,
                    attn_implementation="eager",
                ).to(device).eval()

                expected = [
                    reference_generate(reference, prompt)
                    for prompt in PROMPTS
                ]

                del reference
                torch.cuda.empty_cache()

            dist.barrier(device_ids=[0])

            conn = open_control(
                rank,
                os.environ["MASTER_ADDR"],
                int(os.environ.get("V3_CTRL_PORT", "29649")),
                timeout_s=90,
            )

            if rank == 1:

                def execute(rid, step, position, hidden, cache):
                    result = stage(
                        StageTask(rid, step, position, hidden),
                        cache,
                    )
                    return result.payload, result.cache_state

                stats = run_rank1(
                    conn,
                    device=device,
                    execute=execute,
                )

                assert stage.forward_calls == 16
                assert stats["runs"] == 16
                assert stats["releases"] == 4

                for rid in range(1, 5):
                    observed = [
                        step
                        for req, step, _, _ in stats["trace"]
                        if req == rid
                    ]
                    assert observed == [0, 1, 2, 3]

                print(
                    "PASS: Rank 1 completed 16 Qwen3 forwards",
                    flush=True,
                )
                print(
                    "PASS: Rank 1 released four Qwen3 caches",
                    flush=True,
                )

            else:
                link = OrderedRank0Transport(
                    conn,
                    device=device,
                    max_pending=4,
                    timeout_s=90,
                )

                def first(task, cache):
                    result = stage(task, cache)

                    try:
                        ticket = link.publish(
                            task.request_id,
                            task.step,
                            task.position,
                            result.payload,
                        )
                    except TransportFailure as exc:
                        raise FatalStageError(
                            str(exc)
                        ) from exc

                    return StageResult(
                        ticket,
                        result.cache_state,
                    )

                def second(task, cache):
                    ticket = task.payload

                    assert (
                        ticket.request_id,
                        ticket.step,
                        ticket.position,
                    ) == (
                        task.request_id,
                        task.step,
                        task.position,
                    )

                    try:
                        logits = link.await_result(ticket)
                    except TransportFailure as exc:
                        raise FatalStageError(
                            str(exc)
                        ) from exc

                    return StageResult(logits, cache)

                try:
                    with PipelineScheduler(
                        [first, second],
                        lambda logits: int(
                            logits[0, -1, :].argmax().item()
                        ),
                        max_active_requests=4,
                        on_request_end=link.release_request,
                    ) as scheduler:

                        futures = [
                            scheduler.submit(
                                torch.tensor(
                                    [prompt],
                                    dtype=torch.long,
                                    device=device,
                                ),
                                prompt_length=len(prompt),
                                max_new_tokens=STEPS,
                                timeout_s=150,
                            )
                            for prompt in PROMPTS
                        ]

                        results = [
                            f.result(timeout=160)
                            for f in futures
                        ]

                        for i, (actual, target) in enumerate(
                            zip(results, expected), 1
                        ):
                            assert actual.tokens == target, (
                                i, actual.tokens, target
                            )

                            print(
                                f"PASS: Qwen3 NCCL request {i} "
                                f"matches reference: {actual.tokens}",
                                flush=True,
                            )

                        metrics = scheduler.metrics()

                        assert metrics["healthy"]
                        assert metrics["completed_requests"] == 4
                        assert metrics["generated_tokens"] == 16
                        assert stage.forward_calls == 16

                        assert link.stats()["sent"] == 16
                        assert link.stats()["received"] == 16

                        print(
                            "PASS: 16 ordered Qwen3 NCCL "
                            "activation/reply pairs",
                            flush=True,
                        )
                        print(
                            "Metrics:",
                            {
                                key: metrics.get(key)
                                for key in (
                                    "completed_requests",
                                    "generated_tokens",
                                    "tokens_per_second",
                                    "avg_ttft_ms",
                                    "avg_latency_ms",
                                )
                            },
                            flush=True,
                        )

                finally:
                    link.close()

                print(
                    "PASS: V3.3.3c generic Qwen3 NCCL inference",
                    flush=True,
                )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
