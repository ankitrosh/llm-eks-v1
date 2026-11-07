"""Four-request GPT-2 inference over two NCCL ranks."""

import hashlib
import os
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel

from llm_runtime.v3_execution_factory import GenericStageFactory
from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError, PipelineScheduler, StageResult, StageTask,
)
from llm_runtime.v3_ordered_rank_transport import (
    OrderedRank0Transport, TransportFailure, open_control, run_rank1,
)

PROMPTS = [
    [4, 12, 15, 31, 8, 26],
    [9, 18, 7, 21],
    [3, 16, 27, 5, 11],
    [20, 6, 13, 17, 2, 10],
]
OUTPUT_TOKENS = 4


def reference_generate(model, prompt):
    sequence = torch.tensor(
        [prompt], dtype=torch.long, device="cuda:0"
    )
    generated = []

    with torch.inference_mode():
        for _ in range(OUTPUT_TOKENS):
            logits = model(sequence, use_cache=False).logits
            token = int(logits[0, -1, :].argmax().item())
            generated.append(token)
            sequence = torch.cat(
                (
                    sequence,
                    torch.tensor(
                        [[token]], dtype=torch.long, device="cuda:0"
                    ),
                ),
                dim=1,
            )

    return tuple(generated)


def model_fingerprint(model):
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(
            tensor.detach().contiguous().cpu().numpy().tobytes()
        )
    return hasher.digest()




def main():
    rank = int(os.environ["RANK"])
    assert rank in (0, 1)
    assert int(os.environ["WORLD_SIZE"]) == 2

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.set_num_threads(2)

    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=180)
    )

    try:
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

        with tempfile.TemporaryDirectory() as checkpoint:
            source_model = GPT2LMHeadModel(config).eval()

            fingerprint = model_fingerprint(source_model)
            fp_tensor = torch.tensor(
                list(fingerprint),
                dtype=torch.uint8,
                device=device,
            )
            peer_fingerprints = [
                torch.empty_like(fp_tensor) for _ in range(2)
            ]
            dist.all_gather(peer_fingerprints, fp_tensor)

            assert torch.equal(
                peer_fingerprints[0], peer_fingerprints[1]
            ), "Ranks initialized different checkpoint weights"

            print(
                f"PASS: Rank {rank} model fingerprint "
                f"{fingerprint.hex()[:16]}",
                flush=True,
            )

            source_model.save_pretrained(checkpoint)
            del source_model

            stage = GenericStageFactory(
                factory_path=(
                    "experiments.v3_gpt2_process_factory:"
                    "GPT2ProcessFactory"
                ),
                kwargs={
                    "checkpoint_dir": checkpoint,
                    "manifest_path": (
                        "/workspace/examples/"
                        "v3_gpt2_extension_nccl.yaml"
                    ),
                    "device": "cuda:0",
                    "dtype": "float32",
                },
            )(rank)

            expected = None
            if rank == 0:
                reference = GPT2LMHeadModel.from_pretrained(
                    checkpoint, attn_implementation="eager"
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
                int(os.environ.get("V3_CTRL_PORT", "29639")),
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
                    conn, device=device, execute=execute
                )

                assert stage.forward_calls == 16
                assert stats["runs"] == 16
                assert stats["releases"] == 4

                for rid in range(1, 5):
                    steps = [
                        step
                        for request, step, _, _ in stats["trace"]
                        if request == rid
                    ]
                    assert steps == [0, 1, 2, 3], (rid, steps)

                print(
                    "PASS: Rank 1 executed 16 GPT-2 forwards",
                    flush=True,
                )
                print(
                    "PASS: Rank 1 released four request caches",
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
                        raise FatalStageError(str(exc)) from exc

                    return StageResult(
                        ticket, result.cache_state
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
                        raise FatalStageError(str(exc)) from exc

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
                                max_new_tokens=OUTPUT_TOKENS,
                                timeout_s=150,
                            )
                            for prompt in PROMPTS
                        ]

                        results = [
                            future.result(timeout=160)
                            for future in futures
                        ]

                        for i, (result, target) in enumerate(
                            zip(results, expected)
                        ):
                            assert result.tokens == target, (
                                i, result.tokens, target
                            )

                            print(
                                f"PASS: Request {i + 1} GPT-2 "
                                f"tokens match unsplit reference: "
                                f"{result.tokens}",
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
                            "PASS: 16 ordered GPT-2 NCCL "
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
                    "PASS: V3.2.5d real-model concurrent "
                    "NCCL inference",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
