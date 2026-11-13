"""V3.2 A+B: V3 forward-only pipeline + token-ID feedback.

Dynamic admission and mixed batching belong to phase D.
"""
import os
import time
from dataclasses import fields, is_dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)
from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_qwen3_stage_prefill import (
    PrefillHidden,
    stage_block_prefill,
)
from llm_runtime.v3_2_qwen3_stage_batch import stage_block_decode
from llm_runtime.v3_pipeline_scheduler import (
    FatalStageError,
    PipelineScheduler,
    StageResult,
    StageTask,
)
from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    NRankTransportError,
    TorchP2PBackend,
    open_nrank_control,
    run_nrank_worker,
)


def as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, PrefillHidden):
        for key in ("hidden", "payload", "tensor", "states"):
            candidate = getattr(value, key, None)
            if isinstance(candidate, torch.Tensor):
                return candidate

        if is_dataclass(value):
            for field in fields(value):
                candidate = getattr(value, field.name)
                if isinstance(candidate, torch.Tensor):
                    return candidate

    raise TypeError(f"Unexpected stage output: {type(value)}")


def new_cache(stage):
    block_size = 16
    max_context = 256

    return BlockKVCache(
        layers=len(stage.model.owned_layers),
        capacity_blocks=max_context // block_size,
        block_size=block_size,
        kv_heads=stage.config.num_key_value_heads,
        head_dim=stage.config.head_dim,
        max_context_tokens=max_context,
        device="cuda:0",
        dtype=torch.bfloat16,
    )


def execute_stage(stage, task, cache):
    """Execute one request on this GPU using V3.2 block KV."""
    rid = task.request_id

    if task.step == 0:
        if cache is not None or task.position != 0:
            raise RuntimeError("Prefill requires a fresh cache")

        cache = new_cache(stage)

        if stage.rank == 0:
            prompt = tuple(int(t) for t in task.payload)

            out = stage_block_prefill(
                stage,
                cache,
                (rid,),
                (0,),
                (prompt,),
            )

        else:
            hidden = task.payload

            assert isinstance(hidden, torch.Tensor)
            assert hidden.ndim == 3
            assert hidden.shape[0] == 1

            out = stage_block_prefill(
                stage,
                cache,
                (rid,),
                (0,),
                PrefillHidden(
                    hidden,
                    (hidden.shape[1],),
                ),
            )

    else:
        if cache is None:
            raise RuntimeError("Missing decode KV cache")

        if stage.rank == 0:
            payload = (int(task.payload),)
        else:
            payload = task.payload

        out = stage_block_decode(
            stage,
            cache,
            (rid,),
            payload,
        )

    out = as_tensor(out)

    if stage.rank == 1:
        # Rank 1 samples directly on CUDA.
        # Only return one int64 ID, never full logits.
        if out.ndim == 3:
            out = out[:, -1, :]

        assert out.ndim == 2
        assert out.shape[0] == 1

        # The frozen V3 transport accepts floating-point
        # activation tensors, not a 1D int64 token tensor.
        #
        # float32 represents Qwen3 vocabulary IDs exactly.
        # Send just one scalar in a transport-compatible shape.
        out = (
            out.argmax(dim=-1)
            .to(dtype=torch.float32)
            .reshape(1, 1, 1)
            .contiguous()
        )

    else:
        assert out.is_cuda
        assert out.dtype == torch.bfloat16
        assert out.ndim == 3
        assert out.shape[0] == 1

    return StageResult(
        payload=out,
        cache_state=cache,
    )


def make_prompts(count, length, vocab):
    modulus = min(vocab - 51, 4096)
    assert modulus > 0

    return [
        tuple(
            (token + request * 29) % modulus + 50
            for token in range(length)
        )
        for request in range(count)
    ]


def run_rank0(stage, backend):
    requests = int(os.getenv("V32_REQUESTS", "4"))
    prompt_len = int(os.getenv("V32_PROMPT_TOKENS", "128"))
    output_len = int(os.getenv("V32_NEW_TOKENS", "64"))
    warmup = int(os.getenv("AB_WARMUP_TOKENS", "4"))

    assert 1 <= requests <= 4
    assert 1 <= prompt_len <= 128
    assert 1 <= output_len <= 64
    assert 1 <= warmup <= output_len

    peers = open_nrank_control(
        rank=0,
        world_size=2,
        host=os.environ["MASTER_ADDR"],
        port=int(os.getenv("V31_CTRL_PORT", "29679")),
        timeout_s=180,
    )

    link = NRankCoordinator(
        peers,
        backend,
        max_pending=8,
        timeout_s=180,
    )

    def first(task, cache):
        result = execute_stage(stage, task, cache)

        try:
            ticket = link.publish(
                task.request_id,
                task.step,
                task.position,
                result.payload,
            )
        except NRankTransportError as exc:
            raise FatalStageError(str(exc)) from exc

        # The first-stage worker can process another request
        # while the remote GPU handles this ticket.
        return StageResult(
            ticket,
            result.cache_state,
        )

    def last(task, cache):
        try:
            token = link.await_result(task.payload)
        except NRankTransportError as exc:
            raise FatalStageError(str(exc)) from exc

        return StageResult(token, cache)

    def sample(tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.is_cuda
        assert tensor.dtype == torch.float32
        assert tuple(tensor.shape) == (1, 1, 1)
        assert tensor.numel() == 1

        # V3's Python scheduler still needs one scalar on CPU.
        # This is much smaller than transferring full logits.
        return int(tensor.item())

    try:
        with PipelineScheduler(
            [first, last],
            sample,
            max_active_requests=requests,
            on_request_end=link.release_request,
        ) as scheduler:

            # Full-shape warmup: same concurrency, prompt length
            # and generation length as the measured workload.
            warm_inputs = make_prompts(
                requests,
                prompt_len,
                stage.config.vocab_size,
            )

            warm_start = time.perf_counter()

            warm_futures = [
                scheduler.submit(
                    prompt,
                    prompt_length=len(prompt),
                    max_new_tokens=warmup,
                    timeout_s=360,
                )
                for prompt in warm_inputs
            ]

            warm_results = [
                future.result(timeout=370)
                for future in warm_futures
            ]

            assert all(
                len(result.tokens) == warmup
                for result in warm_results
            )

            torch.cuda.synchronize()

            print(
                f"AB_WARMUP_OK requests={requests} "
                f"input={prompt_len} output={warmup} "
                f"seconds={time.perf_counter() - warm_start:.4f}",
                flush=True,
            )

            inputs = make_prompts(
                requests,
                prompt_len,
                stage.config.vocab_size,
            )

            start = time.perf_counter()

            futures = [
                scheduler.submit(
                    prompt,
                    prompt_length=len(prompt),
                    max_new_tokens=output_len,
                    timeout_s=360,
                )
                for prompt in inputs
            ]

            results = [
                future.result(timeout=370)
                for future in futures
            ]

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            for index, result in enumerate(results):
                assert len(result.tokens) == output_len
                assert all(
                    isinstance(token, int)
                    for token in result.tokens
                )

                print(
                    f"AB_REQUEST_DONE index={index} "
                    f"tokens={len(result.tokens)} "
                    f"first={result.tokens[0]}",
                    flush=True,
                )

            total = requests * output_len
            metrics = scheduler.metrics()

            assert metrics["healthy"], metrics
            assert metrics["failed_requests"] == 0, metrics

            print(
                f"AB_METRICS requests={requests} "
                f"input={prompt_len} output={output_len} "
                f"seconds={elapsed:.4f} "
                f"aggregate_tps={total / elapsed:.4f}",
                flush=True,
            )

            print(
                f"AB_SCHEDULER_METRICS {metrics}",
                flush=True,
            )

            print(
                "PASS: V3.2 A+B overlapped pipeline "
                "and token-only feedback",
                flush=True,
            )

    finally:
        link.close()

    print(
        f"AB_TRANSPORT_STATS {link.stats()}",
        flush=True,
    )


def run_rank1(stage, backend):
    conn = open_nrank_control(
        rank=1,
        world_size=2,
        host=os.environ["MASTER_ADDR"],
        port=int(os.getenv("V31_CTRL_PORT", "29679")),
        timeout_s=180,
    )

    def execute(rank, rid, step, position, payload, cache):
        result = execute_stage(
            stage,
            StageTask(rid, step, position, payload),
            cache,
        )

        return result.payload, result.cache_state

    stats = run_nrank_worker(
        1,
        2,
        conn,
        backend,
        execute,
    )

    print(
        f"PASS: V3.2 A+B rank1 "
        f"{stats['runs']} forwards "
        f"{stats['releases']} releases",
        flush=True,
    )


def main():
    # V3.2 CUDA attention optimization.
    # The original module remains unchanged.
    import llm_runtime.v3_2_qwen3_stage_batch as batch_module
    from llm_runtime.v3_2_block_attention_fast import (
        block_decode_attention_fast,
    )

    batch_module.block_decode_attention = (
        block_decode_attention_fast
    )

    print("AB_ATTENTION_BACKEND=sdpa_block_gather", flush=True)
    rank = int(os.environ["RANK"])

    assert int(os.environ["WORLD_SIZE"]) == 2
    assert rank in (0, 1)
    assert torch.cuda.is_available()

    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    print(
        f"AB_START rank={rank} "
        f"gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=600),
    )

    try:
        stage = Qwen3NRankStageOnlyFactory(
            checkpoint_dir="/models/qwen3-14b",
            layer_boundaries=(0, 20, 40),
            device="cuda:0",
            dtype="bfloat16",
        )(rank)

        torch.cuda.synchronize()

        print(
            f"AB_STAGE_LOADED rank={rank} "
            f"layers={stage.start}..{stage.end - 1} "
            f"gpu_gib={torch.cuda.memory_allocated()/2**30:.3f}",
            flush=True,
        )

        dist.barrier(device_ids=[0])

        backend = TorchP2PBackend("cuda:0")

        if rank == 0:
            run_rank0(stage, backend)
        else:
            run_rank1(stage, backend)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
