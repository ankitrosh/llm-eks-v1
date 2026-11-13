"""Real Qwen3-14B V3.2 stage execution over two NCCL GPUs."""

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


def as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, PrefillHidden):
        for name in ("hidden", "payload", "tensor", "states"):
            result = getattr(value, name, None)
            if isinstance(result, torch.Tensor):
                return result

        if is_dataclass(value):
            for field in fields(value):
                result = getattr(value, field.name)
                if isinstance(result, torch.Tensor):
                    return result

    raise TypeError(f"Unexpected payload: {type(value)}")


def as_logits(value, batch, vocab):
    result = as_tensor(value)

    if result.ndim == 3:
        result = result[:, -1, :]

    assert result.shape == (batch, vocab), result.shape

    return result.float().contiguous()


def main():
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2

    requests = int(os.environ.get("V32_REQUESTS", "1"))
    prompt_len = int(os.environ.get("V32_PROMPT_TOKENS", "16"))
    new_tokens = int(os.environ.get("V32_NEW_TOKENS", "8"))

    assert 1 <= requests <= 4
    assert 1 <= prompt_len <= 128
    assert 1 <= new_tokens <= 64

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    print(
        f"V32_START rank={rank} "
        f"gpu={torch.cuda.get_device_name(0)} "
        f"requests={requests}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=300),
    )

    try:
        start = time.perf_counter()

        stage = Qwen3NRankStageOnlyFactory(
            checkpoint_dir="/models/qwen3-14b",
            layer_boundaries=(0, 20, 40),
            device="cuda:0",
            dtype="bfloat16",
        )(rank)

        torch.cuda.synchronize()

        print(
            f"V32_STAGE_LOADED rank={rank} "
            f"layers={stage.start}..{stage.end - 1} "
            f"load_seconds={time.perf_counter() - start:.3f} "
            f"gpu_gib={torch.cuda.memory_allocated()/2**30:.3f}",
            flush=True,
        )

        config = stage.config
        block_size = 16
        max_context = 256

        cache = BlockKVCache(
            layers=len(stage.model.owned_layers),
            capacity_blocks=requests * (max_context // block_size),
            block_size=block_size,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_context_tokens=max_context,
            dtype=torch.bfloat16,
            device="cuda:0",
        )

        request_ids = tuple(range(1, requests + 1))
        positions = (0,) * requests
        lengths = (prompt_len,) * requests

        hidden_shape = (
            requests, prompt_len, config.hidden_size
        )
        decode_shape = (
            requests, 1, config.hidden_size
        )
        logits_shape = (
            requests, config.vocab_size
        )

        dist.barrier(device_ids=[0])
        inference_start = time.perf_counter()

        # PREFILL
        if rank == 0:
            token_ids = (
                torch.arange(
                    prompt_len, device=device
                )[None, :]
                + torch.arange(
                    requests, device=device
                )[:, None] * 29
            )

            token_ids = (
                token_ids % min(config.vocab_size - 51, 4096)
                + 50
            ).long().contiguous()

            prefill_sequences = tuple(
                tuple(int(token) for token in sequence)
                for sequence in token_ids.cpu().tolist()
            )

            result = stage_block_prefill(
                stage,
                cache,
                request_ids,
                positions,
                prefill_sequences,
            )

            hidden = as_tensor(result).contiguous()

            assert hidden.shape == hidden_shape
            assert hidden.dtype == torch.bfloat16

            dist.send(hidden, dst=1)

            logits = torch.empty(
                logits_shape,
                dtype=torch.float32,
                device=device,
            )
            dist.recv(logits, src=1)

            next_tokens = logits.argmax(dim=-1)

            print("V32_PREFILL_OK rank=0", flush=True)

        else:
            hidden = torch.empty(
                hidden_shape,
                dtype=torch.bfloat16,
                device=device,
            )
            dist.recv(hidden, src=0)

            result = stage_block_prefill(
                stage,
                cache,
                request_ids,
                positions,
                PrefillHidden(hidden, lengths),
            )

            dist.send(
                as_logits(
                    result,
                    requests,
                    config.vocab_size,
                ),
                dst=0,
            )

            print("V32_PREFILL_OK rank=1", flush=True)

        # AUTOREGRESSIVE DECODE
        generated = []

        for step in range(new_tokens):
            if rank == 0:
                result = stage_block_decode(
                    stage,
                    cache,
                    request_ids,
                    tuple(
                        int(token)
                        for token in next_tokens.cpu().tolist()
                    ),
                )

                hidden = as_tensor(result).contiguous()
                assert hidden.shape == decode_shape

                dist.send(hidden, dst=1)

                logits = torch.empty(
                    logits_shape,
                    dtype=torch.float32,
                    device=device,
                )
                dist.recv(logits, src=1)

                generated.append(next_tokens.tolist())
                next_tokens = logits.argmax(dim=-1)

            else:
                hidden = torch.empty(
                    decode_shape,
                    dtype=torch.bfloat16,
                    device=device,
                )
                dist.recv(hidden, src=0)

                result = stage_block_decode(
                    stage, cache, request_ids, hidden,
                )

                dist.send(
                    as_logits(
                        result,
                        requests,
                        config.vocab_size,
                    ),
                    dst=0,
                )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - inference_start

        print(
            f"V32_DECODE_OK rank={rank} "
            f"kv={cache.snapshot()}",
            flush=True,
        )

        for request_id in request_ids:
            assert cache.release_request(request_id)

        dist.barrier(device_ids=[0])

        if rank == 0:
            throughput = requests * new_tokens / elapsed

            print(
                f"V32_METRICS requests={requests} "
                f"input={prompt_len} output={new_tokens} "
                f"seconds={elapsed:.4f} "
                f"aggregate_tps={throughput:.4f}",
                flush=True,
            )
            print(
                f"V32_GENERATED_TOKENS={generated}",
                flush=True,
            )
            print(
                "PASS: V3.2 real Qwen3-14B "
                "20/20 BF16 NCCL block-KV inference",
                flush=True,
            )
        else:
            print(
                "PASS: V3.2 rank1 CUDA execution",
                flush=True,
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
