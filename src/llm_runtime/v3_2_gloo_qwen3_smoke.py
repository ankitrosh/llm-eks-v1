"""Blocking multi-process Qwen3 pipeline correctness reference.

Each process loads only its assigned Qwen3 partition.
Rank 0 performs embedding and sampling decisions.
The final rank sends logits to rank 0 via Gloo.

This is not the asynchronous serving runtime.
"""

from pathlib import Path

import torch
import torch.distributed as dist

from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)
from llm_runtime.v3_2_gloo_wire import (
    WireMicrobatch,
    recv_wire,
    send_wire,
)
from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_qwen3_stage_batch import (
    stage_block_decode,
)
from llm_runtime.v3_2_qwen3_stage_prefill import (
    PrefillHidden,
    stage_block_prefill,
)


def _create_stage(checkpoint, boundaries, rank):
    factory = Qwen3NRankStageOnlyFactory(
        checkpoint_dir=str(checkpoint),
        layer_boundaries=tuple(boundaries),
        device="cpu",
        dtype="float32",
    )

    stage = factory(rank)

    cache = BlockKVCache(
        layers=len(stage.model.owned_layers),
        capacity_blocks=32,
        block_size=2,
        kv_heads=stage.config.num_key_value_heads,
        head_dim=stage.config.head_dim,
        max_context_tokens=32,
        dtype=torch.float32,
    )

    return stage, cache


def _execute_remote(stage, cache, packet):
    if packet.phase == "prefill":
        return stage_block_prefill(
            stage,
            cache,
            packet.request_ids,
            packet.positions,
            PrefillHidden(
                packet.payload,
                packet.lengths,
            ),
        )

    if packet.phase == "decode":
        return stage_block_decode(
            stage,
            cache,
            packet.request_ids,
            packet.payload,
        )

    raise ValueError("Unsupported wire phase")


def _forward_output(stage, packet, output):
    if stage.rank == stage.layout.world_size - 1:
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
        ):
            raise RuntimeError("Final stage must return logits")

        kind = "result"
        destination = 0
        payload = output.contiguous()

    else:
        kind = "run"
        destination = stage.rank + 1

        if packet.phase == "prefill":
            if (
                not isinstance(output, PrefillHidden)
                or output.lengths != packet.lengths
            ):
                raise RuntimeError(
                    "Invalid forwarded prefill lengths"
                )
            payload = output.hidden.contiguous()
        else:
            payload = output.contiguous()

    send_wire(
        destination,
        kind=kind,
        packet=WireMicrobatch(
            packet.sequence,
            packet.phase,
            packet.request_ids,
            packet.positions,
            packet.lengths,
            payload,
        ),
    )


def _rank_zero_roundtrip(
    stage, cache, *,
    sequence, phase, request_ids,
    positions, values,
):
    ids = tuple(request_ids)
    positions = tuple(positions)

    if phase == "prefill":
        sequences = tuple(
            tuple(tokens) for tokens in values
        )
        lengths = tuple(map(len, sequences))

        output = stage_block_prefill(
            stage,
            cache,
            ids,
            positions,
            sequences,
        )

        if not isinstance(output, PrefillHidden):
            raise RuntimeError(
                "Rank 0 must forward hidden states"
            )

        payload = output.hidden.contiguous()

    elif phase == "decode":
        lengths = (1,) * len(ids)

        payload = stage_block_decode(
            stage,
            cache,
            ids,
            tuple(values),
        ).contiguous()

    else:
        raise ValueError("Unsupported phase")

    sent = WireMicrobatch(
        sequence,
        phase,
        ids,
        positions,
        lengths,
        payload,
    )

    send_wire(1, kind="run", packet=sent)

    kind, received = recv_wire(
        stage.layout.world_size - 1
    )

    if (
        kind != "result"
        or received.sequence != sequence
        or received.phase != phase
        or received.request_ids != ids
        or received.positions != positions
        or received.lengths != lengths
    ):
        raise RuntimeError(
            "Gloo result metadata does not match request"
        )

    return received.payload


def _run_rank_zero(stage, cache, prompts, chunk_size, steps):
    ids = tuple(sorted(prompts))
    offsets = {rid: 0 for rid in ids}
    prefill_logits = {}
    decode_logits = {rid: [] for rid in ids}

    sequence = 0

    while any(
        offsets[rid] < len(prompts[rid])
        for rid in ids
    ):
        ready = tuple(
            rid for rid in ids
            if offsets[rid] < len(prompts[rid])
        )

        positions = tuple(
            offsets[rid] for rid in ready
        )

        chunks = tuple(
            prompts[rid][
                offsets[rid]:
                offsets[rid] + (
                    chunk_size
                    if chunk_size is not None
                    else len(prompts[rid])
                )
            ]
            for rid in ready
        )

        sequence += 1

        logits = _rank_zero_roundtrip(
            stage,
            cache,
            sequence=sequence,
            phase="prefill",
            request_ids=ready,
            positions=positions,
            values=chunks,
        )

        for index, (rid, chunk) in enumerate(
            zip(ready, chunks)
        ):
            offsets[rid] += len(chunk)

            if offsets[rid] == len(prompts[rid]):
                prefill_logits[rid] = (
                    logits[index].detach().cpu().clone()
                )

    current_logits = dict(prefill_logits)

    for _ in range(steps):
        tokens = tuple(
            int(current_logits[rid].argmax())
            for rid in ids
        )

        positions = tuple(
            offsets[rid] for rid in ids
        )

        sequence += 1

        logits = _rank_zero_roundtrip(
            stage,
            cache,
            sequence=sequence,
            phase="decode",
            request_ids=ids,
            positions=positions,
            values=tokens,
        )

        for index, rid in enumerate(ids):
            result = logits[index].detach().cpu().clone()

            decode_logits[rid].append(result)
            current_logits[rid] = result
            offsets[rid] += 1

    return {
        "prefill_logits": prefill_logits,
        "decode_logits": {
            rid: tuple(values)
            for rid, values in decode_logits.items()
        },
        "completed_sequences": sequence,
    }


def _run_remote(stage, cache):
    last_sequence = 0

    while True:
        kind, packet = recv_wire(stage.rank - 1)

        if kind == "stop":
            if stage.rank + 1 < stage.layout.world_size:
                send_wire(
                    stage.rank + 1,
                    kind="stop",
                )
            break

        if (
            kind != "run"
            or packet.sequence != last_sequence + 1
        ):
            raise RuntimeError(
                "Out-of-order distributed microbatch"
            )

        last_sequence = packet.sequence

        output = _execute_remote(
            stage, cache, packet
        )

        _forward_output(stage, packet, output)


def _save_cache_and_release(stage, cache, ids, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "rank": stage.rank,
        "owned_layers": tuple(stage.model.owned_layers),
        "kv": {
            rid: tuple(
                (keys.cpu().clone(), values.cpu().clone())
                for keys, values in cache.read_request(rid)
            )
            for rid in ids
        },
        "before_release": cache.snapshot(),
    }

    for rid in ids:
        if not cache.release_request(rid):
            raise RuntimeError(
                f"Failed to release request {rid}"
            )

    state["after_release"] = cache.snapshot()

    torch.save(
        state,
        output_dir / f"rank_{stage.rank}_kv.pt",
    )


def run_gloo_qwen3_rank(
    *, rank, checkpoint, boundaries,
    prompts, chunk_size, decode_steps, output_dir,
):
    """Run inside an initialized Gloo process group."""
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed process group not initialized"
        )

    if dist.get_rank() != rank:
        raise RuntimeError("Incorrect process rank")

    if dist.get_world_size() != len(boundaries) - 1:
        raise RuntimeError("World size/layout mismatch")

    torch.set_num_threads(1)

    stage, cache = _create_stage(
        checkpoint, boundaries, rank
    )

    ids = tuple(sorted(prompts))

    with torch.inference_mode():
        if rank == 0:
            results = _run_rank_zero(
                stage,
                cache,
                prompts,
                chunk_size,
                decode_steps,
            )

            torch.save(
                results,
                Path(output_dir) / "rank_0_results.pt",
            )

            send_wire(1, kind="stop")
        else:
            _run_remote(stage, cache)

        _save_cache_and_release(
            stage, cache, ids, output_dir
        )

    # All ranks must finish their snapshots and receive STOP
    # before the process group is destroyed.
    dist.barrier()
