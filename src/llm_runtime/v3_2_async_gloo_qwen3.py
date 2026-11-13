"""Planner-driven asynchronous multi-process Qwen3 reference.

Rank 0:
  - owns the continuous batch planner and first model stage
  - runs stage 0 locally
  - asynchronously forwards hidden states
  - acknowledges results and updates the planner

Ranks 1..N-1:
  - own their model partitions and KV caches
  - process ordered microbatches independently

Completed KV is retained until shutdown for parity
inspection. Explicit distributed release is a later gate.
"""

from pathlib import Path

import torch
import torch.distributed as dist

from llm_runtime.v3_2_async_gloo import (
    AsyncGlooCoordinator,
    run_async_gloo_stage,
)
from llm_runtime.v3_2_concurrent_planner import (
    ConcurrentBatchPlanner,
)
from llm_runtime.v3_2_gloo_qwen3_smoke import (
    _create_stage,
    _execute_remote,
    _save_cache_and_release,
)
from llm_runtime.v3_2_gloo_wire import WireMicrobatch
from llm_runtime.v3_2_qwen3_stage_batch import (
    stage_block_decode,
)
from llm_runtime.v3_2_qwen3_stage_prefill import (
    PrefillHidden,
    stage_block_prefill,
)


def _planner_config(active, max_batch_tokens):
    return {
        "pipeline": {
            "memory": {
                "max_active_requests": active,
                "max_total_tokens_per_request": 32,
            },
        },
        "scheduler": {
            "mode": "continuous",
            "policy": "fcfs",
            "max_requests_per_batch": 1,
            "max_batch_tokens": max_batch_tokens,
        },
    }


def _stage_zero_execute(stage, cache, batch):
    plan = batch.plan

    ids = tuple(
        item.request_id for item in plan.items
    )
    positions = tuple(
        item.position for item in plan.items
    )
    lengths = tuple(
        len(item.tokens) for item in plan.items
    )

    actual_positions = tuple(
        cache._committed.get(rid, 0)
        for rid in ids
    )

    if actual_positions != positions:
        raise RuntimeError(
            "Stage 0 planner/KV position mismatch"
        )

    if plan.phase == "prefill":
        output = stage_block_prefill(
            stage,
            cache,
            ids,
            positions,
            tuple(
                tuple(item.tokens)
                for item in plan.items
            ),
        )

        if not isinstance(output, PrefillHidden):
            raise RuntimeError(
                "Expected intermediate prefill hidden states"
            )

        payload = output.hidden

    elif plan.phase == "decode":
        if any(length != 1 for length in lengths):
            raise RuntimeError("Invalid decode batch")

        payload = stage_block_decode(
            stage,
            cache,
            ids,
            tuple(
                item.tokens[0]
                for item in plan.items
            ),
        )

    else:
        raise RuntimeError("Unknown planner phase")

    if (
        not isinstance(payload, torch.Tensor)
        or payload.ndim != 3
    ):
        raise RuntimeError(
            "Rank 0 must forward [B,T,H] hidden states"
        )

    return WireMicrobatch(
        sequence=0,  # Assigned by the coordinator caller.
        phase=plan.phase,
        request_ids=ids,
        positions=positions,
        lengths=lengths,
        payload=payload.detach().contiguous(),
    )


def _run_rank_zero(
    stage,
    cache,
    prompts,
    *,
    max_new_tokens,
    max_batch_tokens,
    markers,
    output_dir,
):
    ids = tuple(sorted(prompts))

    planner = ConcurrentBatchPlanner(
        _planner_config(
            len(ids),
            max_batch_tokens,
        )
    )

    for rid in ids:
        planner.submit(
            rid,
            tuple(prompts[rid]),
            max_new_tokens=max_new_tokens,
        )

    pending = {}
    released = set()

    emitted_logits = {rid: [] for rid in ids}
    generated_tokens = {rid: [] for rid in ids}

    sequence = 0
    peak_pending = 0
    phases = []

    with AsyncGlooCoordinator(
        world_size=stage.layout.world_size,
        max_pending=2,
    ) as coordinator:

        for _ in range(512):
            # Admit up to two independent planner batches.
            while len(pending) < 2:
                batch = planner.next_concurrent_batch()

                if batch is None:
                    break

                next_sequence = sequence + 1

                # Prove Rank 0 advances stage 0 while Rank 1
                # is still executing the preceding batch.
                if markers is not None and next_sequence == 2:
                    if not markers[0].wait(20):
                        raise TimeoutError(
                            "Rank 1 did not begin first batch"
                        )

                packet = _stage_zero_execute(
                    stage, cache, batch
                )

                if markers is not None and next_sequence == 2:
                    markers[1].set()

                packet = WireMicrobatch(
                    next_sequence,
                    packet.phase,
                    packet.request_ids,
                    packet.positions,
                    packet.lengths,
                    packet.payload,
                )

                coordinator.submit(packet)

                pending[next_sequence] = batch
                sequence = next_sequence
                phases.append(packet.phase)

                peak_pending = max(
                    peak_pending, len(pending)
                )

            if not pending:
                if set(ids) == released:
                    break

                raise RuntimeError(
                    "Planner stalled with unfinished requests"
                )

            # The wire currently guarantees FIFO results.
            # Acknowledgement frees ownership for that batch.
            oldest = next(iter(pending))

            result = coordinator.wait(
                oldest, timeout=30
            )

            batch = pending.pop(oldest)
            plan = batch.plan

            if (
                result is None
                or result.payload.ndim != 2
                or result.payload.shape[0]
                != len(plan.items)
            ):
                raise RuntimeError(
                    "Invalid distributed Qwen3 logits"
                )

            samples = {}

            for index, item in enumerate(plan.items):
                rid = item.request_id

                if item.emits_token:
                    logits = (
                        result.payload[index]
                        .detach()
                        .cpu()
                        .clone()
                    )

                    token = int(logits.argmax())
                    samples[rid] = token

                    emitted_logits[rid].append(logits)
                    generated_tokens[rid].append(token)

                else:
                    samples[rid] = None

            planner.complete_concurrent_batch(
                batch.batch_id,
                samples,
                stop_ids=(),
            )

            # Record completed IDs. Do not physically free
            # distributed KV until the final parity snapshot.
            released.update(
                planner.drain_releases()
            )

        else:
            raise RuntimeError(
                "Distributed planner exceeded iteration limit"
            )

        state = coordinator.snapshot()

        if (
            state["pending_sequences"]
            or state["failed"]
        ):
            raise RuntimeError(
                "Distributed coordinator not fully drained"
            )

        if released != set(ids):
            raise RuntimeError(
                "Not all requests finished before release"
            )

        # The release traverses every remote rank in FIFO
        # order. ACK proves the final rank completed cleanup.
        release_sequence = coordinator.submit_release(ids)

        acknowledged = coordinator.wait_release(
            release_sequence,
            timeout=30,
        )

        if acknowledged != ids:
            raise RuntimeError(
                "Distributed release ACK mismatch"
            )

        # Rank 0 has no outstanding local stage execution.
        # Preserve KV for parity, then free its blocks.
        _save_cache_and_release(
            stage, cache, ids, output_dir
        )

    if released != set(ids):
        raise RuntimeError(
            "Not all requests reached completion"
        )

    return {
        "emitted_logits": {
            rid: tuple(values)
            for rid, values in emitted_logits.items()
        },
        "generated_tokens": {
            rid: tuple(values)
            for rid, values in generated_tokens.items()
        },
        "completed_sequences": sequence,
        "peak_pending": peak_pending,
        "phases": tuple(phases),
        "finished_request_ids": tuple(sorted(released)),
        "released_request_ids": ids,
        "release_acknowledged": True,
    }


def run_async_gloo_qwen3_rank(
    *,
    rank,
    checkpoint,
    boundaries,
    prompts,
    max_new_tokens,
    max_batch_tokens,
    output_dir,
    markers=None,
):
    """Execute within an initialized distributed process group."""
    if (
        not dist.is_initialized()
        or dist.get_rank() != rank
        or dist.get_world_size() != len(boundaries) - 1
    ):
        raise RuntimeError(
            "Invalid distributed rank/layout"
        )

    torch.set_num_threads(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage, cache = _create_stage(
        checkpoint, boundaries, rank
    )

    with torch.inference_mode():
        if rank == 0:
            result = _run_rank_zero(
                stage,
                cache,
                prompts,
                max_new_tokens=max_new_tokens,
                max_batch_tokens=max_batch_tokens,
                markers=markers,
                output_dir=output_dir,
            )

            torch.save(
                result,
                output_dir / "async_results.pt",
            )

        else:
            def execute(packet):
                # Verify the remote planner position against
                # this rank's own stage-local cache.
                actual = tuple(
                    cache._committed.get(rid, 0)
                    for rid in packet.request_ids
                )

                if actual != packet.positions:
                    raise RuntimeError(
                        "Remote stage KV position mismatch"
                    )

                if markers is not None:
                    rank1_first, rank0_second, (
                        rank2_first
                    ), rank1_second = markers

                    if rank == 1 and packet.sequence == 1:
                        rank1_first.set()

                        if not rank0_second.wait(20):
                            raise TimeoutError(
                                "Rank 0 did not overlap Rank 1"
                            )

                    if rank == 2 and packet.sequence == 1:
                        rank2_first.set()

                        if not rank1_second.wait(20):
                            raise TimeoutError(
                                "Rank 1 did not overlap Rank 2"
                            )

                    if rank == 1 and packet.sequence == 2:
                        if stage.layout.world_size >= 3:
                            if not rank2_first.wait(20):
                                raise TimeoutError(
                                    "Rank 2 did not start"
                                )

                            rank1_second.set()

                output = _execute_remote(
                    stage, cache, packet
                )

                if isinstance(output, PrefillHidden):
                    return output.hidden.contiguous()

                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        "Invalid Qwen3 stage output"
                    )

                return output.contiguous()

            run_async_gloo_stage(
                rank,
                stage.layout.world_size,
                execute,
                release=lambda ids: _save_cache_and_release(
                    stage, cache, ids, output_dir
                ),
            )

        # KV was released on the explicit RELEASE message,
        # not during process-group teardown.

    dist.barrier()
