"""Mixed-phase continuous Qwen3 CPU pipeline.

Supports concurrent prefill and decode microbatches through
policy-controlled stage workers.

Each request has at most one outstanding plan.
Each pipeline stage serializes its local KV transactions.

This is an in-process CPU correctness runtime, not NCCL.
"""

import torch

from llm_runtime.v3_2_concurrent_nrank_runtime import (
    ConcurrentNRankCPURuntime,
)
from llm_runtime.v3_2_policy_transport import (
    PolicyCPUMicrobatchTransport,
)
from llm_runtime.v3_2_qwen3_stage_batch import (
    stage_block_decode,
    validate_stage_alignment,
)
from llm_runtime.v3_2_qwen3_stage_prefill import (
    stage_block_prefill,
)


class MixedNRankCPURuntime(ConcurrentNRankCPURuntime):
    def __init__(
        self,
        stages,
        caches,
        config,
        *,
        max_pending=2,
        batch_policy=None,
        pipeline_policy=None,
        stop_token_ids=(),
    ):
        super().__init__(
            stages,
            caches,
            config,
            max_pending=max_pending,
            batch_policy=batch_policy,
            pipeline_policy=pipeline_policy,
            stop_token_ids=stop_token_ids,
        )

        # Replace decode-only workers with phase-aware workers.
        self.transport.close()

        workers = []

        for stage, cache, guard in zip(
            self.stages, self.caches, self._stage_locks
        ):

            def execute(
                packet, stage=stage, cache=cache, guard=guard
            ):
                with guard:
                    if packet.phase == "prefill":
                        return stage_block_prefill(
                            stage,
                            cache,
                            packet.request_ids,
                            packet.positions,
                            packet.payload,
                        )

                    if packet.phase == "decode":
                        positions = tuple(
                            cache._committed.get(rid)
                            for rid in packet.request_ids
                        )

                        if positions != packet.positions:
                            raise RuntimeError(
                                "Decode KV position mismatch"
                            )

                        return stage_block_decode(
                            stage,
                            cache,
                            packet.request_ids,
                            packet.payload,
                        )

                    raise ValueError(
                        "Unknown microbatch phase"
                    )

            workers.append(execute)

        scheduling = (
            config.get("pipeline", {}).get("scheduling")
        )

        self.transport = PolicyCPUMicrobatchTransport(
            workers,
            max_pending=max_pending,
            scheduling_config=(
                config if scheduling is not None else None
            ),
            policy=pipeline_policy,
        )

        # Sequence -> immutable planner batch, retained until
        # the transport result has been acknowledged.
        self._pending_plans = {}

    def submit(self, prompt, max_new_tokens):
        """Admit any valid prompt, including chunked prefills."""
        if self._failed:
            raise RuntimeError(
                "Runtime failed; restart required"
            )

        prompt = tuple(prompt)

        if (
            not prompt
            or any(
                type(token) is not int or token < 0
                for token in prompt
            )
        ):
            raise ValueError("Invalid prompt")

        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise ValueError("Invalid output length")

        rid = self._next_id + 1

        # Planner enforces active/queue limits, request
        # context capacity and token reservations.
        self.planner.submit(
            rid,
            prompt,
            max_new_tokens=max_new_tokens,
        )

        self._next_id = rid
        self._prompts[rid] = prompt
        self._generated[rid] = []
        return rid

    def _validate_positions(self, plan):
        ids = tuple(
            item.request_id for item in plan.items
        )
        positions = tuple(
            item.position for item in plan.items
        )

        if plan.phase == "decode":
            actual = validate_stage_alignment(
                self.caches, ids
            )

            if actual != positions:
                raise RuntimeError(
                    "Decode planner/KV position mismatch"
                )

        elif plan.phase == "prefill":
            for cache in self.caches:
                actual = tuple(
                    cache._committed.get(rid, 0)
                    for rid in ids
                )

                if actual != positions:
                    raise RuntimeError(
                        "Prefill planner/KV position mismatch"
                    )

        else:
            raise RuntimeError("Unsupported planner phase")

        return ids, positions

    def dispatch_ready(self):
        """Schedule every eligible batch up to transport capacity.

        Prefill and decode may be outstanding concurrently.
        The planner prevents double-scheduling one request.
        """
        if self._failed:
            raise RuntimeError(
                "Runtime failed; restart required"
            )

        submitted = []

        try:
            while len(self._pending_batches) < self.max_pending:
                batch = self.planner.next_concurrent_batch()

                if batch is None:
                    break

                plan = batch.plan
                ids, positions = self._validate_positions(
                    plan
                )

                if plan.phase == "prefill":
                    payload = tuple(
                        tuple(item.tokens)
                        for item in plan.items
                    )

                elif plan.phase == "decode":
                    if any(
                        len(item.tokens) != 1
                        for item in plan.items
                    ):
                        raise RuntimeError(
                            "Invalid decode microbatch"
                        )

                    payload = tuple(
                        item.tokens[0]
                        for item in plan.items
                    )

                else:
                    raise RuntimeError("Unknown batch phase")

                token_count = sum(
                    len(item.tokens)
                    for item in plan.items
                )

                sequence = self.transport.submit(
                    ids,
                    positions,
                    payload,
                    phase=plan.phase,
                    token_count=token_count,
                )

                self._pending_batches[sequence] = (
                    batch.batch_id, ids
                )
                self._pending_plans[sequence] = batch

                submitted.append((
                    sequence,
                    plan.phase,
                    ids,
                ))

        except Exception:
            # A stage may have committed some KV already.
            # Never resume after an unexpected dispatch failure.
            self._poison()
            raise

        return {
            "submitted": tuple(submitted),
            "pending": len(self._pending_batches),
        }

    def collect_ready(self, *, wait=False, timeout=15):
        """Acknowledge completed prefill/decode batches.

        Partial prefill advances the token offset but emits no
        token. Completed prefill and decode sample logits.
        """
        if self._failed:
            raise RuntimeError(
                "Runtime failed; restart required"
            )

        updates = []

        def collect_one(sequence, output):
            batch = self._pending_plans[sequence]
            plan = batch.plan

            if (
                not isinstance(output, torch.Tensor)
                or output.ndim != 2
                or output.shape[0] != len(plan.items)
            ):
                raise RuntimeError(
                    "Invalid final-stage logits"
                )

            samples = {}

            for index, item in enumerate(plan.items):
                if item.emits_token:
                    samples[item.request_id] = int(
                        output[index].argmax()
                    )
                else:
                    samples[item.request_id] = None

            stop_ids = tuple(
                rid for rid, token in samples.items()
                if (
                    token is not None
                    and token in self.stop_token_ids
                )
            )

            update = self.planner.complete_concurrent_batch(
                batch.batch_id,
                samples,
                stop_ids=stop_ids,
            )

            del self._pending_batches[sequence]
            del self._pending_plans[sequence]

            # This releases KV only after final-stage
            # execution and transport acknowledgement.
            self._record_update(update)
            updates.append(update)

        try:
            for sequence in tuple(self._pending_batches):
                try:
                    output = self.transport.wait(
                        sequence,
                        timeout=0,
                    )
                except TimeoutError:
                    continue

                collect_one(sequence, output)

            if (
                not updates
                and wait
                and self._pending_batches
            ):
                sequence = next(
                    iter(self._pending_batches)
                )

                output = self.transport.wait(
                    sequence,
                    timeout=timeout,
                )
                collect_one(sequence, output)

        except TimeoutError:
            # The outstanding batch retains ownership.
            raise

        except Exception:
            self._poison()
            raise

        return tuple(updates)

    def snapshot(self):
        state = super().snapshot()

        state["runtime_mode"] = "mixed_phase_cpu"
        state["pending_phases"] = {
            sequence: batch.plan.phase
            for sequence, batch
            in self._pending_plans.items()
        }
        state["pending_request_ids"] = {
            sequence: tuple(
                item.request_id
                for item in batch.plan.items
            )
            for sequence, batch
            in self._pending_plans.items()
        }

        return state
