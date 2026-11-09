"""V3.2 concurrent N-rank CPU inference runtime.

Multiple independent decode microbatches may execute in
different stages simultaneously. Prefill remains synchronous
and is deferred while decode batches are outstanding.

Single-owner runtime API; stage workers own their local KV.
"""

from threading import RLock

from llm_runtime.v3_2_concurrent_planner import (
    ConcurrentBatchPlanner,
)
from llm_runtime.v3_2_nrank_cpu_runtime import (
    NRankCPUContinuousRuntime,
)
from llm_runtime.v3_2_policy_transport import (
    PolicyCPUMicrobatchTransport,
)
from llm_runtime.v3_2_qwen3_stage_batch import (
    stage_block_decode,
    validate_stage_alignment,
)


class ConcurrentNRankCPURuntime(NRankCPUContinuousRuntime):
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
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Invalid max_pending")

        super().__init__(
            stages,
            caches,
            config,
            batch_policy=batch_policy,
            stop_token_ids=stop_token_ids,
        )

        # Replace the legacy single-slot transport.
        self.transport.close()

        self.planner = ConcurrentBatchPlanner(
            config,
            batch_policy=batch_policy,
        )

        # One lock per stage protects its physical KV cache.
        # Separate stages retain independent execution.
        self._stage_locks = tuple(
            RLock() for _ in self.stages
        )

        functions = []

        for stage, cache, guard in zip(
            self.stages, self.caches, self._stage_locks
        ):

            def execute(
                packet, stage=stage, cache=cache, guard=guard
            ):
                with guard:
                    positions = tuple(
                        cache._committed.get(rid)
                        for rid in packet.request_ids
                    )

                    if positions != packet.positions:
                        raise RuntimeError(
                            "Stage-local KV position mismatch"
                        )

                    return stage_block_decode(
                        stage,
                        cache,
                        packet.request_ids,
                        packet.payload,
                    )

            functions.append(execute)

        configured = (
            config.get("pipeline", {}).get("scheduling")
        )

        if configured is None:
            transport_config = None
        else:
            transport_config = config

        self.transport = PolicyCPUMicrobatchTransport(
            functions,
            max_pending=max_pending,
            scheduling_config=transport_config,
            policy=pipeline_policy,
        )

        self.max_pending = max_pending
        self._pending_batches = {}

    def _cleanup(self):
        """Release finished requests without racing stage KV writes."""
        released = self.planner.drain_releases()

        for cache, guard in zip(
            self.caches, self._stage_locks
        ):
            with guard:
                for rid in released:
                    cache.release_request(rid)

    def _record_update(self, update):
        for rid, token in update.emitted.items():
            self._generated[rid].append(token)

        self._finished.update(update.finished)
        self._cancelled.update(update.cancelled)
        self._cleanup()

    def _stop_requests(self, samples):
        return tuple(
            rid for rid, token in samples.items()
            if token in self.stop_token_ids
        )

    def _poison(self):
        self._failed = True
        self.transport.close()

    def dispatch_ready(self):
        """Prefill idle requests, then enqueue independent decode.

        Never execute synchronous prefill while any decode
        batch is outstanding: per-stage KV transactions
        cannot safely overlap with stage worker transactions.
        """
        if self._failed:
            raise RuntimeError("Runtime failed; restart required")

        prefill_updates = []
        decode_sequences = []

        try:
            if not self._pending_batches:
                while True:
                    batch = self.planner.next_concurrent_batch(
                        phase="prefill",
                        max_requests=1,
                    )

                    if batch is None:
                        break

                    (item,) = batch.plan.items
                    rid = item.request_id
                    tokens = tuple(item.tokens)

                    if (
                        not item.emits_token
                        or tokens != self._prompts[rid]
                        or rid in self.caches[0]._committed
                    ):
                        raise RuntimeError(
                            "Gate 7C.2B requires full prefills"
                        )

                    logits = self._prefill_one(rid, tokens)
                    sample = int(logits.argmax())
                    samples = {rid: sample}

                    update = self.planner.complete_concurrent_batch(
                        batch.batch_id,
                        samples,
                        stop_ids=self._stop_requests(samples),
                    )

                    self._record_update(update)
                    prefill_updates.append(update)

            while len(self._pending_batches) < self.max_pending:
                batch = self.planner.next_concurrent_batch(
                    phase="decode"
                )

                if batch is None:
                    break

                ids = tuple(
                    item.request_id
                    for item in batch.plan.items
                )

                positions = validate_stage_alignment(
                    self.caches, ids
                )

                tokens = tuple(
                    item.tokens[0]
                    for item in batch.plan.items
                )

                sequence = self.transport.submit(
                    ids,
                    positions,
                    tokens,
                    token_count=len(tokens),
                )

                self._pending_batches[sequence] = (
                    batch.batch_id,
                    ids,
                )

                decode_sequences.append(sequence)

        except Exception:
            self._poison()
            raise

        return {
            "prefill_updates": tuple(prefill_updates),
            "decode_sequences": tuple(decode_sequences),
            "pending": len(self._pending_batches),
        }

    def collect_ready(self, *, wait=False, timeout=15):
        """Collect any completed batches, including out of order.

        If wait=True and none are immediately ready, block
        for the oldest pending batch.
        """
        if self._failed:
            raise RuntimeError("Runtime failed; restart required")

        updates = []

        def collect_one(sequence, result):
            batch_id, ids = self._pending_batches[sequence]

            if result.shape[0] != len(ids):
                raise RuntimeError(
                    "Transport returned incorrect batch size"
                )

            samples = {
                rid: int(result[index].argmax())
                for index, rid in enumerate(ids)
            }

            update = self.planner.complete_concurrent_batch(
                batch_id,
                samples,
                stop_ids=self._stop_requests(samples),
            )

            del self._pending_batches[sequence]
            self._record_update(update)
            updates.append(update)

        try:
            for sequence in tuple(self._pending_batches):
                try:
                    result = self.transport.wait(
                        sequence, timeout=0
                    )
                except TimeoutError:
                    continue

                collect_one(sequence, result)

            if not updates and wait and self._pending_batches:
                sequence = next(iter(self._pending_batches))

                result = self.transport.wait(
                    sequence, timeout=timeout
                )

                collect_one(sequence, result)

        except TimeoutError:
            # A timeout does not imply failed execution.
            raise

        except Exception:
            # A stage may already have committed KV.
            # This runtime must not be resumed.
            self._poison()
            raise

        return tuple(updates)

    def advance(self):
        raise RuntimeError(
            "Use dispatch_ready() and collect_ready()"
        )

    def snapshot(self):
        state = super().snapshot()
        state["pending_batches"] = tuple(
            sorted(self._pending_batches)
        )
        state["outstanding_batch_ids"] = (
            self.planner.outstanding_batch_ids
        )
        return state
