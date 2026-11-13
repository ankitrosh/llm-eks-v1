"""Multi-inflight extension of the V3.2 continuous batch planner.

Preserves the original single-inflight planner unchanged.
One outstanding plan per request; independent plans may finish
out of order.
"""

from dataclasses import dataclass

from llm_runtime.v3_2_batch_planner import (
    BatchItem,
    BatchPlan,
    ContinuousBatchPlanner,
)
from llm_runtime.v3_2_policy_api import (
    BatchCandidate,
    BatchContext,
    validate_batch_decision,
)


@dataclass(frozen=True)
class ConcurrentBatch:
    batch_id: int
    plan: BatchPlan


class ConcurrentBatchPlanner(ContinuousBatchPlanner):
    def __init__(self, config, *, batch_policy=None):
        super().__init__(config, batch_policy=batch_policy)

        self._next_batch_id = 0
        self._outstanding = {}
        self._busy = set()
        self._poisoned = False

    @property
    def outstanding_batch_ids(self):
        return tuple(sorted(self._outstanding))

    def next_concurrent_batch(
        self, *, phase=None, max_requests=None
    ):
        if self._poisoned:
            raise RuntimeError("Planner failed; restart required")

        if phase not in (None, "prefill", "decode"):
            raise ValueError("Invalid phase filter")

        if max_requests is None:
            max_requests = self.settings.max_requests_per_batch

        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("Invalid max_requests")

        self._admit()

        # Busy requests remain resident but cannot be planned again.
        order = tuple(
            rid for rid in self._active
            if rid not in self._busy
            and (
                phase is None
                or self._request_phase(rid) == phase
            )
        )

        if not order:
            return None

        limit = min(
            max_requests,
            self.settings.max_requests_per_batch,
        )

        if self._batch_policy is None:
            phases = {
                self._request_phase(rid)
                for rid in order
            }

            if phase is not None:
                selected_phase = phase

            elif (
                self.settings.policy == "fcfs"
                and len(phases) == 2
            ):
                selected_phase = (
                    "decode"
                    if self._last_phase == "prefill"
                    else "prefill"
                )

            else:
                selected_phase = self._request_phase(order[0])

        else:
            candidates = []

            for rid in order:
                state = self._requests[rid]
                pending = (
                    len(state.prompt) - state.offset
                )

                candidates.append(BatchCandidate(
                    request_id=rid,
                    phase=self._request_phase(rid),
                    ready_tokens=max(1, pending),
                ))

            context = BatchContext(
                candidates=tuple(candidates),
                max_requests_per_batch=limit,
                max_batch_tokens=self.settings.max_batch_tokens,
                last_phase=self._last_phase,
            )

            decision = validate_batch_decision(
                self._batch_policy.select_batch(context),
                context,
            )

            selected_phase = decision.phase
            order = decision.request_ids

        budget = self.settings.max_batch_tokens
        items = []

        for rid in order:
            if len(items) >= limit or budget < 1:
                break

            state = self._requests[rid]

            if self._request_phase(rid) != selected_phase:
                continue

            if selected_phase == "prefill":
                position = state.offset
                take = min(
                    len(state.prompt) - position,
                    budget,
                )
                tokens = state.prompt[
                    position:position + take
                ]
                emits = (
                    position + take == len(state.prompt)
                )

            else:
                position = (
                    len(state.prompt)
                    + len(state.generated) - 1
                )
                tokens = (state.generated[-1],)
                emits = True

            items.append(BatchItem(
                rid,
                selected_phase,
                state.step,
                position,
                tokens,
                emits,
            ))

            budget -= len(tokens)

        if not items:
            raise RuntimeError("No schedulable requests")

        plan = BatchPlan(
            selected_phase,
            tuple(items),
            self.settings.max_batch_tokens - budget,
        )

        self._next_batch_id += 1
        batch_id = self._next_batch_id

        self._outstanding[batch_id] = plan
        self._busy.update(
            item.request_id for item in items
        )
        self._last_phase = selected_phase

        if (
            self._batch_policy is None
            and self.settings.policy == "round_robin"
            and len(self._active) > 1
        ):
            self._active.rotate(-1)

        return ConcurrentBatch(batch_id, plan)

    def _request_phase(self, rid):
        state = self._requests[rid]
        return (
            "prefill"
            if state.offset < len(state.prompt)
            else "decode"
        )

    def complete_concurrent_batch(
        self, batch_id, samples, *, stop_ids=()
    ):
        if self._poisoned:
            raise RuntimeError("Planner failed; restart required")

        if batch_id not in self._outstanding:
            raise KeyError(batch_id)

        if self._inflight is not None:
            raise RuntimeError(
                "Legacy single-batch operation in progress"
            )

        plan = self._outstanding[batch_id]

        # Reuse the existing, tested completion state machine.
        # Only the selected plan is presented as _inflight.
        self._inflight = plan

        try:
            update = super().complete_batch(
                samples, stop_ids=stop_ids
            )
        except Exception:
            self._inflight = None
            self._poisoned = True
            raise

        del self._outstanding[batch_id]

        self._busy.difference_update(
            item.request_id for item in plan.items
        )

        return update

    def cancel(self, request_id):
        if self._poisoned:
            raise RuntimeError("Planner failed; restart required")

        state = self._requests.get(request_id)

        if state is None:
            return False

        if request_id in self._busy:
            # Defer KV release until that batch completes.
            state.cancelling = True
            return True

        return super().cancel(request_id)

    def next_batch(self):
        raise RuntimeError(
            "Use next_concurrent_batch() on concurrent planner"
        )

    def complete_batch(self, samples, *, stop_ids=()):
        raise RuntimeError(
            "Use complete_concurrent_batch()"
        )
