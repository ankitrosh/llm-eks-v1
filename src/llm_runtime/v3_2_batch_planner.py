"""V3.2 CPU-only continuous batching planner.

This module plans work but does not execute model layers.
The future execution engine consumes BatchPlan objects.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from llm_runtime.v3_2_scheduler_api import (
    resolve_scheduler_config,
)
from llm_runtime.v3_2_policy_api import (
    BatchCandidate,
    BatchContext,
    load_batch_policy,
    validate_batch_decision,
)


class AdmissionBusy(RuntimeError):
    """Waiting queue is full."""


@dataclass(frozen=True)
class BatchItem:
    request_id: int
    phase: str
    step: int
    position: int
    tokens: tuple[int, ...]
    emits_token: bool


@dataclass(frozen=True)
class BatchPlan:
    phase: str
    items: tuple[BatchItem, ...]
    token_count: int


@dataclass(frozen=True)
class BatchUpdate:
    emitted: dict[int, int]
    finished: tuple[int, ...]
    cancelled: tuple[int, ...]
    admitted: tuple[int, ...]


@dataclass
class _Request:
    prompt: tuple[int, ...]
    max_new_tokens: int
    reserved: int
    offset: int = 0
    step: int = 0
    generated: list[int] = field(default_factory=list)
    cancelling: bool = False


class ContinuousBatchPlanner:
    """Single-owner admission and batch scheduling.

    Caller serializes method calls. KV reservations cover
    the prompt plus the maximum generated token count.
    """

    def __init__(
        self, config, *,
        kv_token_capacity=None,
        batch_policy=None,
    ):
        settings = resolve_scheduler_config(config)

        if settings.mode != "continuous":
            raise ValueError("Continuous mode required")

        memory = config.get("pipeline", {}).get("memory", {})
        maximum = memory.get("max_total_tokens_per_request")

        if type(maximum) is not int or maximum < 2:
            raise ValueError(
                "Explicit max_total_tokens_per_request >= 2 required"
            )

        if kv_token_capacity is None:
            kv_token_capacity = (
                settings.max_active_requests * maximum
            )

        if (
            type(kv_token_capacity) is not int
            or kv_token_capacity < 1
        ):
            raise ValueError("Invalid KV capacity")

        self.settings = settings
        policy_spec = (
            batch_policy
            if batch_policy is not None
            else settings.batch_policy_adapter
        )
        self._batch_policy = load_batch_policy(policy_spec)
        self.max_total_tokens = maximum
        self.kv_token_capacity = kv_token_capacity

        self._reserved = 0
        self._requests = {}
        self._last_request_id = 0
        self._waiting = deque()
        self._active = deque()
        self._inflight = None
        self._last_phase = None
        self._released = []

        self._completed = 0
        self._cancelled = 0

    def _admit(self):
        admitted = []

        while (
            self._waiting
            and len(self._active)
            < self.settings.max_active_requests
        ):
            rid = self._waiting[0]
            state = self._requests[rid]

            if (
                self._reserved + state.reserved
                > self.kv_token_capacity
            ):
                break

            self._waiting.popleft()
            self._active.append(rid)
            self._reserved += state.reserved
            admitted.append(rid)

        return tuple(admitted)

    def submit(self, request_id, prompt_tokens, max_new_tokens):
        if (
            type(request_id) is not int
            or request_id < 1
            or request_id <= self._last_request_id
        ):
            raise ValueError("Request IDs must increase")

        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise ValueError("Invalid output token count")

        prompt = tuple(prompt_tokens)

        if not prompt or any(
            type(token) is not int or token < 0
            for token in prompt
        ):
            raise ValueError("Invalid prompt tokens")

        reservation = len(prompt) + max_new_tokens

        if (
            reservation > self.max_total_tokens
            or reservation > self.kv_token_capacity
        ):
            raise ValueError("Request exceeds token budget")

        direct = (
            not self._waiting
            and len(self._active)
            < self.settings.max_active_requests
            and self._reserved + reservation
            <= self.kv_token_capacity
        )

        if (
            not direct
            and len(self._waiting)
            >= self.settings.max_queue_requests
        ):
            raise AdmissionBusy("Waiting queue full")

        self._requests[request_id] = _Request(
            prompt, max_new_tokens, reservation
        )

        self._last_request_id = request_id
        self._waiting.append(request_id)
        self._admit()

        return request_id

    def next_batch(self):
        if self._inflight is not None:
            raise RuntimeError("Previous batch still in flight")

        self._admit()

        if not self._active:
            return None

        order = tuple(self._active)
        if self._batch_policy is None:
            first = self._requests[order[0]]
            first_phase = (
                "prefill"
                if first.offset < len(first.prompt)
                else "decode"
            )

            phases = {
                "prefill"
                if self._requests[rid].offset
                < len(self._requests[rid].prompt)
                else "decode"
                for rid in order
            }

            if (
                self.settings.policy == "fcfs"
                and len(phases) == 2
            ):
                phase = (
                    "decode"
                    if self._last_phase == "prefill"
                    else "prefill"
                )
            else:
                phase = first_phase
        else:
            candidates = []

            for rid in order:
                state = self._requests[rid]
                pending = len(state.prompt) - state.offset

                candidates.append(BatchCandidate(
                    request_id=rid,
                    phase=(
                        "prefill" if pending > 0 else "decode"
                    ),
                    ready_tokens=max(1, pending),
                ))

            context = BatchContext(
                candidates=tuple(candidates),
                max_requests_per_batch=(
                    self.settings.max_requests_per_batch
                ),
                max_batch_tokens=self.settings.max_batch_tokens,
                last_phase=self._last_phase,
            )

            decision = validate_batch_decision(
                self._batch_policy.select_batch(context),
                context,
            )

            phase = decision.phase
            order = decision.request_ids

        budget = self.settings.max_batch_tokens
        items = []

        for rid in order:
            if (
                len(items)
                >= self.settings.max_requests_per_batch
                or budget < 1
            ):
                break

            state = self._requests[rid]

            current = (
                "prefill"
                if state.offset < len(state.prompt)
                else "decode"
            )

            if current != phase:
                continue

            if phase == "prefill":
                take = min(
                    len(state.prompt) - state.offset,
                    budget,
                )

                position = state.offset
                tokens = state.prompt[
                    position:position + take
                ]

                emits = (
                    position + take == len(state.prompt)
                )

            else:
                tokens = (state.generated[-1],)
                position = (
                    len(state.prompt)
                    + len(state.generated) - 1
                )
                emits = True

            items.append(BatchItem(
                rid, phase, state.step,
                position, tokens, emits,
            ))

            budget -= len(tokens)

        if not items:
            raise RuntimeError("No schedulable requests")

        plan = BatchPlan(
            phase,
            tuple(items),
            self.settings.max_batch_tokens - budget,
        )

        self._inflight = plan
        self._last_phase = phase

        if (
            self._batch_policy is None
            and self.settings.policy == "round_robin"
            and len(self._active) > 1
        ):
            self._active.rotate(-1)

        return plan

    def _finish(self, rid, *, cancelled):
        state = self._requests.pop(rid)

        if rid in self._active:
            self._active.remove(rid)
            self._reserved -= state.reserved
        else:
            self._waiting.remove(rid)

        self._released.append(rid)

        if cancelled:
            self._cancelled += 1
        else:
            self._completed += 1

    def cancel(self, request_id):
        state = self._requests.get(request_id)

        if state is None:
            return False

        in_flight = (
            self._inflight is not None
            and any(
                item.request_id == request_id
                for item in self._inflight.items
            )
        )

        if in_flight:
            state.cancelling = True
        else:
            self._finish(request_id, cancelled=True)
            self._admit()

        return True

    def complete_batch(self, samples, *, stop_ids=()):
        plan = self._inflight

        if plan is None:
            raise RuntimeError("No in-flight batch")

        if not isinstance(samples, Mapping):
            raise TypeError("Samples must be a mapping")

        expected = {
            item.request_id for item in plan.items
        }

        if set(samples) != expected:
            raise ValueError("Batch result IDs mismatch")

        stop = set(stop_ids)

        if not stop.issubset(expected):
            raise ValueError("Invalid stop IDs")

        # Validate every result before changing state.
        for item in plan.items:
            value = samples[item.request_id]
            state = self._requests[item.request_id]

            if state.cancelling:
                continue

            if item.emits_token:
                if (
                    type(value) is not int
                    or value < 0
                ):
                    raise ValueError("Invalid sampled token")
            elif value is not None:
                raise ValueError(
                    "Partial prefill must not emit a token"
                )

        emitted = {}
        finished = []
        cancelled = []

        for item in plan.items:
            rid = item.request_id
            state = self._requests[rid]

            if state.cancelling:
                self._finish(rid, cancelled=True)
                cancelled.append(rid)
                continue

            if item.phase == "prefill":
                state.offset += len(item.tokens)

            state.step += 1

            if item.emits_token:
                token = samples[rid]
                state.generated.append(token)
                emitted[rid] = token

                if (
                    rid in stop
                    or len(state.generated)
                    == state.max_new_tokens
                ):
                    self._finish(rid, cancelled=False)
                    finished.append(rid)

        self._inflight = None
        admitted = self._admit()

        return BatchUpdate(
            emitted,
            tuple(finished),
            tuple(cancelled),
            admitted,
        )

    def drain_releases(self):
        """IDs whose local/remote KV can now be released."""
        released = tuple(self._released)
        self._released.clear()
        return released

    def snapshot(self):
        return {
            "waiting": tuple(self._waiting),
            "running": tuple(self._active),
            "in_flight": (
                ()
                if self._inflight is None
                else tuple(
                    item.request_id
                    for item in self._inflight.items
                )
            ),
            "reserved_kv_tokens": self._reserved,
            "kv_token_capacity": self.kv_token_capacity,
            "completed": self._completed,
            "cancelled": self._cancelled,
        }
