"""CPU microbatch transport with policy-driven stage dispatch."""

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Condition, Event, Thread

from llm_runtime.v3_2_pipeline_policy import (
    PipelineBusy,
    PipelineDispatchController,
)


class PolicyTransportBackpressure(RuntimeError):
    pass


class PolicyTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyMicrobatch:
    sequence: int
    request_ids: tuple[int, ...]
    positions: tuple[int, ...]
    payload: object
    phase: str = "decode"


class _Ticket:
    def __init__(self, request_ids):
        self.request_ids = request_ids
        self.ready = Event()
        self.result = None
        self.error = None


class PolicyCPUMicrobatchTransport:
    """Policy dispatch with one worker per stage.

    The controller is accessed under a single condition lock.
    A request remains reserved until the final result is
    acknowledged. Stage failure poisons the entire transport.
    """

    def __init__(
        self, stage_functions, *, max_pending=4,
        scheduling_config=None, policy=None,
    ):
        functions = tuple(stage_functions)

        if len(functions) < 2 or not all(
            callable(fn) for fn in functions
        ):
            raise ValueError(
                "At least two callable stages required"
            )

        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Invalid max_pending")

        if scheduling_config is None:
            scheduling_config = {
                "pipeline": {
                    "num_ranks": len(functions),
                    "scheduling": {
                        "policy": "downstream_first",
                        "max_inflight_microbatches": max_pending,
                        "max_queued_microbatches": max_pending,
                    },
                }
            }

        self._controller = PipelineDispatchController(
            scheduling_config,
            stages=len(functions),
            policy=policy,
        )

        self._functions = functions
        self.max_pending = max_pending

        self._cv = Condition()
        self._queues = [Queue() for _ in functions]

        self._tickets = {}
        self._reserved = set()
        self._positions = {}
        self._payloads = {}
        self._phases = {}
        self._last_sequence = 0

        self._closed = False
        self._failure = None
        self._stage_calls = [0] * len(functions)
        self._completed = 0
        self._dispatch_trace = []

        self._scheduler = Thread(
            target=self._schedule,
            daemon=True,
            name="v32-policy-scheduler",
        )

        self._workers = [
            Thread(
                target=self._worker,
                args=(rank,),
                daemon=True,
                name=f"v32-policy-stage-{rank}",
            )
            for rank in range(len(functions))
        ]

        self._scheduler.start()

        for worker in self._workers:
            worker.start()

    def submit(
        self, request_ids, positions, payload, *,
        token_count=None,
        phase="decode",
    ):
        ids = tuple(request_ids)
        pos = tuple(positions)

        if type(phase) is not str or phase not in (
            "prefill", "decode"
        ):
            raise ValueError("Invalid microbatch phase")

        if (
            not ids
            or len(ids) != len(pos)
            or len(set(ids)) != len(ids)
            or any(
                type(rid) is not int or rid < 1
                for rid in ids
            )
            or any(
                type(position) is not int
                or position < (0 if phase == "prefill" else 1)
                for position in pos
            )
        ):
            raise ValueError(
                "Invalid request IDs or positions"
            )

        if phase == "prefill":
            if (
                not isinstance(payload, (tuple, list))
                or len(payload) != len(ids)
                or any(
                    not isinstance(seq, (tuple, list))
                    or not seq
                    or any(
                        type(token) is not int or token < 0
                        for token in seq
                    )
                    for seq in payload
                )
            ):
                raise ValueError(
                    "Invalid prefill token sequences"
                )

            required_tokens = sum(
                len(seq) for seq in payload
            )
        else:
            required_tokens = len(ids)

        if token_count is None:
            token_count = required_tokens

        if (
            type(token_count) is not int
            or token_count < 1
            or token_count != required_tokens
        ):
            raise ValueError(
                "Microbatch token count mismatch"
            )

        with self._cv:
            if self._closed:
                raise RuntimeError("Transport closed")

            if self._failure is not None:
                raise PolicyTransportError(
                    "Pipeline failed; restart required"
                ) from self._failure

            if len(self._tickets) >= self.max_pending:
                raise PolicyTransportBackpressure(
                    "Too many pending microbatches"
                )

            if not self._reserved.isdisjoint(ids):
                raise ValueError(
                    "Request has an unacknowledged microbatch"
                )

            sequence = self._last_sequence + 1

            try:
                self._controller.enqueue(
                    sequence, ids, token_count
                )
            except PipelineBusy as exc:
                raise PolicyTransportBackpressure(
                    "Pipeline dispatch queue full"
                ) from exc

            self._last_sequence = sequence
            self._tickets[sequence] = _Ticket(ids)
            self._reserved.update(ids)
            self._positions[sequence] = pos
            self._payloads[sequence] = payload
            self._phases[sequence] = phase

            self._cv.notify_all()
            return sequence

    def wait(self, sequence, timeout=20):
        if type(sequence) is not int or sequence < 1:
            raise ValueError("Invalid sequence")

        with self._cv:
            ticket = self._tickets.get(sequence)

        if ticket is None:
            raise KeyError(sequence)

        if not ticket.ready.wait(timeout):
            raise TimeoutError(
                f"Microbatch {sequence} timed out"
            )

        with self._cv:
            if self._tickets.get(sequence) is not ticket:
                raise KeyError(sequence)

            del self._tickets[sequence]

            self._reserved.difference_update(
                ticket.request_ids
            )

            self._cv.notify_all()

            error = ticket.error
            result = ticket.result

        if error is not None:
            raise PolicyTransportError(
                f"Pipeline failed during microbatch {sequence}"
            ) from error

        return result

    def _fail_locked(self, error):
        if self._failure is None:
            self._failure = error

        for ticket in self._tickets.values():
            if not ticket.ready.is_set():
                ticket.error = self._failure
                ticket.ready.set()

        self._cv.notify_all()

    def _schedule(self):
        while True:
            with self._cv:
                if self._closed or self._failure is not None:
                    return

                try:
                    task = self._controller.dispatch()
                except Exception as error:
                    self._fail_locked(error)
                    return

                if task is None:
                    self._cv.wait()
                    continue

                sequence = task.microbatch_id

                packet = PolicyMicrobatch(
                    sequence,
                    task.request_ids,
                    self._positions[sequence],
                    self._payloads[sequence],
                    self._phases[sequence],
                )

                self._dispatch_trace.append(
                    (task.stage, sequence)
                )

                self._queues[task.stage].put(packet)

    def _worker(self, stage):
        while True:
            try:
                packet = self._queues[stage].get(
                    timeout=0.05
                )
            except Empty:
                with self._cv:
                    if (
                        self._closed
                        or self._failure is not None
                    ):
                        return
                continue

            with self._cv:
                if (
                    self._closed
                    or self._failure is not None
                ):
                    return

            try:
                result = self._functions[stage](packet)
            except Exception as error:
                with self._cv:
                    self._fail_locked(error)
                return

            with self._cv:
                if (
                    self._closed
                    or self._failure is not None
                ):
                    return

                sequence = packet.sequence

                try:
                    self._payloads[sequence] = result

                    finished = self._controller.complete(
                        stage, sequence
                    )

                    self._stage_calls[stage] += 1

                    if finished:
                        ticket = self._tickets[sequence]
                        ticket.result = result
                        ticket.ready.set()

                        self._completed += 1

                        del self._payloads[sequence]
                        del self._positions[sequence]
                        del self._phases[sequence]

                except Exception as error:
                    self._fail_locked(error)
                    return

                self._cv.notify_all()

    def snapshot(self):
        with self._cv:
            return {
                "pending_sequences": tuple(
                    sorted(self._tickets)
                ),
                "reserved_request_ids": tuple(
                    sorted(self._reserved)
                ),
                "stage_calls": tuple(self._stage_calls),
                "completed_microbatches": self._completed,
                "failed": self._failure is not None,
                "dispatch_trace": tuple(
                    self._dispatch_trace
                ),
                "controller": self._controller.snapshot(),
            }

    def close(self):
        with self._cv:
            if not self._closed:
                self._closed = True

                for ticket in self._tickets.values():
                    if not ticket.ready.is_set():
                        ticket.error = RuntimeError(
                            "Transport closed"
                        )
                        ticket.ready.set()

                self._cv.notify_all()

        self._scheduler.join(timeout=2)

        for worker in self._workers:
            worker.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
