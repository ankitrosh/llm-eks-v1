"""CPU microbatch transport for V3.2.

One worker per stage preserves stage-local FIFO.
Different stages can process different microbatches concurrently.

This is an in-process reference, not NCCL transport.
"""

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread


class MicrobatchBackpressure(RuntimeError):
    pass


class MicrobatchPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class MicrobatchEnvelope:
    sequence: int
    request_ids: tuple[int, ...]
    positions: tuple[int, ...]
    payload: object


class _Ticket:
    def __init__(self, ids):
        self.request_ids = ids
        self.ready = Event()
        self.result = None
        self.error = None


class CPUMicrobatchTransport:
    """Forward-only, bounded, ordered microbatch pipeline.

    Stage functions receive an envelope and return the
    output payload for the following stage.

    Requests cannot have multiple unconsumed microbatches.
    Any stage failure poisons the entire pipeline.
    """

    def __init__(self, stage_functions, *, max_pending=4):
        functions = tuple(stage_functions)

        if (
            len(functions) < 2
            or not all(callable(fn) for fn in functions)
        ):
            raise ValueError(
                "Provide at least two callable stages"
            )

        if type(max_pending) is not int or max_pending < 1:
            raise ValueError(
                "max_pending must be a positive integer"
            )

        self.max_pending = max_pending
        self._functions = functions
        self._queues = [Queue() for _ in functions]
        self._lock = Lock()
        self._stop = Event()

        self._tickets = {}
        self._reserved = set()
        self._next_sequence = 0

        self._closed = False
        self._failure = None

        self._stage_calls = [0] * len(functions)
        self._completed = 0

        self._workers = [
            Thread(
                target=self._worker,
                args=(rank,),
                daemon=True,
                name=f"v3.2-cpu-stage-{rank}",
            )
            for rank in range(len(functions))
        ]

        for worker in self._workers:
            worker.start()

    def submit(self, request_ids, positions, payload):
        ids = tuple(request_ids)
        positions = tuple(positions)

        if (
            not ids
            or len(ids) != len(positions)
            or len(set(ids)) != len(ids)
            or any(
                type(rid) is not int or rid < 1
                for rid in ids
            )
            or any(
                type(pos) is not int or pos < 1
                for pos in positions
            )
        ):
            raise ValueError(
                "Invalid request IDs or positions"
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("Transport closed")

            if self._failure is not None:
                raise MicrobatchPipelineError(
                    "Pipeline failed; restart required"
                ) from self._failure

            if len(self._tickets) >= self.max_pending:
                raise MicrobatchBackpressure(
                    "Too many pending microbatches"
                )

            if not self._reserved.isdisjoint(ids):
                raise ValueError(
                    "Request already has a pending microbatch"
                )

            self._next_sequence += 1
            sequence = self._next_sequence

            self._tickets[sequence] = _Ticket(ids)
            self._reserved.update(ids)

            packet = MicrobatchEnvelope(
                sequence,
                ids,
                positions,
                payload,
            )
            self._queues[0].put(packet)

            return sequence

    def wait(self, sequence, timeout=20):
        if type(sequence) is not int:
            raise ValueError("Invalid sequence")

        with self._lock:
            ticket = self._tickets.get(sequence)

        if ticket is None:
            raise KeyError(sequence)

        if not ticket.ready.wait(timeout):
            raise TimeoutError(
                f"Microbatch {sequence} timed out"
            )

        with self._lock:
            if self._tickets.get(sequence) is not ticket:
                raise KeyError(sequence)

            del self._tickets[sequence]

            self._reserved.difference_update(
                ticket.request_ids
            )

            error = ticket.error
            result = ticket.result

        if error is not None:
            raise MicrobatchPipelineError(
                f"Pipeline failed during microbatch {sequence}"
            ) from error

        return result

    def _fail(self, error):
        with self._lock:
            if self._failure is None:
                self._failure = error

            for ticket in self._tickets.values():
                ticket.error = self._failure
                ticket.ready.set()

        self._stop.set()

    def _worker(self, rank):
        while not self._stop.is_set():
            try:
                packet = self._queues[rank].get(
                    timeout=0.05
                )
            except Empty:
                continue

            if self._stop.is_set():
                return

            try:
                output = self._functions[rank](packet)
            except Exception as error:
                self._fail(error)
                return

            with self._lock:
                if self._failure is not None or self._closed:
                    return

                self._stage_calls[rank] += 1

                if rank + 1 < len(self._queues):
                    next_packet = MicrobatchEnvelope(
                        packet.sequence,
                        packet.request_ids,
                        packet.positions,
                        output,
                    )

                    self._queues[rank + 1].put(
                        next_packet
                    )

                else:
                    ticket = self._tickets.get(
                        packet.sequence
                    )

                    if ticket is None:
                        self._failure = RuntimeError(
                            "Missing microbatch ticket"
                        )
                        self._stop.set()
                        return

                    ticket.result = output
                    ticket.ready.set()
                    self._completed += 1

    def snapshot(self):
        with self._lock:
            return {
                "pending_sequences": tuple(
                    sorted(self._tickets)
                ),
                "reserved_request_ids": tuple(
                    sorted(self._reserved)
                ),
                "stage_calls": tuple(
                    self._stage_calls
                ),
                "completed_microbatches": self._completed,
                "failed": self._failure is not None,
            }

    def close(self):
        with self._lock:
            if self._closed:
                return

            self._closed = True

            for ticket in self._tickets.values():
                if not ticket.ready.is_set():
                    ticket.error = RuntimeError(
                        "Transport closed"
                    )
                    ticket.ready.set()

        self._stop.set()

        for worker in self._workers:
            worker.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
