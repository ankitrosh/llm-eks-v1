"""Bounded asynchronous CPU/Gloo transport for V3.2.

The rank-0 caller computes its local stage before submit().
It then forwards hidden states to rank 1.

Ranks 1..N-1 process FIFO microbatches independently.
Rank 0 receives results through a separate thread.

CPU/Gloo correctness reference, not NCCL or GPU serving.
"""

from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread

from llm_runtime.v3_2_gloo_wire import (
    WireMicrobatch,
    WireRelease,
    WireFailure,
    validate_wire_release,
    recv_wire,
    send_wire,
    validate_wire_microbatch,
)


class AsyncGlooBackpressure(RuntimeError):
    pass


class AsyncGlooFailure(RuntimeError):
    pass


@dataclass
class _Ticket:
    request_ids: tuple[int, ...]
    packet: WireMicrobatch
    ready: Event
    result: WireMicrobatch | None = None
    error: Exception | None = None


@dataclass
class _ReleaseTicket:
    control: WireRelease
    ready: Event
    error: Exception | None = None


class AsyncGlooCoordinator:
    """Rank-0 asynchronous microbatch coordinator.

    Request IDs remain reserved until wait() acknowledges
    the result. Separate requests may be in flight together.
    """

    def __init__(self, *, world_size, max_pending=4):
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() != 0:
            raise RuntimeError(
                "Coordinator requires distributed rank 0"
            )

        if type(world_size) is not int or world_size < 2:
            raise ValueError("Need at least two ranks")

        if dist.get_world_size() != world_size:
            raise ValueError("World size mismatch")

        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("Invalid max_pending")

        self.world_size = world_size
        self.max_pending = max_pending

        self._lock = Lock()
        self._outbox = Queue()

        self._tickets = {}
        self._reserved = set()

        self._release_tickets = {}
        self._release_reserved = set()
        self._next_release = 0

        self._next_sequence = 0
        self._last_received = 0

        self._sent = 0
        self._received = 0

        self._failure = None
        self._failure_observed = False
        self._closed = False
        self._stop_received = Event()

        self._sender = Thread(
            target=self._send_loop,
            daemon=True,
            name="v32-gloo-sender",
        )

        self._receiver = Thread(
            target=self._recv_loop,
            daemon=True,
            name="v32-gloo-receiver",
        )

        self._sender.start()
        self._receiver.start()

    def submit(self, packet):
        validate_wire_microbatch(packet)

        if packet.payload.ndim != 3:
            raise ValueError(
                "RUN requires [B,T,H] tensor"
            )

        # The caller may reuse its tensor after submit().
        packet = WireMicrobatch(
            packet.sequence,
            packet.phase,
            packet.request_ids,
            packet.positions,
            packet.lengths,
            packet.payload.detach().clone().contiguous(),
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("Coordinator closed")

            if self._failure is not None:
                raise AsyncGlooFailure(
                    "Pipeline failed"
                ) from self._failure

            if packet.sequence != self._next_sequence + 1:
                raise ValueError(
                    "Sequence must increase by one"
                )

            if len(self._tickets) >= self.max_pending:
                raise AsyncGlooBackpressure(
                    "Too many unacknowledged microbatches"
                )

            if not self._reserved.isdisjoint(
                packet.request_ids
            ):
                raise ValueError(
                    "Request already has an outstanding "
                    "microbatch"
                )

            if not self._release_reserved.isdisjoint(
                packet.request_ids
            ):
                raise ValueError(
                    "Request release is pending"
                )

            ticket = _Ticket(
                packet.request_ids,
                packet,
                Event(),
            )

            self._tickets[packet.sequence] = ticket
            self._reserved.update(packet.request_ids)
            self._next_sequence = packet.sequence

            self._outbox.put(packet)
            return packet.sequence

    def submit_release(self, request_ids):
        """Queue a stage-ordered KV release."""
        ids = tuple(request_ids)

        validate_wire_release(
            WireRelease(1, ids)
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("Coordinator closed")

            if self._failure is not None:
                raise AsyncGlooFailure(
                    "Pipeline failed"
                ) from self._failure

            if len(self._release_tickets) >= self.max_pending:
                raise AsyncGlooBackpressure(
                    "Too many unacknowledged releases"
                )

            if not self._reserved.isdisjoint(ids):
                raise ValueError(
                    "Request has an outstanding microbatch"
                )

            if not self._release_reserved.isdisjoint(ids):
                raise ValueError(
                    "Request release is already pending"
                )

            self._next_release += 1

            control = WireRelease(
                self._next_release, ids
            )

            self._release_tickets[
                control.sequence
            ] = _ReleaseTicket(
                control, Event()
            )

            self._release_reserved.update(ids)
            self._outbox.put(control)

            return control.sequence

    def wait_release(self, sequence, timeout=20):
        if type(sequence) is not int or sequence < 1:
            raise ValueError(
                "Invalid release sequence"
            )

        with self._lock:
            ticket = self._release_tickets.get(
                sequence
            )

        if ticket is None:
            raise KeyError(sequence)

        if not ticket.ready.wait(timeout):
            raise TimeoutError(
                f"KV release {sequence} timed out"
            )

        with self._lock:
            if (
                self._release_tickets.get(sequence)
                is not ticket
            ):
                raise KeyError(sequence)

            del self._release_tickets[sequence]

            self._release_reserved.difference_update(
                ticket.control.request_ids
            )

            failure = ticket.error

        if failure is not None:
            with self._lock:
                self._failure_observed = True

            raise AsyncGlooFailure(
                f"KV release {sequence} failed"
            ) from failure

        return ticket.control.request_ids

    def wait(self, sequence, timeout=20):
        if type(sequence) is not int or sequence < 1:
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

            result = ticket.result
            error = ticket.error

        if error is not None:
            with self._lock:
                self._failure_observed = True

            raise AsyncGlooFailure(
                f"Microbatch {sequence} failed"
            ) from error

        return result

    def _fail(self, error):
        with self._lock:
            if self._failure is None:
                self._failure = error

            for ticket in self._tickets.values():
                ticket.error = self._failure
                ticket.ready.set()

            for ticket in self._release_tickets.values():
                ticket.error = self._failure
                ticket.ready.set()

    def _send_loop(self):
        try:
            while True:
                packet = self._outbox.get()

                if packet is None:
                    send_wire(1, kind="stop")
                    return

                # Send queued messages even after ERROR,
                # so remote workers can drain until STOP.
                if isinstance(packet, WireRelease):
                    send_wire(
                        1,
                        kind="release",
                        control=packet,
                    )
                else:
                    send_wire(
                        1,
                        kind="run",
                        packet=packet,
                    )

                    with self._lock:
                        self._sent += 1

        except Exception as error:
            self._fail(error)

    def _recv_loop(self):
        try:
            while True:
                kind, packet = recv_wire(
                    self.world_size - 1
                )

                if kind == "stop":
                    with self._lock:
                        expected = (
                            self._closed
                            and not self._tickets
                            and not self._release_tickets
                        )

                    if not expected:
                        self._fail(RuntimeError(
                            "Unexpected pipeline STOP"
                        ))

                    self._stop_received.set()
                    return

                if kind == "error":
                    self._fail(RuntimeError(
                        f"Remote stage {packet.rank} failed "
                        f"at sequence {packet.sequence} "
                        f"(code {packet.code})"
                    ))
                    # Wait for the regular STOP acknowledgement.
                    continue

                with self._lock:
                    failed = self._failure is not None

                if failed and kind in (
                    "result", "release_ack"
                ):
                    # Drain late replies, never commit them
                    # after a fatal distributed failure.
                    continue

                if kind == "release_ack":
                    with self._lock:
                        ticket = self._release_tickets.get(
                            packet.sequence
                        )

                        if (
                            ticket is None
                            or ticket.control != packet
                            or ticket.ready.is_set()
                        ):
                            raise RuntimeError(
                                "Unknown or mismatched release ACK"
                            )

                        ticket.ready.set()

                    continue

                if kind != "result":
                    raise RuntimeError(
                        "Expected RESULT from final stage"
                    )

                with self._lock:
                    seq = packet.sequence
                    ticket = self._tickets.get(seq)

                    if (
                        ticket is None
                        or seq != self._last_received + 1
                        or packet.phase != ticket.packet.phase
                        or packet.request_ids
                        != ticket.packet.request_ids
                        or packet.positions
                        != ticket.packet.positions
                        or packet.lengths
                        != ticket.packet.lengths
                    ):
                        raise RuntimeError(
                            "Out-of-order or mismatched result"
                        )

                    self._last_received = seq
                    self._received += 1

                    ticket.result = packet
                    ticket.ready.set()

        except Exception as error:
            self._fail(error)
            self._stop_received.set()

    def snapshot(self):
        with self._lock:
            return {
                "pending_sequences": tuple(
                    sorted(self._tickets)
                ),
                "reserved_request_ids": tuple(
                    sorted(self._reserved)
                ),
                "pending_release_sequences": tuple(
                    sorted(self._release_tickets)
                ),
                "releasing_request_ids": tuple(
                    sorted(self._release_reserved)
                ),
                "sent": self._sent,
                "received": self._received,
                "failed": self._failure is not None,
                "closed": self._closed,
            }

    def close(self, timeout=20):
        with self._lock:
            if self._closed:
                return

            if self._tickets or self._release_tickets:
                if self._failure is None:
                    raise RuntimeError(
                        "Acknowledge pending batches and releases "
                        "before close"
                    )

                # Fatal failure wakes all waiters.
                # Discard ownership during shutdown.
                self._tickets.clear()
                self._release_tickets.clear()
                self._reserved.clear()
                self._release_reserved.clear()

            self._closed = True

        self._outbox.put(None)

        self._sender.join(timeout=timeout)

        if self._sender.is_alive():
            raise TimeoutError(
                "Sender thread did not exit"
            )

        if not self._stop_received.wait(timeout):
            raise TimeoutError(
                "Distributed STOP acknowledgement timed out"
            )

        self._receiver.join(timeout=timeout)

        if self._receiver.is_alive():
            raise TimeoutError(
                "Receiver thread did not exit"
            )

        with self._lock:
            failure = self._failure

        if failure is not None:
            raise AsyncGlooFailure(
                "Pipeline failed"
            ) from failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        with self._lock:
            already_observed = (
                self._failure is not None
                and self._failure_observed
                and not self._tickets
                and not self._release_tickets
            )

        try:
            self.close()
        except AsyncGlooFailure:
            if not already_observed:
                raise

        return False


def run_async_gloo_stage(
    rank, world_size, execute, *, release=None
):
    """Remote worker with fail-fast callback error handling.

    Following a fatal error, stop modifying KV and drain
    incoming frames until STOP arrives.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() != rank:
        raise RuntimeError("Invalid rank context")

    if (
        not 1 <= rank < world_size
        or dist.get_world_size() != world_size
    ):
        raise ValueError("Invalid distributed stage layout")

    if not callable(execute):
        raise ValueError("Invalid stage callback")

    destination = (
        rank + 1
        if rank + 1 < world_size
        else 0
    )

    expected_sequence = 1
    failed = False

    def send_failure(sequence, code):
        send_wire(
            destination,
            kind="error",
            control=WireFailure(sequence, rank, code),
        )

    while True:
        kind, packet = recv_wire(rank - 1)

        if kind == "stop":
            send_wire(destination, kind="stop")
            return

        if kind == "error":
            if not failed:
                failed = True
                send_wire(
                    destination,
                    kind="error",
                    control=packet,
                )
            continue

        if kind == "release":
            if failed:
                continue

            try:
                if not callable(release):
                    raise RuntimeError(
                        "Missing KV release callback"
                    )

                release(packet.request_ids)

            except Exception:
                failed = True
                send_failure(packet.sequence, 2)
                continue

            send_wire(
                destination,
                kind=(
                    "release"
                    if rank + 1 < world_size
                    else "release_ack"
                ),
                control=packet,
            )
            continue

        if (
            kind != "run"
            or packet.sequence != expected_sequence
        ):
            raise RuntimeError(
                "Unexpected stage sequence"
            )

        expected_sequence += 1

        if failed:
            continue

        try:
            output = execute(packet)

            response = WireMicrobatch(
                packet.sequence,
                packet.phase,
                packet.request_ids,
                packet.positions,
                packet.lengths,
                output,
            )

            validate_wire_microbatch(response)

            required_rank = (
                3 if rank + 1 < world_size else 2
            )

            if output.ndim != required_rank:
                raise ValueError(
                    "Unexpected stage output rank"
                )

        except Exception:
            failed = True
            send_failure(packet.sequence, 1)
            continue

        # A broken process-group send cannot be recovered
        # by sending another message over that channel.
        send_wire(
            destination,
            kind=(
                "run"
                if rank + 1 < world_size
                else "result"
            ),
            packet=response,
        )
