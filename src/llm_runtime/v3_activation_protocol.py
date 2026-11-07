"""Ordered activation handoff for forward-only inference.

CPU protocol simulation, not NCCL.

The producer stages an activation without waiting for Stage 1.
A dedicated communication worker transfers it only after
the consumer declares readiness.

NCCL cannot match tensors using request-ID tags, so every
handoff must use a consistent sequence across both ranks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic
from typing import Any


class ActivationProtocolError(RuntimeError):
    """Fatal protocol mismatch; pipeline recovery required."""


class ActivationBackpressure(RuntimeError):
    """The bounded activation buffer is full."""


@dataclass(frozen=True)
class ActivationTicket:
    sequence: int
    request_id: int
    step: int
    position: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class _Pending:
    ticket: ActivationTicket
    payload: Any
    ready: bool = False


class OrderedActivationLane:
    """Strictly ordered, ready-before-send activation lane."""

    def __init__(
        self,
        *,
        max_pending: int = 4,
        timeout_s: float = 5.0,
    ):
        if max_pending < 1 or timeout_s <= 0:
            raise ValueError("Invalid activation lane limits")

        self._cv = Condition()
        self._pending = deque()
        self._delivered = {}
        self._next_sequence = 0

        self._closed = False
        self._error = None

        self._limit = max_pending
        self._timeout_s = timeout_s

        self._published = 0
        self._transferred = 0
        self._discarded = 0
        self._peak_pending = 0

        self._worker = Thread(
            target=self._send_loop,
            name="cpu-activation-lane",
            daemon=True,
        )
        self._worker.start()

    def _check(self):
        if self._error is not None:
            raise ActivationProtocolError(self._error)

        if self._closed:
            raise ActivationProtocolError(
                "Activation lane closed"
            )

    def publish(
        self,
        *,
        request_id: int,
        step: int,
        position: int,
        payload: Any,
        shape: tuple[int, ...],
        dtype: str,
    ) -> ActivationTicket:
        """Stage an activation without waiting for Stage 1."""

        with self._cv:
            self._check()

            if (
                len(self._pending) + len(self._delivered)
                >= self._limit
            ):
                raise ActivationBackpressure(
                    "Activation staging buffer full"
                )

            if request_id < 1 or step < 0 or position < 0:
                raise ValueError("Invalid request metadata")

            if (
                not shape
                or any(dim <= 0 for dim in shape)
                or not dtype
            ):
                raise ValueError("Invalid tensor metadata")

            ticket = ActivationTicket(
                self._next_sequence,
                request_id,
                step,
                position,
                tuple(shape),
                dtype,
            )

            self._next_sequence += 1
            self._pending.append(
                _Pending(ticket, payload)
            )

            self._published += 1
            self._peak_pending = max(
                self._peak_pending,
                len(self._pending),
            )

            self._cv.notify_all()
            return ticket

    def receive(
        self,
        ticket: ActivationTicket,
        *,
        request_id: int,
        step: int,
        position: int,
    ) -> Any:
        """Declare readiness, then receive the next activation."""

        with self._cv:
            self._check()

            expected = (
                request_id,
                step,
                position,
            )

            actual = (
                ticket.request_id,
                ticket.step,
                ticket.position,
            )

            if actual != expected:
                self._fail(
                    "Activation request/step/position mismatch"
                )
                self._check()

            if (
                not self._pending
                or self._pending[0].ticket != ticket
            ):
                self._fail(
                    "Activation receive order or ticket mismatch"
                )
                self._check()

            self._pending[0].ready = True
            self._cv.notify_all()

            deadline = monotonic() + self._timeout_s

            while ticket.sequence not in self._delivered:
                self._check()

                remaining = deadline - monotonic()

                if remaining <= 0:
                    self._fail(
                        "Activation rendezvous timeout"
                    )
                    self._check()

                self._cv.wait(remaining)

            return self._delivered.pop(
                ticket.sequence
            )

    def _fail(self, reason: str):
        if self._error is None:
            self._error = reason

        self._cv.notify_all()

    def _send_loop(self):
        """Dedicated CPU communication worker."""

        while True:
            with self._cv:
                while (
                    not self._closed
                    and self._error is None
                    and (
                        not self._pending
                        or not self._pending[0].ready
                    )
                ):
                    self._cv.wait()

                if self._closed or self._error is not None:
                    return

                pending = self._pending.popleft()

                self._delivered[
                    pending.ticket.sequence
                ] = pending.payload

                self._transferred += 1
                self._cv.notify_all()

    def discard_request(self, request_id: int):
        """Drop unsent activations for a finished request."""

        with self._cv:
            kept = deque()

            for pending in self._pending:
                if pending.ticket.request_id == request_id:
                    self._discarded += 1
                else:
                    kept.append(pending)

            self._pending = kept
            self._cv.notify_all()

    def stats(self):
        with self._cv:
            return {
                "published": self._published,
                "transferred": self._transferred,
                "discarded": self._discarded,
                "pending": len(self._pending),
                "unconsumed": len(self._delivered),
                "peak_pending": self._peak_pending,
            }

    def close(self):
        with self._cv:
            self._closed = True
            self._pending.clear()
            self._delivered.clear()
            self._cv.notify_all()

        self._worker.join(timeout=self._timeout_s)

        if self._worker.is_alive():
            raise RuntimeError(
                "Activation communication worker did not stop"
            )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
