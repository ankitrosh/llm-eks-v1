"""Two-rank ordered forward-pipeline transport.

A dedicated rank-0 IO thread owns all distributed send/recv calls.
TCP metadata establishes tensor order before PyTorch P2P operations.

Supports Gloo CPU and NCCL CUDA, depending on the process group.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .v3_pipeline_scheduler import FatalStageError


class TransportFailure(RuntimeError):
    """Fatal peer, transport or metadata failure."""


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _write(conn, obj):
    data = (
        json.dumps(obj, separators=(",", ":")) + "\n"
    ).encode()

    if len(data) > 4096:
        raise TransportFailure("Control message too large")

    conn.sendall(data)


def _read(reader):
    data = reader.readline(4097)

    if (
        not data
        or len(data) > 4096
        or not data.endswith(b"\n")
    ):
        raise TransportFailure(
            "Control channel closed or invalid frame"
        )

    try:
        result = json.loads(data)
    except ValueError as exc:
        raise TransportFailure(
            "Invalid JSON header"
        ) from exc

    if not isinstance(result, dict):
        raise TransportFailure(
            "Control header must be an object"
        )

    return result


def open_control(rank, host, port, timeout_s=60):
    """Establish a single persistent TCP metadata connection."""

    if rank == 0:
        listener = socket.socket()
        listener.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        listener.bind(("0.0.0.0", port))
        listener.listen(1)
        listener.settimeout(timeout_s)

        try:
            conn, _ = listener.accept()
        finally:
            listener.close()

    else:
        deadline = time.monotonic() + timeout_s

        while True:
            try:
                conn = socket.create_connection(
                    (host, port), timeout=3
                )
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)

    conn.settimeout(timeout_s)
    return conn


def _shape(value):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            type(dim) is not int or dim < 1 or dim > 4096
            for dim in value
        )
    ):
        raise TransportFailure(
            "Unsupported tensor shape"
        )

    if value[0] * value[1] * value[2] > 8_000_000:
        raise TransportFailure("Tensor too large")

    return tuple(value)


@dataclass
class TensorTicket:
    request_id: int
    step: int
    position: int
    sequence: int
    future: Future = field(
        default_factory=Future,
        repr=False,
    )


class OrderedRank0Transport:
    """Asynchronous activation delivery and result collection."""

    def __init__(
        self,
        conn,
        *,
        device,
        max_pending=4,
        timeout_s=60,
    ):
        self.conn = conn
        self.reader = conn.makefile("rb")
        self.device = torch.device(device)

        self.timeout_s = timeout_s
        self.max_pending = max_pending

        self._work = queue.Queue()
        self._lock = threading.Lock()
        self._tickets = {}

        self._next = 0
        self._error = None

        self._stats = {
            "sent": 0,
            "received": 0,
            "releases": 0,
        }

        self._thread = threading.Thread(
            target=self._io,
            name="rank0-io",
            daemon=True,
        )
        self._thread.start()

    def publish(
        self,
        request_id,
        step,
        position,
        tensor,
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.device
        ):
            raise TransportFailure(
                "Expected activation tensor on transport device"
            )

        if (
            tensor.ndim != 3
            or tensor.dtype not in _DTYPES.values()
        ):
            raise TransportFailure(
                "Invalid activation tensor"
            )

        with self._lock:
            if self._error:
                raise TransportFailure(self._error)

            if len(self._tickets) >= self.max_pending:
                raise TransportFailure(
                    "Activation buffer capacity exceeded"
                )

            ticket = TensorTicket(
                request_id,
                step,
                position,
                self._next,
            )

            self._next += 1
            self._tickets[ticket.sequence] = ticket

            self._work.put((
                "RUN",
                ticket,
                tensor.contiguous(),
            ))

            return ticket

    def await_result(self, ticket):
        try:
            return ticket.future.result(
                timeout=self.timeout_s
            )
        except FutureTimeout as exc:
            self._trip("Rank 1 reply timed out")
            raise TransportFailure(
                "Rank 1 reply timed out"
            ) from exc

    def release_request(self, request_id):
        """Nonblocking request cleanup callback."""
        with self._lock:
            if self._error is None:
                self._work.put((
                    "RELEASE",
                    request_id,
                    None,
                ))

    def stats(self):
        return dict(self._stats)

    def _trip(self, reason):
        with self._lock:
            if self._error is None:
                self._error = reason

            pending = list(self._tickets.values())
            self._tickets.clear()

        for ticket in pending:
            if not ticket.future.done():
                ticket.future.set_exception(
                    TransportFailure(reason)
                )

    def _io(self):
        try:
            while True:
                op, obj, tensor = self._work.get()

                if op == "STOP":
                    _write(self.conn, {"op": "STOP"})

                    if _read(self.reader).get("op") != "STOPPED":
                        raise TransportFailure(
                            "STOP acknowledgement missing"
                        )
                    return

                if op == "RELEASE":
                    _write(self.conn, {
                        "op": "RELEASE",
                        "rid": obj,
                    })

                    ack = _read(self.reader)

                    if ack != {
                        "op": "RELEASED",
                        "rid": obj,
                    }:
                        raise TransportFailure(
                            "RELEASE acknowledgement mismatch"
                        )

                    self._stats["releases"] += 1
                    continue

                ticket = obj

                if self._error:
                    continue

                _write(self.conn, {
                    "op": "RUN",
                    "seq": ticket.sequence,
                    "rid": ticket.request_id,
                    "step": ticket.step,
                    "position": ticket.position,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix(
                        "torch."
                    ),
                })

                reply = _read(self.reader)

                if reply != {
                    "op": "READY",
                    "seq": ticket.sequence,
                }:
                    raise TransportFailure(
                        "RUN acknowledgement mismatch"
                    )

                if tensor.is_cuda:
                    torch.cuda.synchronize(
                        device=tensor.device
                    )

                dist.send(tensor, dst=1)
                self._stats["sent"] += 1

                reply = _read(self.reader)

                if (
                    reply.get("seq") != ticket.sequence
                    or reply.get("rid") != ticket.request_id
                ):
                    raise TransportFailure(
                        "Result metadata mismatch"
                    )

                if reply.get("op") == "FATAL":
                    raise TransportFailure(
                        reply.get(
                            "error",
                            "Rank 1 fatal error",
                        )
                    )

                if reply.get("op") == "ERROR":
                    ticket.future.set_exception(
                        RuntimeError(
                            reply.get(
                                "error",
                                "Rank 1 execution error",
                            )
                        )
                    )

                elif reply.get("op") == "RESULT":
                    shape = _shape(reply.get("shape"))
                    dtype = _DTYPES.get(
                        reply.get("dtype")
                    )

                    if dtype is None:
                        raise TransportFailure(
                            "Unexpected output dtype"
                        )

                    result = torch.empty(
                        shape,
                        device=self.device,
                        dtype=dtype,
                    )

                    dist.recv(result, src=1)
                    self._stats["received"] += 1
                    ticket.future.set_result(result)

                else:
                    raise TransportFailure(
                        "Unknown result message"
                    )

                with self._lock:
                    self._tickets.pop(
                        ticket.sequence,
                        None,
                    )

        except BaseException as exc:
            self._trip(
                f"{type(exc).__name__}: {exc}"
            )

    def close(self):
        if (
            self._thread.is_alive()
            and self._error is None
        ):
            self._work.put(("STOP", None, None))

        self._thread.join(timeout=self.timeout_s)

        if self._thread.is_alive():
            self._trip(
                "Communication thread did not stop"
            )

        self.reader.close()
        self.conn.close()

        if self._thread.is_alive():
            raise TransportFailure(
                "Communication thread stuck"
            )


def run_rank1(conn, *, device, execute):
    """Execute remote stage operations and own their KV caches."""

    reader = conn.makefile("rb")
    device = torch.device(device)

    caches = {}
    next_steps = {}
    last_sequence = -1

    stats = {
        "runs": 0,
        "releases": 0,
        "peak_caches": 0,
        "trace": [],
    }

    try:
        while True:
            cmd = _read(reader)
            op = cmd.get("op")

            if op == "STOP":
                _write(conn, {"op": "STOPPED"})
                return stats

            if op == "RELEASE":
                rid = cmd.get("rid")

                caches.pop(rid, None)
                next_steps.pop(rid, None)

                stats["releases"] += 1

                _write(conn, {
                    "op": "RELEASED",
                    "rid": rid,
                })
                continue

            if op != "RUN":
                raise TransportFailure(
                    f"Unknown opcode: {op}"
                )

            seq, rid, step, position = (
                cmd.get(k)
                for k in (
                    "seq",
                    "rid",
                    "step",
                    "position",
                )
            )

            if (
                type(rid) is not int
                or rid < 1
                or any(
                    type(v) is not int or v < 0
                    for v in (seq, step, position)
                )
            ):
                raise TransportFailure(
                    "Invalid command metadata"
                )

            if (
                seq <= last_sequence
                or next_steps.get(rid, 0) != step
            ):
                raise TransportFailure(
                    "Out-of-order sequence or decoding step"
                )

            last_sequence = seq

            shape = _shape(cmd.get("shape"))
            dtype = _DTYPES.get(
                cmd.get("dtype")
            )

            if dtype is None:
                raise TransportFailure(
                    "Unrecognized activation dtype"
                )

            hidden = torch.empty(
                shape,
                dtype=dtype,
                device=device,
            )

            _write(conn, {
                "op": "READY",
                "seq": seq,
            })

            dist.recv(hidden, src=0)

            started = time.monotonic()

            try:
                result, cache = execute(
                    rid,
                    step,
                    position,
                    hidden,
                    caches.get(rid),
                )

                if (
                    not isinstance(result, torch.Tensor)
                    or result.ndim != 3
                    or result.device != device
                ):
                    raise ValueError(
                        "Remote model returned invalid tensor"
                    )

                caches[rid] = cache
                next_steps[rid] = step + 1

                stats["runs"] += 1
                stats["peak_caches"] = max(
                    stats["peak_caches"],
                    len(caches),
                )

                stats["trace"].append((
                    rid,
                    step,
                    started,
                    time.monotonic(),
                ))

                _write(conn, {
                    "op": "RESULT",
                    "seq": seq,
                    "rid": rid,
                    "shape": list(result.shape),
                    "dtype": str(
                        result.dtype
                    ).removeprefix("torch."),
                })

                if result.is_cuda:
                    torch.cuda.synchronize(
                        device=device
                    )

                dist.send(
                    result.contiguous(),
                    dst=0,
                )

            except Exception as exc:
                _write(conn, {
                    "op": (
                        "FATAL"
                        if isinstance(exc, FatalStageError)
                        else "ERROR"
                    ),
                    "seq": seq,
                    "rid": rid,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

                caches.pop(rid, None)
                next_steps.pop(rid, None)

    finally:
        reader.close()
        conn.close()
