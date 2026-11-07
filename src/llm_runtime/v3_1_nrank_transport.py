"""V3.1 ordered N-rank control and activation routing.

Data path: rank0 -> rank1 -> ... -> rankN-1 -> rank0.
Rank0 owns one control socket per remote rank.
Tensor transfers use the supplied P2P backend.
"""

from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field


class NRankTransportError(RuntimeError):
    pass


_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


def _send(conn, message):
    frame = (
        json.dumps(message, separators=(",", ":")) + "\n"
    ).encode("utf8")

    if len(frame) > 4096:
        raise NRankTransportError("Oversized control frame")

    conn.sendall(frame)


def _recv(reader):
    frame = reader.readline(4097)

    if (
        not frame
        or len(frame) > 4096
        or not frame.endswith(b"\n")
    ):
        raise NRankTransportError(
            "Invalid or closed control connection"
        )

    try:
        result = json.loads(frame)
    except ValueError as exc:
        raise NRankTransportError(
            "Invalid JSON control frame"
        ) from exc

    if type(result) is not dict:
        raise NRankTransportError(
            "Control frame must be an object"
        )

    return result


def _spec(spec, max_bytes=512 * 1024 * 1024):
    if type(spec) is not dict:
        raise NRankTransportError("Missing tensor spec")

    shape = spec.get("shape")
    dtype = spec.get("dtype")

    if (
        type(shape) is not list
        or not 1 <= len(shape) <= 8
        or any(
            type(dim) is not int or dim < 1
            for dim in shape
        )
        or type(dtype) is not str
        or dtype not in _DTYPE_BYTES
    ):
        raise NRankTransportError(
            "Invalid tensor shape or dtype"
        )

    if math.prod(shape) * _DTYPE_BYTES[dtype] > max_bytes:
        raise NRankTransportError(
            "Tensor exceeds transfer limit"
        )

    return {
        "shape": list(shape),
        "dtype": dtype,
    }


@dataclass
class NRankTicket:
    request_id: int
    step: int
    position: int
    sequence: int
    future: Future = field(
        default_factory=Future,
        repr=False,
    )


class NRankCoordinator:
    """Rank0 coordinator for N-rank tensor forwarding.

    Backend contract:
      describe(tensor) -> {"shape": [...], "dtype": "..."}
      send(tensor, dst=rank)
      recv(src=rank, spec=...) -> tensor
    """

    def __init__(
        self,
        peers,
        backend,
        *,
        max_pending=4,
        timeout_s=30,
    ):
        if (
            type(max_pending) is not int
            or max_pending < 1
            or not isinstance(peers, dict)
            or not peers
            or set(peers) != set(range(1, len(peers) + 1))
        ):
            raise ValueError(
                "Expected contiguous remote ranks 1..N-1"
            )

        self.world_size = len(peers) + 1
        self.peers = peers

        self.readers = {
            rank: conn.makefile("rb")
            for rank, conn in peers.items()
        }

        self.backend = backend
        self.timeout_s = timeout_s
        self.max_pending = max_pending

        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._tickets = {}

        self._next = 0
        self._error = None
        self._closed = False

        self._stats = {
            "sent": 0,
            "received": 0,
            "releases": 0,
            "hops": 0,
        }

        self._thread = threading.Thread(
            target=self._io,
            daemon=True,
            name="nrank-rank0-io",
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
            type(request_id) is not int
            or request_id < 1
            or any(
                type(x) is not int or x < 0
                for x in (step, position)
            )
        ):
            raise ValueError("Invalid request metadata")

        _spec(self.backend.describe(tensor))

        with self._lock:
            if self._closed or self._error:
                raise NRankTransportError(
                    self._error or "Transport closed"
                )

            if len(self._tickets) >= self.max_pending:
                raise NRankTransportError(
                    "Pending activation limit reached"
                )

            ticket = NRankTicket(
                request_id,
                step,
                position,
                self._next,
            )

            self._next += 1
            self._tickets[ticket.sequence] = ticket

            self._queue.put((
                "RUN",
                ticket,
                tensor,
            ))

            return ticket

    def await_result(self, ticket):
        try:
            return ticket.future.result(
                timeout=self.timeout_s
            )
        except FutureTimeout as exc:
            self._trip("N-rank result timed out")
            raise NRankTransportError(
                "N-rank result timed out"
            ) from exc

    def release_request(self, request_id):
        with self._lock:
            if not self._closed and not self._error:
                self._queue.put((
                    "RELEASE",
                    request_id,
                    None,
                ))

    def stats(self):
        with self._lock:
            return dict(self._stats)

    def _trip(self, reason):
        with self._lock:
            if self._error is None:
                self._error = str(reason)

            tickets = list(self._tickets.values())
            self._tickets.clear()

        for ticket in tickets:
            if not ticket.future.done():
                ticket.future.set_exception(
                    NRankTransportError(str(reason))
                )

    def _expect(self, rank, op, **fields):
        response = _recv(self.readers[rank])

        if response.get("op") in ("ERROR", "FATAL"):
            raise NRankTransportError(
                f"Rank {rank}: "
                f"{response.get('error', 'execution failed')}"
            )

        if (
            response.get("op") != op
            or any(
                response.get(key) != value
                for key, value in fields.items()
            )
        ):
            raise NRankTransportError(
                f"Unexpected reply from rank {rank}: "
                f"{response}"
            )

        return response

    def _run(self, ticket, tensor):
        seq = ticket.sequence
        rid = ticket.request_id
        step = ticket.step
        position = ticket.position

        previous = 0
        payload_spec = _spec(
            self.backend.describe(tensor)
        )

        for rank in range(1, self.world_size):
            _send(self.peers[rank], {
                "op": "RUN",
                "seq": seq,
                "rid": rid,
                "step": step,
                "position": position,
                "spec": payload_spec,
            })

            self._expect(
                rank,
                "READY",
                seq=seq,
            )

            if rank == 1:
                self.backend.send(
                    tensor,
                    dst=rank,
                )

                with self._lock:
                    self._stats["sent"] += 1

            else:
                _send(self.peers[previous], {
                    "op": "SEND",
                    "seq": seq,
                    "dst": rank,
                })

                self._expect(
                    previous,
                    "SENT",
                    seq=seq,
                )

                with self._lock:
                    self._stats["hops"] += 1

            output = self._expect(
                rank,
                "OUTPUT",
                seq=seq,
                rid=rid,
            )

            payload_spec = _spec(
                output.get("spec")
            )

            previous = rank

        # Final rank returns logits directly to rank0.
        _send(self.peers[previous], {
            "op": "SEND",
            "seq": seq,
            "dst": 0,
        })

        result = self.backend.recv(
            src=previous,
            spec=payload_spec,
        )

        self._expect(
            previous,
            "SENT",
            seq=seq,
        )

        with self._lock:
            self._stats["received"] += 1

        return result

    def _io(self):
        try:
            while True:
                op, obj, tensor = self._queue.get()

                if op == "STOP":
                    for rank in self.peers:
                        _send(
                            self.peers[rank],
                            {"op": "STOP"},
                        )

                    for rank in self.peers:
                        self._expect(rank, "STOPPED")

                    return

                if op == "RELEASE":
                    for rank in self.peers:
                        _send(self.peers[rank], {
                            "op": "RELEASE",
                            "rid": obj,
                        })

                        self._expect(
                            rank,
                            "RELEASED",
                            rid=obj,
                        )

                    with self._lock:
                        self._stats["releases"] += 1

                    continue

                ticket = obj
                result = self._run(ticket, tensor)

                if not ticket.future.done():
                    ticket.future.set_result(result)

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
        with self._lock:
            was_closed = self._closed
            self._closed = True
            healthy = self._error is None

        if not was_closed and healthy:
            self._queue.put((
                "STOP",
                None,
                None,
            ))

        self._thread.join(timeout=self.timeout_s)

        if self._thread.is_alive():
            self._trip(
                "N-rank IO thread failed to stop"
            )
            raise NRankTransportError(
                "N-rank communication thread stuck"
            )

        for rank in self.peers:
            try:
                self.readers[rank].close()
                self.peers[rank].close()
            except OSError:
                pass


def run_nrank_worker(
    rank,
    world_size,
    conn,
    backend,
    execute,
):
    """Run one remote stage with per-request KV caches.

    execute(rank, rid, step, position, tensor, cache)
        -> (output_tensor, updated_cache)
    """
    if (
        type(rank) is not int
        or type(world_size) is not int
        or not 1 <= rank < world_size
    ):
        raise ValueError("Invalid worker rank")

    reader = conn.makefile("rb")

    caches = {}
    next_steps = {}
    last_seq = -1

    stats = {
        "runs": 0,
        "releases": 0,
        "peak_caches": 0,
        "trace": [],
    }

    try:
        while True:
            cmd = _recv(reader)
            op = cmd.get("op")

            if op == "STOP":
                _send(conn, {"op": "STOPPED"})
                return stats

            if op == "RELEASE":
                rid = cmd.get("rid")

                if type(rid) is not int or rid < 1:
                    raise NRankTransportError(
                        "Invalid release ID"
                    )

                caches.pop(rid, None)
                next_steps.pop(rid, None)

                stats["releases"] += 1

                _send(conn, {
                    "op": "RELEASED",
                    "rid": rid,
                })
                continue

            if op != "RUN":
                raise NRankTransportError(
                    f"Invalid remote operation: {op}"
                )

            seq = cmd.get("seq")
            rid = cmd.get("rid")
            step = cmd.get("step")
            position = cmd.get("position")

            if (
                type(rid) is not int
                or rid < 1
                or any(
                    type(x) is not int or x < 0
                    for x in (seq, step, position)
                )
                or seq <= last_seq
                or next_steps.get(rid, 0) != step
            ):
                _send(conn, {
                    "op": "FATAL",
                    "seq": seq,
                    "rid": rid,
                    "error": "Invalid sequence or request step",
                })
                raise NRankTransportError(
                    "Invalid sequence or request step"
                )

            last_seq = seq

            try:
                incoming_spec = _spec(
                    cmd.get("spec")
                )

                _send(conn, {
                    "op": "READY",
                    "seq": seq,
                })

                incoming = backend.recv(
                    src=rank - 1,
                    spec=incoming_spec,
                )

                output, cache = execute(
                    rank,
                    rid,
                    step,
                    position,
                    incoming,
                    caches.get(rid),
                )

                output_spec = _spec(
                    backend.describe(output)
                )

                caches[rid] = cache
                next_steps[rid] = step + 1

                stats["runs"] += 1
                stats["peak_caches"] = max(
                    stats["peak_caches"],
                    len(caches),
                )
                stats["trace"].append(
                    (rid, step)
                )

                _send(conn, {
                    "op": "OUTPUT",
                    "seq": seq,
                    "rid": rid,
                    "spec": output_spec,
                })

                send_cmd = _recv(reader)

                expected_dst = (
                    rank + 1
                    if rank < world_size - 1
                    else 0
                )

                if send_cmd != {
                    "op": "SEND",
                    "seq": seq,
                    "dst": expected_dst,
                }:
                    raise NRankTransportError(
                        "Invalid send instruction"
                    )

                backend.send(
                    output,
                    dst=expected_dst,
                )

                _send(conn, {
                    "op": "SENT",
                    "seq": seq,
                })

            except BaseException as exc:
                try:
                    _send(conn, {
                        "op": "FATAL",
                        "seq": seq,
                        "rid": rid,
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    })
                except OSError:
                    pass

                raise

    finally:
        reader.close()
        conn.close()


def open_nrank_control(
    rank,
    world_size,
    host,
    port,
    timeout_s=60,
):
    """Rank0 receives a peer dictionary; workers receive one socket."""
    if (
        type(rank) is not int
        or type(world_size) is not int
        or not 0 <= rank < world_size
        or world_size < 2
    ):
        raise ValueError("Invalid process topology")

    if rank == 0:
        listener = socket.socket()
        listener.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        listener.bind(("0.0.0.0", port))
        listener.listen(world_size - 1)
        listener.settimeout(timeout_s)

        peers = {}

        try:
            while len(peers) < world_size - 1:
                conn, _ = listener.accept()
                conn.settimeout(timeout_s)

                reader = conn.makefile("rb")

                try:
                    hello = _recv(reader)
                    remote_rank = hello.get("rank")

                    if (
                        hello.get("op") != "HELLO"
                        or type(remote_rank) is not int
                        or not 1 <= remote_rank < world_size
                        or remote_rank in peers
                    ):
                        raise NRankTransportError(
                            "Invalid or duplicate rank handshake"
                        )

                    _send(conn, {
                        "op": "WELCOME",
                        "rank": remote_rank,
                    })

                    peers[remote_rank] = conn

                finally:
                    reader.close()

        except BaseException:
            for conn in peers.values():
                conn.close()
            raise

        finally:
            listener.close()

        return peers

    deadline = time.monotonic() + timeout_s

    while True:
        try:
            conn = socket.create_connection(
                (host, port),
                timeout=2,
            )
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)

    conn.settimeout(timeout_s)

    _send(conn, {
        "op": "HELLO",
        "rank": rank,
    })

    reader = conn.makefile("rb")

    try:
        if _recv(reader) != {
            "op": "WELCOME",
            "rank": rank,
        }:
            raise NRankTransportError(
                "Rank handshake failed"
            )
    finally:
        reader.close()

    return conn


class TorchP2PBackend:
    """PyTorch P2P backend: Gloo CPU or NCCL CUDA."""

    def __init__(self, device):
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize torch process group first"
            )

        self.torch = torch
        self.dist = dist
        self.device = torch.device(device)

        self.dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

    def describe(self, tensor):
        if (
            not isinstance(tensor, self.torch.Tensor)
            or tensor.device != self.device
            or tensor.layout != self.torch.strided
        ):
            raise NRankTransportError(
                "Expected dense tensor on transport device"
            )

        dtype_name = str(
            tensor.dtype
        ).removeprefix("torch.")

        return _spec({
            "shape": list(tensor.shape),
            "dtype": dtype_name,
        })

    def send(self, tensor, dst):
        self.describe(tensor)

        tensor = tensor.contiguous()

        if tensor.is_cuda:
            self.torch.cuda.synchronize(
                tensor.device
            )

        self.dist.send(tensor, dst=dst)

    def recv(self, src, spec):
        spec = _spec(spec)

        result = self.torch.empty(
            tuple(spec["shape"]),
            dtype=self.dtypes[spec["dtype"]],
            device=self.device,
        )

        self.dist.recv(result, src=src)
        return result
