"""Packet-aware V3.1 N-rank inference transport.

Supports tensor and byte payloads with encode-once preparation.
Leaves the legacy N-rank transport unchanged.
"""

from __future__ import annotations

import json
import threading

import torch
import torch.distributed as dist

from llm_runtime.v3_1_communication_api import (
    MAX_METADATA_BYTES,
    WirePacket,
)
from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    NRankTransportError,
    TorchP2PBackend,
    _recv,
    _send,
    _spec,
)


MAX_WIRE_BYTES = 512 * 1024 * 1024


def _wire_metadata(value):
    if not isinstance(value, dict):
        raise NRankTransportError("Expected packet metadata")

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NRankTransportError(
            "Invalid packet metadata"
        ) from exc

    if len(encoded) > MAX_METADATA_BYTES:
        raise NRankTransportError(
            "Packet metadata exceeds maximum"
        )

    if (
        not isinstance(value.get("codec"), str)
        or not value["codec"]
    ):
        raise NRankTransportError("Missing codec identifier")

    kind = value.get("wire_kind")

    if kind not in ("tensor", "bytes"):
        raise NRankTransportError("Invalid wire kind")

    _spec({
        "shape": value.get("shape"),
        "dtype": value.get("dtype"),
    })

    if kind == "bytes":
        length = value.get("byte_length")

        if (
            type(length) is not int
            or not 0 < length <= MAX_WIRE_BYTES
        ):
            raise NRankTransportError(
                "Invalid encoded payload length"
            )

    return dict(value)


class TorchDistributedByteTransport:
    """Gloo point-to-point transport for encoded CPU bytes.

    Uses packet metadata on the control channel for framing.
    """

    def __init__(self, *, max_bytes=MAX_WIRE_BYTES):
        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize the process group first"
            )

        if dist.get_backend() != "gloo":
            raise RuntimeError(
                "Byte transport currently requires Gloo"
            )

        self.max_bytes = max_bytes

    def send(self, packet, *, dst):
        metadata = _wire_metadata(dict(packet.metadata))

        if metadata["wire_kind"] != "bytes":
            raise NRankTransportError(
                "Byte transport requires byte payloads"
            )

        if metadata["byte_length"] > self.max_bytes:
            raise NRankTransportError(
                "Payload exceeds transport limit"
            )

        # A private, mutable CPU buffer is kept alive until the
        # blocking Gloo send returns.
        data = bytearray(packet.payload)
        tensor = torch.frombuffer(
            data, dtype=torch.uint8
        )

        dist.send(tensor, dst=dst)

    def recv(self, *, src, metadata):
        metadata = _wire_metadata(dict(metadata))

        if metadata["wire_kind"] != "bytes":
            raise NRankTransportError(
                "Byte transport requires byte payloads"
            )

        length = metadata["byte_length"]

        if length > self.max_bytes:
            raise NRankTransportError(
                "Received payload exceeds transport limit"
            )

        buffer = torch.empty(
            length, dtype=torch.uint8, device="cpu"
        )

        dist.recv(buffer, src=src)

        return WirePacket(
            buffer.numpy().tobytes(),
            metadata,
        )

    def close(self):
        # The caller owns the PyTorch process group.
        pass


class PacketBackend:
    """Connect a user codec and transport to N-rank inference."""

    def __init__(self, *, codec, transport, device):
        self.codec = codec
        self.transport = transport
        self.device = str(torch.device(device))

        # Reuse existing source-tensor validation.
        self.validator = TorchP2PBackend(self.device)

    def describe(self, tensor):
        return self.validator.describe(tensor)

    def prepare(self, tensor):
        original = self.describe(tensor)
        packet = self.codec.encode(tensor)

        if not isinstance(packet, WirePacket):
            raise NRankTransportError(
                "Codec must return WirePacket"
            )

        metadata = _wire_metadata(dict(packet.metadata))

        if (
            metadata["shape"] != original["shape"]
            or metadata["dtype"] != original["dtype"]
        ):
            raise NRankTransportError(
                "Codec changed the declared tensor signature"
            )

        return packet

    def send_packet(self, packet, dst):
        self.transport.send(packet, dst=dst)

    def recv_packet(self, src, metadata):
        metadata = _wire_metadata(metadata)

        packet = self.transport.recv(
            src=src,
            metadata=metadata,
        )

        if not isinstance(packet, WirePacket):
            raise NRankTransportError(
                "Transport must return WirePacket"
            )

        if dict(packet.metadata) != metadata:
            raise NRankTransportError(
                "Received packet metadata mismatch"
            )

        result = self.codec.decode(
            packet,
            device=self.device,
        )

        actual = self.describe(result)
        expected = _spec({
            "shape": metadata["shape"],
            "dtype": metadata["dtype"],
        })

        if actual != expected:
            raise NRankTransportError(
                "Decoded tensor signature mismatch"
            )

        return result

    def close(self):
        self.transport.close()


class PacketNRankCoordinator(NRankCoordinator):
    """Use the established coordinator with packet metadata."""

    def _run(self, ticket, tensor):
        seq = ticket.sequence
        rid = ticket.request_id
        step = ticket.step
        position = ticket.position

        packet = self.backend.prepare(tensor)
        metadata = _wire_metadata(dict(packet.metadata))
        previous = 0

        for rank in range(1, self.world_size):
            _send(self.peers[rank], {
                "op": "RUN",
                "seq": seq,
                "rid": rid,
                "step": step,
                "position": position,
                "packet": metadata,
            })

            self._expect(rank, "READY", seq=seq)

            if rank == 1:
                self.backend.send_packet(packet, dst=rank)

                with self._lock:
                    self._stats["sent"] += 1

            else:
                _send(self.peers[previous], {
                    "op": "SEND",
                    "seq": seq,
                    "dst": rank,
                })

                self._expect(
                    previous, "SENT", seq=seq
                )

                with self._lock:
                    self._stats["hops"] += 1

            response = self._expect(
                rank, "OUTPUT", seq=seq, rid=rid
            )

            metadata = _wire_metadata(
                response.get("packet")
            )

            previous = rank

        _send(self.peers[previous], {
            "op": "SEND",
            "seq": seq,
            "dst": 0,
        })

        result = self.backend.recv_packet(
            src=previous,
            metadata=metadata,
        )

        self._expect(previous, "SENT", seq=seq)

        with self._lock:
            self._stats["received"] += 1

        return result


def run_packet_nrank_worker(
    rank,
    world_size,
    conn,
    backend,
    execute,
):
    """Execute remote model stages using packet-aware transfers."""

    if (
        type(rank) is not int
        or type(world_size) is not int
        or not 1 <= rank < world_size
    ):
        raise ValueError("Invalid rank/world_size")

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
            command = _recv(reader)
            op = command.get("op")

            if op == "STOP":
                _send(conn, {"op": "STOPPED"})
                return stats

            if op == "RELEASE":
                rid = command.get("rid")

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
                    f"Invalid worker operation: {op}"
                )

            seq = command.get("seq")
            rid = command.get("rid")
            step = command.get("step")
            position = command.get("position")

            valid = (
                type(seq) is int
                and type(rid) is int
                and type(step) is int
                and type(position) is int
                and seq >= 0
                and rid >= 1
                and step >= 0
                and position >= 0
                and seq > last_seq
                and next_steps.get(rid, 0) == step
            )

            if not valid:
                _send(conn, {
                    "op": "FATAL",
                    "seq": seq,
                    "rid": rid,
                    "error": "Invalid request sequence",
                })
                raise NRankTransportError(
                    "Invalid request sequence"
                )

            last_seq = seq

            try:
                metadata = _wire_metadata(
                    command.get("packet")
                )

                _send(conn, {
                    "op": "READY",
                    "seq": seq,
                })

                incoming = backend.recv_packet(
                    src=rank - 1,
                    metadata=metadata,
                )

                output, cache = execute(
                    rank,
                    rid,
                    step,
                    position,
                    incoming,
                    caches.get(rid),
                )

                packet = backend.prepare(output)
                output_metadata = _wire_metadata(
                    dict(packet.metadata)
                )

                caches[rid] = cache
                next_steps[rid] = step + 1

                stats["runs"] += 1
                stats["peak_caches"] = max(
                    stats["peak_caches"],
                    len(caches),
                )
                stats["trace"].append((rid, step))

                _send(conn, {
                    "op": "OUTPUT",
                    "seq": seq,
                    "rid": rid,
                    "packet": output_metadata,
                })

                send_command = _recv(reader)

                destination = (
                    rank + 1
                    if rank < world_size - 1
                    else 0
                )

                if send_command != {
                    "op": "SEND",
                    "seq": seq,
                    "dst": destination,
                }:
                    raise NRankTransportError(
                        "Invalid packet send command"
                    )

                backend.send_packet(
                    packet,
                    dst=destination,
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
