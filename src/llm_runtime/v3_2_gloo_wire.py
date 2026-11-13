"""Versioned CPU/Gloo microbatch transport framing.

Only numeric metadata and tensors are transmitted.
Operations are blocking; asynchronous dispatch comes later.
"""

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist


_MAGIC = 0x56333247
_VERSION = 2
_WIDTH = 13

_KINDS = {"run": 1, "result": 2, "stop": 3, "release": 4, "release_ack": 5, "error": 6}
_PHASES = {"prefill": 1, "decode": 2}

_MAX_REQUESTS = 64
_MAX_TOKENS = 8192
_MAX_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class WireMicrobatch:
    sequence: int
    phase: str
    request_ids: tuple[int, ...]
    positions: tuple[int, ...]
    lengths: tuple[int, ...]
    payload: torch.Tensor


@dataclass(frozen=True)
class WireRelease:
    sequence: int
    request_ids: tuple[int, ...]


def validate_wire_release(control):
    if not isinstance(control, WireRelease):
        raise TypeError("Expected WireRelease")

    ids = control.request_ids

    if type(control.sequence) is not int or control.sequence < 1:
        raise ValueError("Invalid release sequence")

    if (
        not isinstance(ids, tuple)
        or not 0 < len(ids) <= _MAX_REQUESTS
        or any(
            type(rid) is not int or rid < 1
            for rid in ids
        )
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid release request IDs")

    return control


@dataclass(frozen=True)
class WireFailure:
    sequence: int
    rank: int
    code: int


def validate_wire_failure(control):
    if not isinstance(control, WireFailure):
        raise TypeError("Expected WireFailure")

    if (
        type(control.sequence) is not int
        or control.sequence < 1
        or type(control.rank) is not int
        or control.rank < 1
        or type(control.code) is not int
        or control.code not in (1, 2)
    ):
        raise ValueError("Invalid failure metadata")

    return control


def validate_wire_microbatch(packet):
    if not isinstance(packet, WireMicrobatch):
        raise TypeError("Expected WireMicrobatch")

    ids = packet.request_ids
    positions = packet.positions
    lengths = packet.lengths

    if (
        type(packet.sequence) is not int
        or packet.sequence < 1
    ):
        raise ValueError("Invalid sequence")

    if packet.phase not in _PHASES:
        raise ValueError("Invalid phase")

    if not (
        0 < len(ids) <= _MAX_REQUESTS
        and len(ids) == len(positions) == len(lengths)
    ):
        raise ValueError("Invalid metadata lengths")

    if (
        any(type(x) is not int or x < 1 for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid request IDs")

    minimum = 0 if packet.phase == "prefill" else 1

    if any(
        type(x) is not int or x < minimum
        for x in positions
    ):
        raise ValueError("Invalid KV positions")

    if any(
        type(x) is not int
        or not 1 <= x <= _MAX_TOKENS
        for x in lengths
    ):
        raise ValueError("Invalid token lengths")

    if (
        sum(lengths) > _MAX_TOKENS
        or (
            packet.phase == "decode"
            and any(x != 1 for x in lengths)
        )
    ):
        raise ValueError("Invalid token budget")

    tensor = packet.payload

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.ndim not in (2, 3)
    ):
        raise ValueError(
            "Expected CPU float32 [B,H] or [B,T,H]"
        )

    if (
        tensor.shape[0] != len(ids)
        or any(x <= 0 for x in tensor.shape)
        or tensor.numel() > _MAX_ELEMENTS
    ):
        raise ValueError("Invalid tensor geometry")

    if (
        tensor.ndim == 3
        and tensor.shape[1] != max(lengths)
    ):
        raise ValueError(
            "Padded width must match longest sequence"
        )

    return packet


def send_wire(dst, *, kind, packet=None, control=None):
    if kind not in _KINDS:
        raise ValueError("Unknown wire operation")

    if kind == "error":
        if packet is not None:
            raise ValueError("ERROR cannot carry a tensor")

        validate_wire_failure(control)

        header = [
            _MAGIC, _VERSION, _KINDS[kind],
            control.sequence, control.rank, control.code,
        ] + [0] * (_WIDTH - 6)

        dist.send(
            torch.tensor(header, dtype=torch.int64),
            dst=dst,
        )
        return

    if kind in ("release", "release_ack"):
        if packet is not None:
            raise ValueError(
                "Release frames cannot carry tensors"
            )

        validate_wire_release(control)

        header = [
            _MAGIC,
            _VERSION,
            _KINDS[kind],
            control.sequence,
            0,
            len(control.request_ids),
        ] + [0] * (_WIDTH - 6)

        dist.send(
            torch.tensor(header, dtype=torch.int64),
            dst=dst,
        )

        dist.send(
            torch.tensor(
                control.request_ids,
                dtype=torch.int64,
            ),
            dst=dst,
        )
        return

    if control is not None:
        raise ValueError("Unexpected control payload")

    if kind == "stop":
        if packet is not None:
            raise ValueError(
                "STOP cannot carry a payload"
            )

        header = [
            _MAGIC, _VERSION, _KINDS[kind]
        ] + [0] * (_WIDTH - 3)

        dist.send(
            torch.tensor(header, dtype=torch.int64),
            dst=dst,
        )
        return

    validate_wire_microbatch(packet)

    if kind == "run" and packet.payload.ndim != 3:
        raise ValueError("RUN expects [B,T,H]")

    if kind == "result" and packet.payload.ndim != 2:
        raise ValueError("RESULT expects [B,V]")

    dims = list(packet.payload.shape)
    dims += [0] * (3 - len(dims))

    header = [
        _MAGIC,
        _VERSION,
        _KINDS[kind],
        packet.sequence,
        _PHASES[packet.phase],
        len(packet.request_ids),
        packet.payload.ndim,
        *dims,
        1,  # float32
        sum(packet.lengths),
        0,
    ]

    dist.send(
        torch.tensor(header, dtype=torch.int64),
        dst=dst,
    )

    metadata = torch.tensor(
        list(packet.request_ids)
        + list(packet.positions)
        + list(packet.lengths),
        dtype=torch.int64,
    )

    dist.send(metadata, dst=dst)
    dist.send(packet.payload.contiguous(), dst=dst)


def recv_wire(src):
    header = torch.empty(
        _WIDTH, dtype=torch.int64
    )

    dist.recv(header, src=src)

    (
        magic, version, kind, sequence,
        phase, count, ndim, d0, d1, d2,
        dtype_code, tokens, reserved,
    ) = map(int, header.tolist())

    if (
        magic != _MAGIC
        or version != _VERSION
        or kind not in (1, 2, 3, 4, 5, 6)
        or reserved != 0
    ):
        raise ValueError("Invalid wire frame")

    if kind == 3:
        if any((
            sequence, phase, count, ndim,
            d0, d1, d2, dtype_code, tokens
        )):
            raise ValueError("Invalid STOP frame")

        return "stop", None

    if kind == 6:
        if any((ndim, d0, d1, d2, dtype_code, tokens)):
            raise ValueError("Invalid ERROR frame")

        failure = WireFailure(
            sequence, phase, count
        )
        validate_wire_failure(failure)

        return "error", failure

    if kind in (4, 5):
        if (
            not 1 <= count <= _MAX_REQUESTS
            or sequence < 1
            or any((
                phase, ndim, d0, d1, d2,
                dtype_code, tokens
            ))
        ):
            raise ValueError("Invalid release frame")

        ids = torch.empty(
            count, dtype=torch.int64
        )
        dist.recv(ids, src=src)

        control = WireRelease(
            sequence,
            tuple(map(int, ids.tolist())),
        )

        validate_wire_release(control)

        return (
            "release" if kind == 4 else "release_ack",
            control,
        )

    if (
        (kind == 1 and ndim != 3)
        or (kind == 2 and ndim != 2)
    ):
        raise ValueError("Invalid payload rank")

    if not (
        1 <= count <= _MAX_REQUESTS
        and d0 == count
        and d1 >= 1
        and d2 >= 0
        and dtype_code == 1
    ):
        raise ValueError("Invalid tensor header")

    shape = (
        (d0, d1, d2)
        if ndim == 3
        else (d0, d1)
    )

    if (
        (ndim == 3 and d2 < 1)
        or math.prod(shape) > _MAX_ELEMENTS
        or not 1 <= tokens <= _MAX_TOKENS
    ):
        raise ValueError("Wire frame exceeds limits")

    if phase not in (1, 2) or sequence < 1:
        raise ValueError("Invalid phase or sequence")

    metadata = torch.empty(
        3 * count, dtype=torch.int64
    )

    dist.recv(metadata, src=src)

    data = torch.empty(
        shape, dtype=torch.float32
    )

    dist.recv(data, src=src)

    ids = tuple(
        int(x) for x in metadata[:count]
    )
    positions = tuple(
        int(x) for x in metadata[count:2 * count]
    )
    lengths = tuple(
        int(x) for x in metadata[2 * count:]
    )

    packet = WireMicrobatch(
        sequence,
        "prefill" if phase == 1 else "decode",
        ids,
        positions,
        lengths,
        data,
    )

    validate_wire_microbatch(packet)

    if sum(lengths) != tokens:
        raise ValueError("Wire token count mismatch")

    return (
        "run" if kind == 1 else "result",
        packet,
    )
