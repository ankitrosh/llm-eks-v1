import queue
import zlib

from dataclasses import dataclass

import pytest

from llm_runtime.v3_1_communication_api import (
    CommunicationContext,
    CommunicationPluginError,
    CommunicationRegistry,
    IdentityTensorCodec,
    WirePacket,
)


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str = "float32"
    value: int = 1

    def to(self, device):
        return FakeTensor(
            self.shape,
            self.dtype,
            self.value,
        )


class ZlibExampleCodec:
    """Demo codec for byte payloads, not production tensors."""

    def encode(self, data):
        compressed = zlib.compress(data)

        return WirePacket(
            compressed,
            {
                "codec": "zlib-demo",
                "wire_kind": "bytes",
                "byte_length": len(compressed),
                "original_length": len(data),
            },
        )

    def decode(self, packet, *, device):
        data = zlib.decompress(packet.payload)

        assert len(data) == packet.metadata[
            "original_length"
        ]

        return data


class MemoryBus:
    def __init__(self):
        self.channels = {}


class MemoryPacketTransport:
    def __init__(self, rank, bus):
        self.rank = rank
        self.bus = bus
        self.closed = False

    def send(self, packet, *, dst):
        assert not self.closed

        channel = self.bus.channels.setdefault(
            (self.rank, dst),
            queue.Queue(),
        )

        channel.put(packet)

    def recv(self, *, src, metadata):
        channel = self.bus.channels[
            (src, self.rank)
        ]

        packet = channel.get_nowait()

        assert packet.metadata == metadata

        return packet

    def close(self):
        self.closed = True


def test_identity_codec():
    value = FakeTensor(
        shape=(3, 5),
        dtype="float16",
        value=7,
    )

    codec = IdentityTensorCodec()
    encoded = codec.encode(value)

    assert encoded.payload is value

    assert encoded.metadata == {
        "codec": "identity",
        "wire_kind": "tensor",
        "dtype": "float16",
        "shape": [3, 5],
    }

    assert codec.decode(
        encoded, device="cpu"
    ) == value


def test_custom_codec_and_four_rank_transport():
    registry = CommunicationRegistry()
    bus = MemoryBus()

    registry.register_codec(
        "zlib-demo",
        lambda ctx: ZlibExampleCodec(),
    )

    registry.register_transport(
        "memory",
        lambda ctx: MemoryPacketTransport(
            ctx.rank, bus
        ),
    )

    codec = registry.create_codec(
        "zlib-demo",
        CommunicationContext(0, 4, "cpu"),
    )

    transports = [
        registry.create_transport(
            "memory",
            CommunicationContext(
                rank, 4, "cpu"
            ),
        )
        for rank in range(4)
    ]

    data = b"abcdef" * 200
    packet = codec.encode(data)

    for rank in range(3):
        transports[rank].send(
            packet,
            dst=rank + 1,
        )

        packet = transports[
            rank + 1
        ].recv(
            src=rank,
            metadata=packet.metadata,
        )

    assert codec.decode(
        packet, device="cpu"
    ) == data

    for transport in transports:
        transport.close()
        assert transport.closed


def test_invalid_metadata():
    with pytest.raises(
        CommunicationPluginError
    ):
        WirePacket(
            b"x",
            {
                "codec": "x",
                "wire_kind": "bytes",
                "byte_length": 2,
            },
        )

    with pytest.raises(
        CommunicationPluginError
    ):
        WirePacket(
            b"x",
            {
                "codec": "x",
                "wire_kind": "bytes",
                "byte_length": 1,
                "invalid": object(),
            },
        )


def test_invalid_registry_configuration():
    with pytest.raises(
        CommunicationPluginError
    ):
        CommunicationContext(
            rank=3,
            world_size=3,
            device="cpu",
        )

    registry = CommunicationRegistry()

    with pytest.raises(
        CommunicationPluginError
    ):
        registry.create_transport(
            "unknown",
            CommunicationContext(0, 2, "cpu"),
        )

    with pytest.raises(
        CommunicationPluginError
    ):
        registry.register_codec(
            "identity",
            lambda ctx: ZlibExampleCodec(),
        )
