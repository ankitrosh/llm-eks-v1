"""Compatibility between V3.1 tensor backends and the packet plugin API.

Version 1 supports identity-encoded tensor payloads only.
Variable-length/byte codecs require control-protocol integration.
"""

from __future__ import annotations

from llm_runtime.v3_1_communication_api import (
    CommunicationPluginError,
    IdentityTensorCodec,
    WirePacket,
)


class TorchPacketTransport:
    """Expose an existing tensor backend as a PacketTransport."""

    def __init__(self, backend):
        self.backend = backend

    def send(self, packet: WirePacket, *, dst: int) -> None:
        if packet.metadata["wire_kind"] != "tensor":
            raise CommunicationPluginError(
                "TorchPacketTransport requires a tensor payload"
            )

        spec = self.backend.describe(packet.payload)

        if (
            list(spec["shape"]) != list(packet.metadata["shape"])
            or spec["dtype"] != packet.metadata["dtype"]
        ):
            raise CommunicationPluginError(
                "Encoded tensor metadata differs from payload"
            )

        self.backend.send(packet.payload, dst)

    def recv(self, *, src: int, metadata):
        if metadata.get("wire_kind") != "tensor":
            raise CommunicationPluginError(
                "TorchPacketTransport requires a tensor payload"
            )

        spec = {
            "shape": list(metadata["shape"]),
            "dtype": metadata["dtype"],
        }

        tensor = self.backend.recv(src, spec)

        return WirePacket(tensor, metadata)

    def close(self) -> None:
        # Process-group lifecycle is still owned by the runtime.
        pass


class LegacyIdentityBackend:
    """Make the packet API usable with the existing N-rank runtime.

    Only identity codecs are valid in this compatibility mode:
    the existing control protocol describes the original tensor.
    """

    def __init__(
        self,
        backend,
        *,
        device: str,
        transport=None,
    ):
        self.backend = backend
        self.device = device
        self.codec = IdentityTensorCodec()
        self.transport = (
            transport
            if transport is not None
            else TorchPacketTransport(backend)
        )

    def describe(self, tensor):
        return self.backend.describe(tensor)

    def send(self, tensor, dst):
        # Encoding happens once per actual transmission.
        packet = self.codec.encode(tensor)

        if packet.metadata["codec"] != "identity":
            raise CommunicationPluginError(
                "Compatibility bridge requires identity codec"
            )

        self.transport.send(packet, dst=dst)

    def recv(self, src, spec):
        # The existing control channel supplies shape and dtype.
        # Reconstruct the identity codec's wire metadata.
        metadata = {
            "codec": "identity",
            "wire_kind": "tensor",
            "shape": list(spec["shape"]),
            "dtype": spec["dtype"],
        }

        packet = self.transport.recv(
            src=src,
            metadata=metadata,
        )

        return self.codec.decode(
            packet,
            device=self.device,
        )
