"""V3.1 public, transport-agnostic communication plugin contract.

Independent of PyTorch, CUDA, and Kubernetes.
Does not modify the existing N-rank transport implementation.
"""

from __future__ import annotations

import importlib
import json

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


MAX_METADATA_BYTES = 16384


class CommunicationPluginError(ValueError):
    """Invalid plugin, encoded packet, or configuration."""


def _check_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise CommunicationPluginError(
            "Packet metadata must be a mapping"
        )

    try:
        result = dict(metadata)
        encoded = json.dumps(
            result,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CommunicationPluginError(
            "Packet metadata must be JSON-serializable"
        ) from exc

    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise CommunicationPluginError(
            "Packet metadata exceeds 16 KiB"
        )

    if (
        not isinstance(result.get("codec"), str)
        or not result["codec"]
    ):
        raise CommunicationPluginError(
            "Packet metadata requires a codec name"
        )

    if result.get("wire_kind") not in ("tensor", "bytes"):
        raise CommunicationPluginError(
            "wire_kind must be tensor or bytes"
        )

    return result


@dataclass(frozen=True)
class WirePacket:
    """Prepared payload and JSON-compatible control metadata.

    The payload may be a PyTorch tensor or bytes.
    Its lifetime must extend through transport.send().
    """

    payload: Any
    metadata: Mapping[str, Any]

    def __post_init__(self):
        checked = _check_metadata(self.metadata)

        if checked["wire_kind"] == "bytes":
            if not isinstance(self.payload, bytes):
                raise CommunicationPluginError(
                    "bytes packet requires bytes payload"
                )

            if checked.get("byte_length") != len(self.payload):
                raise CommunicationPluginError(
                    "byte_length does not match payload"
                )

        object.__setattr__(self, "metadata", checked)


@dataclass(frozen=True)
class CommunicationContext:
    rank: int
    world_size: int
    device: str
    options: Mapping[str, Any] = field(default_factory=dict)
    process_group: Any = None

    def __post_init__(self):
        if (
            type(self.rank) is not int
            or type(self.world_size) is not int
            or self.world_size < 2
            or not 0 <= self.rank < self.world_size
        ):
            raise CommunicationPluginError(
                "Invalid communication rank/world_size"
            )

        if not isinstance(self.device, str) or not self.device:
            raise CommunicationPluginError(
                "Device must be a nonempty string"
            )

        if not isinstance(self.options, Mapping):
            raise CommunicationPluginError(
                "Plugin options must be a mapping"
            )


@runtime_checkable
class TensorCodec(Protocol):
    """Encode once, then decode on the destination rank."""

    def encode(self, tensor: Any) -> WirePacket:
        ...

    def decode(
        self,
        packet: WirePacket,
        *,
        device: str,
    ) -> Any:
        ...


@runtime_checkable
class PacketTransport(Protocol):
    """Transport prepared packets.

    Control-plane coordination supplies metadata to recv()
    before the data payload arrives.
    """

    def send(
        self,
        packet: WirePacket,
        *,
        dst: int,
    ) -> None:
        ...

    def recv(
        self,
        *,
        src: int,
        metadata: Mapping[str, Any],
    ) -> WirePacket:
        ...

    def close(self) -> None:
        ...


class IdentityTensorCodec:
    """Preserve the original tensor and avoid encoding copies."""

    def encode(self, tensor):
        try:
            shape = [int(dim) for dim in tensor.shape]
            dtype = str(tensor.dtype).removeprefix("torch.")
        except (AttributeError, TypeError, ValueError) as exc:
            raise CommunicationPluginError(
                "Identity codec requires a tensor-like object"
            ) from exc

        if not shape or any(dim < 1 for dim in shape):
            raise CommunicationPluginError(
                "Empty-dimensional tensor not supported"
            )

        return WirePacket(
            tensor,
            {
                "codec": "identity",
                "wire_kind": "tensor",
                "dtype": dtype,
                "shape": shape,
            },
        )

    def decode(self, packet, *, device):
        if packet.metadata.get("codec") != "identity":
            raise CommunicationPluginError(
                "Expected identity codec"
            )

        tensor = packet.payload

        return (
            tensor.to(device)
            if hasattr(tensor, "to")
            else tensor
        )


PluginFactory = Callable[[CommunicationContext], Any]


class CommunicationRegistry:
    """Register built-in or explicitly trusted custom plugins."""

    def __init__(self):
        self._codecs = {
            "identity": lambda context: IdentityTensorCodec(),
        }
        self._transports = {}

    @staticmethod
    def _register(table, name, factory):
        if (
            not isinstance(name, str)
            or not name
            or ":" in name
        ):
            raise CommunicationPluginError(
                "Invalid plugin name"
            )

        if name in table:
            raise CommunicationPluginError(
                f"Plugin already registered: {name}"
            )

        if not callable(factory):
            raise CommunicationPluginError(
                "Factory must be callable"
            )

        table[name] = factory

    def register_codec(self, name, factory):
        self._register(self._codecs, name, factory)

    def register_transport(self, name, factory):
        self._register(self._transports, name, factory)

    @staticmethod
    def _load_factory(spec):
        module, sep, symbol = spec.partition(":")

        if (
            not sep
            or not module
            or not symbol
            or ":" in symbol
        ):
            raise CommunicationPluginError(
                "Custom plugin must use module:function"
            )

        try:
            obj = importlib.import_module(module)

            for component in symbol.split("."):
                if not component.isidentifier():
                    raise CommunicationPluginError(
                        "Invalid factory symbol"
                    )

                obj = getattr(obj, component)

        except (ImportError, AttributeError) as exc:
            raise CommunicationPluginError(
                f"Cannot load plugin {spec!r}"
            ) from exc

        if not callable(obj):
            raise CommunicationPluginError(
                "Plugin factory is not callable"
            )

        return obj

    @classmethod
    def _create(cls, table, spec, context, kind):
        if not isinstance(spec, str) or not spec:
            raise CommunicationPluginError(
                f"Invalid {kind} plugin name"
            )

        if spec in table:
            factory = table[spec]
        elif ":" in spec:
            factory = cls._load_factory(spec)
        else:
            raise CommunicationPluginError(
                f"Unknown {kind} plugin: {spec}"
            )

        instance = factory(context)

        required = (
            ("encode", "decode")
            if kind == "codec"
            else ("send", "recv", "close")
        )

        if any(
            not callable(getattr(instance, method, None))
            for method in required
        ):
            raise CommunicationPluginError(
                f"Invalid {kind} contract: {spec}"
            )

        return instance

    def create_codec(self, spec, context):
        return self._create(
            self._codecs, spec, context, "codec"
        )

    def create_transport(self, spec, context):
        return self._create(
            self._transports, spec, context, "transport"
        )
