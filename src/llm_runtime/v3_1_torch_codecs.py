"""Reference lossless tensor codec for V3.1.

CPU staging + zlib demonstrates the plugin contract.
It is not necessarily a performance optimization.
"""

from __future__ import annotations

import hashlib
import math
import zlib

import torch

from llm_runtime.v3_1_communication_api import (
    CommunicationPluginError,
    WirePacket,
)


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class ZlibTensorCodec:
    """Lossless tensor-to-bytes codec.

    Encode once per wire transfer. Decode on the receiving device.
    """

    CODEC_NAME = "torch-zlib-v1"

    def __init__(
        self,
        *,
        level=3,
        max_raw_bytes=512 * 1024 * 1024,
    ):
        if type(level) is not int or not 0 <= level <= 9:
            raise CommunicationPluginError(
                "zlib level must be 0..9"
            )

        if (
            type(max_raw_bytes) is not int
            or max_raw_bytes <= 0
        ):
            raise CommunicationPluginError(
                "max_raw_bytes must be positive"
            )

        self.level = level
        self.max_raw_bytes = max_raw_bytes

    def encode(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise CommunicationPluginError(
                "Expected a PyTorch tensor"
            )

        if tensor.layout != torch.strided:
            raise CommunicationPluginError(
                "Only dense strided tensors are supported"
            )

        dtype = str(tensor.dtype).removeprefix("torch.")

        if dtype not in _DTYPES:
            raise CommunicationPluginError(
                f"Unsupported tensor dtype: {dtype}"
            )

        shape = list(tensor.shape)

        if not shape or any(dim < 1 for dim in shape):
            raise CommunicationPluginError(
                "Empty or scalar tensors are not supported"
            )

        size = tensor.numel() * tensor.element_size()

        if size > self.max_raw_bytes:
            raise CommunicationPluginError(
                "Tensor exceeds raw byte budget"
            )

        # Blocking GPU -> CPU transfer when input is CUDA.
        host = (
            tensor.detach()
            .contiguous()
            .to("cpu", non_blocking=False)
        )

        # Raw bytes preserve BF16 without converting to FP32.
        raw = host.view(torch.uint8).numpy().tobytes()

        if len(raw) != size:
            raise CommunicationPluginError(
                "Raw tensor byte-size mismatch"
            )

        compressed = zlib.compress(raw, self.level)

        return WirePacket(
            compressed,
            {
                "codec": self.CODEC_NAME,
                "wire_kind": "bytes",
                "byte_length": len(compressed),
                "raw_byte_length": size,
                "dtype": dtype,
                "shape": shape,
                "checksum_blake2b_128": hashlib.blake2b(
                    raw,
                    digest_size=16,
                ).hexdigest(),
            },
        )

    def decode(self, packet, *, device):
        if not isinstance(packet, WirePacket):
            raise CommunicationPluginError(
                "Expected a WirePacket"
            )

        metadata = packet.metadata

        if (
            metadata.get("codec") != self.CODEC_NAME
            or metadata.get("wire_kind") != "bytes"
        ):
            raise CommunicationPluginError(
                "Unexpected codec or wire kind"
            )

        dtype = metadata.get("dtype")

        if dtype not in _DTYPES:
            raise CommunicationPluginError(
                "Unsupported encoded tensor dtype"
            )

        shape = metadata.get("shape")

        if (
            not isinstance(shape, list)
            or not shape
            or any(
                type(dim) is not int or dim < 1
                for dim in shape
            )
        ):
            raise CommunicationPluginError(
                "Invalid encoded tensor shape"
            )

        element_size = torch.empty(
            (), dtype=_DTYPES[dtype]
        ).element_size()

        expected = math.prod(shape) * element_size

        if (
            expected > self.max_raw_bytes
            or expected != metadata.get("raw_byte_length")
        ):
            raise CommunicationPluginError(
                "Decoded byte budget/shape mismatch"
            )

        if not isinstance(
            metadata.get("checksum_blake2b_128"), str
        ):
            raise CommunicationPluginError(
                "Missing payload checksum"
            )

        try:
            decompressor = zlib.decompressobj()

            raw = decompressor.decompress(
                packet.payload,
                expected + 1,
            )

        except zlib.error as exc:
            raise CommunicationPluginError(
                "Corrupted compressed packet"
            ) from exc

        if (
            len(raw) != expected
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise CommunicationPluginError(
                "Invalid or oversized compressed tensor"
            )

        digest = hashlib.blake2b(
            raw,
            digest_size=16,
        ).hexdigest()

        if digest != metadata["checksum_blake2b_128"]:
            raise CommunicationPluginError(
                "Tensor checksum mismatch"
            )

        buffer = torch.frombuffer(
            bytearray(raw),
            dtype=torch.uint8,
        ).clone()

        result = buffer.view(
            _DTYPES[dtype]
        ).reshape(shape)

        return result.to(device, non_blocking=False)
