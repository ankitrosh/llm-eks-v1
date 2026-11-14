"""CPU validation for the V3.1 lossless tensor codec."""

import torch

from llm_runtime.v3_1_communication_api import (
    CommunicationContext,
    CommunicationPluginError,
    CommunicationRegistry,
    WirePacket,
)
from llm_runtime.v3_1_torch_codecs import (
    ZlibTensorCodec,
)


registry = CommunicationRegistry()

registry.register_codec(
    "torch-zlib-v1",
    lambda context: ZlibTensorCodec(),
)

codec = registry.create_codec(
    "torch-zlib-v1",
    CommunicationContext(
        rank=0,
        world_size=4,
        device="cpu",
    ),
)

for dtype in (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int8,
    torch.uint8,
    torch.bool,
):
    original = torch.arange(
        256,
        dtype=torch.int32,
    ).reshape(16, 16).to(dtype)

    packet = codec.encode(original)

    assert packet.metadata["wire_kind"] == "bytes"
    assert packet.metadata["codec"] == "torch-zlib-v1"

    restored = codec.decode(
        packet,
        device="cpu",
    )

    assert torch.equal(restored, original)
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape

    print(f"PASS: exact tensor roundtrip for {dtype}")


original = torch.ones(
    (32, 32),
    dtype=torch.bfloat16,
)

packet = codec.encode(original)

# A corrupted compressed payload must not be accepted.
corrupted = WirePacket(
    b"x",
    {
        **packet.metadata,
        "byte_length": 1,
    },
)

try:
    codec.decode(corrupted, device="cpu")
except CommunicationPluginError:
    print("PASS: corrupted payload rejected")
else:
    raise AssertionError(
        "Corrupted payload was accepted"
    )


# Check that metadata cannot claim a different raw size.
invalid_metadata = WirePacket(
    packet.payload,
    {
        **packet.metadata,
        "raw_byte_length": 1,
    },
)

try:
    codec.decode(invalid_metadata, device="cpu")
except CommunicationPluginError:
    print("PASS: invalid tensor size rejected")
else:
    raise AssertionError(
        "Invalid tensor size was accepted"
    )


print("PASS: V3.1 lossless PyTorch codec validation")
