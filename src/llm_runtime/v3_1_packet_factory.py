"""Create packet-aware V3.1 backends from user configuration.

Call after torch.distributed process-group initialization.
Custom factories are trusted Python code.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.distributed as dist

from llm_runtime.v3_1_communication_api import (
    CommunicationContext,
    CommunicationPluginError,
    CommunicationRegistry,
)
from llm_runtime.v3_1_packet_bridge import (
    TorchPacketTransport,
)
from llm_runtime.v3_1_packet_runtime import (
    PacketBackend,
    TorchDistributedByteTransport,
)
from llm_runtime.v3_1_torch_codecs import (
    ZlibTensorCodec,
)
from llm_runtime.v3_1_transport_backends import (
    create_backend,
)


def create_packet_backend(
    config,
    *,
    rank,
    world_size,
    device,
):
    """Select codec and transport for the current rank."""

    if not dist.is_initialized():
        raise CommunicationPluginError(
            "Initialize torch.distributed first"
        )

    if (
        rank != dist.get_rank()
        or world_size != dist.get_world_size()
    ):
        raise CommunicationPluginError(
            "Configuration rank/world_size mismatch"
        )

    if not isinstance(config, Mapping):
        raise CommunicationPluginError(
            "Configuration must be an object"
        )

    communication = config.get("communication", {})

    if not isinstance(communication, Mapping):
        raise CommunicationPluginError(
            "communication must be an object"
        )

    options = communication.get("options", {})

    if not isinstance(options, Mapping):
        raise CommunicationPluginError(
            "communication.options must be an object"
        )

    context = CommunicationContext(
        rank=rank,
        world_size=world_size,
        device=str(device),
        options=dict(options),
        process_group=dist.group.WORLD,
    )

    registry = CommunicationRegistry()

    registry.register_codec(
        "torch-zlib-v1",
        lambda ctx: ZlibTensorCodec(
            level=ctx.options.get("zlib_level", 3),
        ),
    )

    codec_name = communication.get(
        "codec", "identity"
    )

    codec_spec = (
        communication.get("codec_factory")
        if codec_name == "custom"
        else codec_name
    )

    codec = registry.create_codec(
        codec_spec, context
    )

    backend_name = communication.get(
        "backend", "auto"
    )

    process_backend = str(dist.get_backend()).lower()
    device_type = torch.device(device).type

    if backend_name == "auto":
        if process_backend == "nccl" and device_type == "cuda":
            backend_name = "nccl"
        elif process_backend == "gloo":
            backend_name = "gloo_staged"
        else:
            raise CommunicationPluginError(
                "Cannot automatically select transport"
            )

    if backend_name == "nccl":
        if (
            process_backend != "nccl"
            or device_type != "cuda"
            or codec_name != "identity"
        ):
            raise CommunicationPluginError(
                "Current NCCL transport requires "
                "CUDA and identity tensor encoding"
            )

        transport = TorchPacketTransport(
            create_backend("nccl", device)
        )

    elif backend_name == "gloo_staged":
        if process_backend != "gloo":
            raise CommunicationPluginError(
                "Gloo transport requires a Gloo process group"
            )

        if codec_name == "identity":
            transport = TorchPacketTransport(
                create_backend("gloo_staged", device)
            )
        else:
            # Non-identity codecs must emit byte WirePackets.
            transport = TorchDistributedByteTransport()

    elif backend_name == "custom":
        transport_spec = communication.get(
            "transport_factory"
        )

        transport = registry.create_transport(
            transport_spec, context
        )

    else:
        raise CommunicationPluginError(
            f"Unknown transport backend: {backend_name}"
        )

    return PacketBackend(
        codec=codec,
        transport=transport,
        device=device,
    )
