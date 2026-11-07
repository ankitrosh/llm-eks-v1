"""V3.1 startup protocol and capability negotiation.

CPU-testable validation, followed by optional torch.distributed
all-gather. Does not change frozen V3 or the transport state machine.
"""

from __future__ import annotations

PROTOCOL_VERSION = 1


class StartupProtocolError(RuntimeError):
    pass


def make_local_record(
    spec,
    rank,
    *,
    process_backend,
    device,
    codec_wire_kind=None,
    custom_transport_wire_kinds=None,
    plugins_loaded=False,
):
    config = spec.plan.as_dict()["communication"]
    transport = config["backend"]
    codec = config["codec"]

    if codec == "identity":
        wire_kind = "tensor"
    elif codec == "torch-zlib-v1":
        wire_kind = "bytes"
    elif codec == "custom":
        if not plugins_loaded or codec_wire_kind not in (
            "tensor", "bytes"
        ):
            raise StartupProtocolError(
                "Custom codec requires loaded plugin and "
                "declared wire kind"
            )
        wire_kind = codec_wire_kind
    else:
        raise StartupProtocolError(
            f"Unknown codec: {codec}"
        )

    if transport == "nccl":
        wire_kinds = ["tensor"]
    elif transport == "gloo_staged":
        wire_kinds = ["tensor", "bytes"]
    elif transport == "auto":
        wire_kinds = (
            ["tensor"]
            if process_backend == "nccl"
            else ["tensor", "bytes"]
            if process_backend == "gloo"
            else []
        )
    elif transport == "custom":
        if not plugins_loaded:
            raise StartupProtocolError(
                "Custom transport must be loaded before negotiation"
            )

        wire_kinds = list(
            custom_transport_wire_kinds or ()
        )

        if (
            not wire_kinds
            or any(
                kind not in ("tensor", "bytes")
                for kind in wire_kinds
            )
        ):
            raise StartupProtocolError(
                "Invalid custom transport capabilities"
            )
    else:
        raise StartupProtocolError(
            f"Unknown transport: {transport}"
        )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "launch_format_version": spec.as_dict()["format_version"],
        "rank": rank,
        "world_size": spec.world_size,
        "plan_sha256": spec.plan_sha256,
        "transport": transport,
        "codec": codec,
        "wire_kind": wire_kind,
        "supported_wire_kinds": wire_kinds,
        "process_backend": str(process_backend).lower(),
        "device_type": str(device).split(":")[0],
    }


def validate_startup_records(spec, records):
    """Validate collective startup declarations from every rank."""

    if (
        not isinstance(records, list)
        or len(records) != spec.world_size
    ):
        raise StartupProtocolError(
            "Startup reports do not match world size"
        )

    config = spec.plan.as_dict()["communication"]
    expected_transport = config["backend"]
    expected_codec = config["codec"]

    seen = set()
    group_backends = set()
    wire_kinds_seen = set()

    # The configured codec determines the actual payload type.
    # Transport support alone is not sufficient.
    expected_wire_kind = {
        "identity": "tensor",
        "torch-zlib-v1": "bytes",
    }.get(expected_codec)

    for record in records:
        if not isinstance(record, dict):
            raise StartupProtocolError(
                "Malformed startup record"
            )

        if record.get("error"):
            raise StartupProtocolError(
                f"Rank startup error: {record['error']}"
            )

        rank = record.get("rank")

        if (
            type(rank) is not int
            or not 0 <= rank < spec.world_size
            or rank in seen
        ):
            raise StartupProtocolError(
                "Missing, duplicate, or invalid rank"
            )

        seen.add(rank)

        if record.get("protocol_version") != PROTOCOL_VERSION:
            raise StartupProtocolError(
                f"Rank {rank}: protocol version mismatch"
            )

        if record.get("launch_format_version") != (
            spec.as_dict()["format_version"]
        ):
            raise StartupProtocolError(
                f"Rank {rank}: launch format mismatch"
            )

        if record.get("world_size") != spec.world_size:
            raise StartupProtocolError(
                f"Rank {rank}: world size mismatch"
            )

        if record.get("plan_sha256") != spec.plan_sha256:
            raise StartupProtocolError(
                f"Rank {rank}: launch plan mismatch"
            )

        if (
            record.get("transport") != expected_transport
            or record.get("codec") != expected_codec
        ):
            raise StartupProtocolError(
                f"Rank {rank}: communication configuration mismatch"
            )

        kind = record.get("wire_kind")
        supported = record.get("supported_wire_kinds")

        if (
            kind not in ("tensor", "bytes")
            or not isinstance(supported, list)
            or kind not in supported
        ):
            raise StartupProtocolError(
                f"Rank {rank}: incompatible wire capabilities"
            )

        if (
            expected_wire_kind is not None
            and kind != expected_wire_kind
        ):
            raise StartupProtocolError(
                f"Rank {rank}: codec {expected_codec} "
                f"requires {expected_wire_kind} wire payloads, "
                f"but rank declared {kind}"
            )

        wire_kinds_seen.add(kind)

        group = record.get("process_backend")
        device = record.get("device_type")

        if group not in ("gloo", "nccl"):
            raise StartupProtocolError(
                f"Rank {rank}: unsupported process group"
            )

        if device not in ("cpu", "cuda"):
            raise StartupProtocolError(
                f"Rank {rank}: unsupported device type"
            )

        if expected_transport == "nccl":
            if (
                group != "nccl"
                or device != "cuda"
                or kind != "tensor"
            ):
                raise StartupProtocolError(
                    f"Rank {rank}: NCCL capability mismatch"
                )

        elif expected_transport == "gloo_staged":
            if group != "gloo":
                raise StartupProtocolError(
                    f"Rank {rank}: Gloo capability mismatch"
                )

        elif expected_transport == "auto":
            if group == "nccl" and (
                device != "cuda" or kind != "tensor"
            ):
                raise StartupProtocolError(
                    f"Rank {rank}: invalid automatic NCCL selection"
                )

        group_backends.add(group)

    if len(seen) != spec.world_size:
        raise StartupProtocolError(
            "Not all ranks reported capabilities"
        )

    if len(group_backends) != 1:
        raise StartupProtocolError(
            "Ranks disagree on process-group backend"
        )

    if len(wire_kinds_seen) != 1:
        raise StartupProtocolError(
            "Ranks disagree on encoded wire payload type"
        )

    return True


def negotiate_startup(
    spec,
    *,
    device,
    codec_wire_kind=None,
    custom_transport_wire_kinds=None,
    plugins_loaded=False,
):
    """Collect and validate capabilities before running stages.

    Requires an initialized, trusted torch.distributed process group.
    Its configured timeout bounds the collective.
    """

    import torch.distributed as dist

    if not dist.is_initialized():
        raise StartupProtocolError(
            "Distributed process group is not initialized"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != spec.world_size:
        raise StartupProtocolError(
            "Process group and launch plan world sizes differ"
        )

    try:
        local = make_local_record(
            spec,
            rank,
            process_backend=dist.get_backend(),
            device=device,
            codec_wire_kind=codec_wire_kind,
            custom_transport_wire_kinds=(
                custom_transport_wire_kinds
            ),
            plugins_loaded=plugins_loaded,
        )
    except Exception as exc:
        # Report local capability failures to other ranks instead
        # of raising before they reach the collective.
        local = {
            "rank": rank,
            "error": f"{type(exc).__name__}: {exc}",
        }

    records = [None] * world_size
    dist.all_gather_object(records, local)

    validate_startup_records(spec, records)

    return records
