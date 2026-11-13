"""Selectable tensor-transfer backends for V3.1."""

import torch
import torch.distributed as dist

from .v3_1_nrank_transport import (
    TorchP2PBackend,
    _spec,
)


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TorchGlooStagedBackend:
    """Gloo P2P with explicit CPU staging.

    Model tensors remain on their original device.
    Only transmitted activations are copied to CPU.
    """

    def __init__(self, device):
        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize process group first"
            )

        if str(dist.get_backend()) != "gloo":
            raise RuntimeError(
                "Gloo process group required"
            )

        self.device = torch.device(device)
        self.native = TorchP2PBackend(device)

    def describe(self, tensor):
        return self.native.describe(tensor)

    def send(self, tensor, dst):
        self.describe(tensor)

        # Blocking CUDA -> CPU copy.
        # This ensures the CPU buffer is ready.
        cpu_tensor = (
            tensor.detach()
            .to(device="cpu", non_blocking=False)
            .contiguous()
        )

        dist.send(cpu_tensor, dst=dst)

    def recv(self, src, spec):
        spec = _spec(spec)

        cpu_tensor = torch.empty(
            tuple(spec["shape"]),
            dtype=_DTYPES[spec["dtype"]],
            device="cpu",
        )

        dist.recv(cpu_tensor, src=src)

        # Blocking CPU -> model device copy.
        return cpu_tensor.to(
            self.device,
            non_blocking=False,
        )


def create_backend(name, device):
    if name == "nccl":
        if str(dist.get_backend()) != "nccl":
            raise RuntimeError(
                "NCCL process group required"
            )
        return TorchP2PBackend(device)

    if name == "gloo_staged":
        return TorchGlooStagedBackend(device)

    raise ValueError(
        f"Unsupported transport backend: {name}"
    )
