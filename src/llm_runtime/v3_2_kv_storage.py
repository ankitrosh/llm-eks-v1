"""Stage-local physical KV storage.

Correctness-first: reads materialize contiguous K/V tensors.
Direct block-backed attention is implemented in Gate 6C.
"""

from importlib import import_module
import re
from typing import Protocol, runtime_checkable

import torch

from llm_runtime.v3_2_kv_blocks import KVBlockManager


@runtime_checkable
class KVStorageAdapter(Protocol):
    """Physical I/O; framework retains block ownership."""

    def write(self, layer, block_id, offset, keys, values):
        ...

    def read(self, layer, block_id, start, stop):
        ...


class TorchKVBlockStorage:
    def __init__(
        self, *, layers, capacity_blocks, block_size,
        kv_heads, head_dim, dtype=torch.float32, device="cpu",
    ):
        shape = (
            layers, capacity_blocks, kv_heads,
            block_size, head_dim,
        )
        if any(type(x) is not int or x < 1 for x in shape):
            raise ValueError("Invalid KV storage dimensions")
        if dtype not in (
            torch.float16, torch.bfloat16, torch.float32
        ):
            raise ValueError("Unsupported KV dtype")

        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)

        self.keys = torch.empty(
            shape, dtype=dtype, device=self.device
        )
        self.values = torch.empty(
            shape, dtype=dtype, device=self.device
        )

    @property
    def allocated_bytes(self):
        return (
            self.keys.numel() + self.values.numel()
        ) * self.keys.element_size()

    def _bounds(self, layer, block_id, start, stop):
        if any(
            type(x) is not int
            for x in (layer, block_id, start, stop)
        ):
            raise ValueError("Block indices must be integers")
        if not (
            0 <= layer < self.shape[0]
            and 0 <= block_id < self.shape[1]
            and 0 <= start < stop <= self.shape[3]
        ):
            raise ValueError("KV block access out of bounds")

    def write(self, layer, block_id, offset, keys, values):
        if not all(
            isinstance(x, torch.Tensor)
            for x in (keys, values)
        ):
            raise TypeError("KV values must be tensors")

        if keys.shape != values.shape or keys.ndim != 3:
            raise ValueError("KV write shape mismatch")

        count = keys.shape[1]
        self._bounds(layer, block_id, offset, offset + count)

        if tuple(keys.shape) != (
            self.shape[2], count, self.shape[4]
        ):
            raise ValueError("KV write geometry mismatch")

        if any(
            x.dtype != self.dtype or x.device != self.device
            for x in (keys, values)
        ):
            raise ValueError("KV write dtype/device mismatch")

        end = offset + count
        self.keys[
            layer, block_id, :, offset:end, :
        ].copy_(keys)
        self.values[
            layer, block_id, :, offset:end, :
        ].copy_(values)

    def read(self, layer, block_id, start, stop):
        self._bounds(layer, block_id, start, stop)
        return (
            self.keys[
                layer, block_id, :, start:stop, :
            ].clone(),
            self.values[
                layer, block_id, :, start:stop, :
            ].clone(),
        )


def load_kv_storage_adapter(
    spec, *, layers, capacity_blocks, block_size,
    kv_heads, head_dim, dtype, device,
):
    options = dict(
        layers=layers,
        capacity_blocks=capacity_blocks,
        block_size=block_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    if spec is None or (
        isinstance(spec, str) and spec == "torch"
    ):
        backend = TorchKVBlockStorage(**options)

    elif isinstance(spec, str):
        if spec.count(":") != 1:
            raise ValueError("Use 'torch' or module:Class")

        module_name, class_name = spec.split(":")
        pattern = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"

        if not re.fullmatch(
            pattern, module_name
        ) or not re.fullmatch(pattern, class_name):
            raise ValueError("Invalid KV adapter path")

        obj = import_module(module_name)
        for part in class_name.split("."):
            obj = getattr(obj, part)

        backend = obj(**options)

    else:
        backend = spec

    if not isinstance(backend, KVStorageAdapter):
        raise TypeError(
            "Storage adapter must implement read/write"
        )

    return backend


class BlockKVCache:
    """Framework-owned allocation with transactional KV writes."""

    def __init__(
        self, *, layers, capacity_blocks, block_size,
        kv_heads, head_dim, max_context_tokens,
        dtype=torch.float32, device="cpu",
        storage_adapter=None,
    ):
        if any(
            type(x) is not int or x < 1
            for x in (layers, kv_heads, head_dim)
        ):
            raise ValueError("Invalid KV shape")

        if dtype not in (
            torch.float16, torch.bfloat16, torch.float32
        ):
            raise ValueError("Unsupported KV dtype")

        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        self.manager = KVBlockManager(
            block_size=block_size,
            capacity_blocks=capacity_blocks,
            max_context_tokens=max_context_tokens,
        )

        self.storage = load_kv_storage_adapter(
            storage_adapter,
            layers=layers,
            capacity_blocks=capacity_blocks,
            block_size=block_size,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=self.device,
        )

        self._pending = None
        self._committed = {}

    def prepare_batch(self, lengths):
        if self._pending is not None:
            raise RuntimeError("KV transaction in progress")

        receipt = self.manager.prepare_batch(lengths)
        self._pending = (receipt, {})
        return receipt

    def stage_write(self, request_id, layer, keys, values):
        if self._pending is None:
            raise RuntimeError("No KV reservation")

        receipt, staged = self._pending
        allocations = {
            a.request_id: a for a in receipt.allocations
        }

        if (
            type(request_id) is not int
            or request_id not in allocations
        ):
            raise ValueError("Request not in reservation")

        if (
            type(layer) is not int
            or not 0 <= layer < self.layers
        ):
            raise ValueError("Invalid stage-local layer")

        if (request_id, layer) in staged:
            raise ValueError("Duplicate KV write")

        allocation = allocations[request_id]
        delta = (
            allocation.new_tokens - allocation.old_tokens
        )

        expected = (
            1, self.kv_heads, delta, self.head_dim
        )

        if not all(
            isinstance(x, torch.Tensor)
            for x in (keys, values)
        ):
            raise TypeError("KV values must be tensors")

        if (
            tuple(keys.shape) != expected
            or tuple(values.shape) != expected
        ):
            raise ValueError(f"Expected KV shape {expected}")

        if any(
            x.dtype != self.dtype or x.device != self.device
            for x in (keys, values)
        ):
            raise ValueError("KV dtype/device mismatch")

        staged[(request_id, layer)] = (
            keys.detach().clone(),
            values.detach().clone(),
        )

    def commit_batch(self, receipt):
        if (
            self._pending is None
            or receipt is not self._pending[0]
        ):
            raise ValueError("Stale KV transaction")

        _, staged = self._pending

        for allocation in receipt.allocations:
            for layer in range(self.layers):
                if (
                    allocation.request_id, layer
                ) not in staged:
                    raise ValueError("Missing KV layer write")

        try:
            for allocation in receipt.allocations:
                rid = allocation.request_id

                for layer in range(self.layers):
                    keys, values = staged[(rid, layer)]
                    position = allocation.old_tokens

                    while position < allocation.new_tokens:
                        block_index, offset = divmod(
                            position, self.block_size
                        )
                        count = min(
                            self.block_size - offset,
                            allocation.new_tokens - position,
                        )
                        source = (
                            position - allocation.old_tokens
                        )

                        self.storage.write(
                            layer,
                            allocation.block_ids[block_index],
                            offset,
                            keys[
                                0, :, source:source + count, :
                            ],
                            values[
                                0, :, source:source + count, :
                            ],
                        )
                        position += count

            self.manager.commit_batch(receipt)

        except Exception:
            self.manager.rollback_batch(receipt)
            self._pending = None
            raise

        for allocation in receipt.allocations:
            self._committed[
                allocation.request_id
            ] = allocation.new_tokens

        self._pending = None

    def rollback_batch(self, receipt):
        if (
            self._pending is None
            or receipt is not self._pending[0]
        ):
            raise ValueError("Stale KV transaction")

        self.manager.rollback_batch(receipt)
        self._pending = None

    def read_request(self, request_id):
        if self._pending is not None:
            raise RuntimeError(
                "Read not allowed during KV reservation"
            )

        if (
            type(request_id) is not int
            or request_id not in self._committed
        ):
            raise KeyError(request_id)

        length = self._committed[request_id]
        blocks = self.manager.block_table(request_id)
        output = []

        for layer in range(self.layers):
            pieces_k = []
            pieces_v = []
            remaining = length

            for block in blocks:
                if remaining <= 0:
                    break

                take = min(self.block_size, remaining)
                k, v = self.storage.read(
                    layer, block, 0, take
                )

                expected = (
                    self.kv_heads, take, self.head_dim
                )

                if not all(
                    isinstance(x, torch.Tensor)
                    and tuple(x.shape) == expected
                    and x.dtype == self.dtype
                    and x.device == self.device
                    for x in (k, v)
                ):
                    raise ValueError(
                        "Incompatible KV adapter output"
                    )

                pieces_k.append(k)
                pieces_v.append(v)
                remaining -= take

            if remaining:
                raise RuntimeError("Incomplete KV block table")

            output.append((
                torch.cat(pieces_k, dim=1).unsqueeze(0),
                torch.cat(pieces_v, dim=1).unsqueeze(0),
            ))

        return tuple(output)

    def release_request(self, request_id):
        if self._pending is not None:
            raise RuntimeError(
                "Cannot release during KV reservation"
            )

        released = self.manager.release_request(request_id)

        if released:
            self._committed.pop(request_id, None)

        return released

    def snapshot(self):
        state = self.manager.snapshot()

        element_bytes = torch.empty(
            (), dtype=self.dtype
        ).element_size()

        state["estimated_storage_bytes"] = (
            self.layers
            * self.manager.capacity_blocks
            * self.block_size
            * self.kv_heads
            * self.head_dim
            * element_bytes
            * 2
        )

        state["physical_storage_bytes"] = getattr(
            self.storage, "allocated_bytes", None
        )

        return state
