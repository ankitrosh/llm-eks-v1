"""Stage-local KV block allocation for V3.2.

Owns block tables and reservations, not physical K/V tensors.
Caller serializes operations and settles each prepared batch.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass


class KVCapacityError(RuntimeError):
    """Insufficient free KV blocks."""


@dataclass(frozen=True)
class KVAllocation:
    request_id: int
    old_tokens: int
    new_tokens: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class KVBatchReservation:
    allocations: tuple[KVAllocation, ...]


class KVBlockManager:
    def __init__(
        self, *,
        block_size: int,
        capacity_blocks: int,
        max_context_tokens: int,
    ):
        for name, value in (
            ("block_size", block_size),
            ("capacity_blocks", capacity_blocks),
            ("max_context_tokens", max_context_tokens),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{name} must be a positive integer"
                )

        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.max_context_tokens = max_context_tokens

        self._free = deque(range(capacity_blocks))
        self._tables = {}
        self._lengths = {}
        self._pending = None

    def prepare_batch(self, next_lengths: Mapping):
        """Reserve pages atomically before model execution."""
        if self._pending is not None:
            raise RuntimeError("Previous batch not settled")

        if not isinstance(next_lengths, Mapping) or not next_lengths:
            raise ValueError("Batch must be a nonempty mapping")

        requirements = []
        extra = 0

        for rid, length in next_lengths.items():
            if type(rid) is not int or rid < 1:
                raise ValueError("Invalid request ID")

            if type(length) is not int or length < 1:
                raise ValueError("Invalid target KV length")

            old = self._lengths.get(rid, 0)

            if length <= old:
                raise ValueError("KV length must increase")

            if length > self.max_context_tokens:
                raise ValueError("Context length exceeded")

            existing = len(self._tables.get(rid, ()))
            needed = (
                length + self.block_size - 1
            ) // self.block_size

            extra += needed - existing
            requirements.append(
                (rid, old, length, existing, needed)
            )

        if extra > len(self._free):
            raise KVCapacityError(
                "Insufficient free KV blocks"
            )

        originals = {}
        allocations = []

        for rid, old, length, existing, needed in requirements:
            originals[rid] = existing
            table = self._tables.setdefault(rid, [])

            for _ in range(needed - existing):
                table.append(self._free.popleft())

            allocations.append(
                KVAllocation(
                    rid, old, length, tuple(table)
                )
            )

        receipt = KVBatchReservation(tuple(allocations))
        self._pending = (receipt, originals)

        return receipt

    def _verify(self, receipt):
        if (
            self._pending is None
            or receipt is not self._pending[0]
        ):
            raise ValueError(
                "Stale or foreign KV reservation"
            )

    def commit_batch(self, receipt):
        """Commit positions after successful execution."""
        self._verify(receipt)

        for allocation in receipt.allocations:
            self._lengths[
                allocation.request_id
            ] = allocation.new_tokens

        self._pending = None

    def rollback_batch(self, receipt):
        """Undo allocation if model execution fails."""
        self._verify(receipt)

        _, originals = self._pending
        acquired = []

        for allocation in receipt.allocations:
            rid = allocation.request_id
            table = self._tables[rid]
            previous = originals[rid]

            acquired.extend(table[previous:])
            del table[previous:]

            if previous == 0:
                del self._tables[rid]

        for block in reversed(acquired):
            self._free.appendleft(block)

        self._pending = None

    def release_request(self, request_id):
        """Release blocks after request execution finishes."""
        if self._pending is not None:
            raise RuntimeError(
                "Cannot release during in-flight batch"
            )

        if type(request_id) is not int or request_id < 1:
            raise ValueError("Invalid request ID")

        table = self._tables.pop(request_id, None)

        if table is None:
            return False

        self._lengths.pop(request_id, None)
        self._free.extend(table)
        return True

    def block_table(self, request_id):
        if self._pending is not None:
            raise RuntimeError(
                "KV reservation in progress"
            )

        return tuple(
            self._tables.get(request_id, ())
        )

    def snapshot(self):
        committed = sum(self._lengths.values())
        allocated = sum(
            len(table)
            for table in self._tables.values()
        )

        return {
            "requests": len(self._lengths),
            "free_blocks": len(self._free),
            "allocated_blocks": allocated,
            "capacity_blocks": self.capacity_blocks,
            "committed_tokens": committed,
            "allocated_token_slots": (
                allocated * self.block_size
            ),
            "fragmented_token_slots": (
                allocated * self.block_size - committed
                if self._pending is None else None
            ),
            "pending_batch": self._pending is not None,
        }
