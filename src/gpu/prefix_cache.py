"""Bounded exact-token prefix cache with LRU eviction.

Stores opaque, immutable KV snapshots. The inference runtime is
responsible for cloning/cropping tensors before using them.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class PrefixHit:
    payload: Any
    matched_tokens: int
    cached_tokens: int
    model_key: str
    namespace: str


@dataclass(frozen=True)
class _Entry:
    tokens: tuple[int, ...]
    payload: Any
    nbytes: int
    model_key: str
    namespace: str


class PrefixCache:
    def __init__(
        self,
        max_bytes: int,
        max_entries: int = 4,
        min_reuse_tokens: int = 16,
    ):
        if max_bytes < 0 or max_entries < 1 or min_reuse_tokens < 1:
            raise ValueError("Invalid prefix-cache limits")

        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.min_reuse_tokens = min_reuse_tokens

        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._bytes = 0

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.reused_tokens = 0

    @staticmethod
    def _tokens(ids: Sequence[int]) -> tuple[int, ...]:
        result = tuple(ids)

        if any(type(x) is not int or x < 0 for x in result):
            raise ValueError(
                "Token IDs must be nonnegative Python integers"
            )

        return result

    @property
    def used_bytes(self) -> int:
        return self._bytes

    @property
    def size(self) -> int:
        return len(self._entries)

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        model_key: str,
        namespace: str = "default",
    ) -> PrefixHit | None:
        """Find the longest exact prefix.

        Retain at least one token for prefill/logit computation.

        If matched_tokens < cached_tokens, the inference runtime
        must crop a private copy of the KV snapshot.
        """
        tokens = self._tokens(token_ids)
        max_match = len(tokens) - 1

        best_key = None
        best_len = 0

        if max_match >= self.min_reuse_tokens:
            # Search most-recently-used entries first.
            for key, entry in reversed(self._entries.items()):
                if entry.model_key != model_key:
                    continue

                if entry.namespace != namespace:
                    continue

                length = 0

                for a, b in zip(
                    entry.tokens,
                    tokens[:max_match],
                ):
                    if a != b:
                        break
                    length += 1

                if length > best_len:
                    best_key = key
                    best_len = length

                if best_len == max_match:
                    break

        if best_key is None or best_len < self.min_reuse_tokens:
            self.misses += 1
            return None

        self._entries.move_to_end(best_key)
        entry = self._entries[best_key]

        self.hits += 1
        self.reused_tokens += best_len

        return PrefixHit(
            payload=entry.payload,
            matched_tokens=best_len,
            cached_tokens=len(entry.tokens),
            model_key=entry.model_key,
            namespace=entry.namespace,
        )

    def put(
        self,
        token_ids: Sequence[int],
        payload: Any,
        *,
        nbytes: int,
        model_key: str,
        namespace: str = "default",
    ) -> bool:
        """Store a prompt-only, immutable KV snapshot.

        nbytes represents the GPU memory retained exclusively
        by this snapshot. The manager does not clone tensors.
        """
        tokens = self._tokens(token_ids)

        if not tokens or nbytes <= 0 or not model_key or not namespace:
            raise ValueError(
                "Tokens, model identity, namespace and bytes required"
            )

        # Oversized entries should not evict existing useful data.
        if self.max_bytes == 0 or nbytes > self.max_bytes:
            return False

        key = (model_key, namespace, tokens)

        # Replace an existing entry for the same exact prefix.
        if key in self._entries:
            old = self._entries.pop(key)
            self._bytes -= old.nbytes

        # Evict least-recently-used entries as necessary.
        while self._entries and (
            self._bytes + nbytes > self.max_bytes
            or len(self._entries) >= self.max_entries
        ):
            _, removed = self._entries.popitem(last=False)
            self._bytes -= removed.nbytes
            self.evictions += 1

        self._entries[key] = _Entry(
            tokens=tokens,
            payload=payload,
            nbytes=nbytes,
            model_key=model_key,
            namespace=namespace,
        )

        self._bytes += nbytes
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def stats(self) -> dict[str, int]:
        return {
            "entries": self.size,
            "used_bytes": self.used_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "reused_tokens": self.reused_tokens,
        }
