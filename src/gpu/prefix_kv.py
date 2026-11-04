"""Prompt KV snapshot/restore for dense DynamicCache layers.

This module has no direct PyTorch import so its storage logic can
be tested using CPU-only test doubles.

The runtime must call snapshot_prefix() immediately after prefill,
before appending generated tokens.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LayerKV:
    index: int
    keys: Any
    values: Any


@dataclass(frozen=True)
class KVSnapshot:
    start_layer: int
    end_layer: int
    token_count: int
    layers: tuple[LayerKV, ...]
    nbytes: int


def _validate_dense(layer):
    if type(layer).__name__ != "DynamicLayer":
        raise NotImplementedError(
            "Prefix caching currently supports dense "
            "DynamicLayer KV caches only"
        )

    if getattr(layer, "is_sliding", False):
        raise NotImplementedError(
            "Sliding-window cache reuse is not supported"
        )


def snapshot_prefix(
    cache,
    start_layer: int,
    end_layer: int,
    token_count: int,
) -> KVSnapshot:
    """Save independent copies of prompt-only K/V tensors."""

    if not (0 <= start_layer < end_layer):
        raise ValueError("Invalid layer range")

    if token_count < 1:
        raise ValueError("A positive token count is required")

    saved = []
    total_bytes = 0

    for idx in range(start_layer, end_layer):
        layer = cache.layers[idx]
        _validate_dense(layer)

        if cache.get_seq_length(idx) != token_count:
            raise ValueError(
                f"Layer {idx}: unexpected cache length"
            )

        keys = layer.keys
        values = layer.values

        if keys is None or values is None:
            raise ValueError(f"Layer {idx}: empty KV cache")

        if (
            len(keys.shape) != 4
            or len(values.shape) != 4
            or keys.shape[-2] != token_count
            or values.shape[-2] != token_count
        ):
            raise ValueError(
                f"Layer {idx}: unsupported KV shape"
            )

        # Clone ensures the stored prefix cannot be modified
        # by subsequent decoding operations.
        key_copy = keys.detach().clone()
        value_copy = values.detach().clone()

        total_bytes += (
            key_copy.numel() * key_copy.element_size()
            + value_copy.numel() * value_copy.element_size()
        )

        saved.append(LayerKV(
            index=idx,
            keys=key_copy,
            values=value_copy,
        ))

    return KVSnapshot(
        start_layer=start_layer,
        end_layer=end_layer,
        token_count=token_count,
        layers=tuple(saved),
        nbytes=total_bytes,
    )


def restore_prefix(
    make_cache: Callable,
    snapshot: KVSnapshot,
    reuse_tokens: int,
):
    """Restore an independent cache containing only reused tokens.

    Uses DynamicCache.update() so the configured cache layers
    retain their expected internal metadata.
    """

    if not 0 < reuse_tokens <= snapshot.token_count:
        raise ValueError("Invalid reusable-prefix length")

    cache = make_cache()

    for entry in snapshot.layers:
        keys = entry.keys[..., :reuse_tokens, :].clone()
        values = entry.values[..., :reuse_tokens, :].clone()

        cache.update(
            keys,
            values,
            entry.index,
        )

        if cache.get_seq_length(entry.index) != reuse_tokens:
            raise RuntimeError(
                f"Layer {entry.index}: KV restore mismatch"
            )

    return cache
