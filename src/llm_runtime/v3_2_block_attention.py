"""Blockwise single-token attention over stage-local KV storage.

Q and K are assumed to have positional embeddings already applied.
Supports grouped-query attention and variable history lengths.
No Qwen3 projection, RoPE or optimized CUDA kernel is included here.
"""

import math

import torch


def block_decode_attention(
    cache, *, layer, request_ids,
    query, new_keys, new_values, scale=None,
):
    """Attend to committed KV blocks plus one new token per request.

    query:      [B, query_heads, head_dim]
    new_keys:   [B, kv_heads, head_dim]
    new_values: [B, kv_heads, head_dim]

    Returns [B, query_heads, head_dim].
    This operation does not modify the cache.
    """
    if type(layer) is not int or not 0 <= layer < cache.layers:
        raise ValueError("Invalid stage-local layer index")

    if cache._pending is not None:
        raise RuntimeError("KV reservation is pending")

    ids = tuple(request_ids)

    if (
        not ids
        or any(type(r) is not int or r < 1 for r in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid or duplicate request IDs")

    if not all(
        isinstance(t, torch.Tensor)
        for t in (query, new_keys, new_values)
    ):
        raise TypeError("Q/K/V must be tensors")

    if query.ndim != 3:
        raise ValueError("Expected query [B, QH, D]")

    batch, heads, dim = query.shape

    if (
        batch != len(ids)
        or dim != cache.head_dim
        or heads < 1
        or heads % cache.kv_heads
    ):
        raise ValueError("Invalid query geometry")

    expected = (batch, cache.kv_heads, dim)

    if (
        tuple(new_keys.shape) != expected
        or tuple(new_values.shape) != expected
    ):
        raise ValueError("Invalid new K/V geometry")

    for tensor in (query, new_keys, new_values):
        if (
            tensor.dtype != cache.dtype
            or tensor.device != cache.device
        ):
            raise ValueError("Q/K/V dtype/device mismatch")

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid attention scale")

    group = heads // cache.kv_heads
    head_map = torch.arange(
        heads, device=cache.device
    ) // group

    outputs = []

    for index, rid in enumerate(ids):
        length = cache._committed.get(rid, 0)
        blocks = cache.manager.block_table(rid)

        expected_blocks = (
            length + cache.block_size - 1
        ) // cache.block_size

        if len(blocks) != expected_blocks:
            raise RuntimeError("Inconsistent KV block table")

        q = query[index].float()

        maximum = torch.full(
            (heads,), -torch.inf, device=cache.device
        )
        denominator = torch.zeros(
            heads, device=cache.device
        )
        numerator = torch.zeros(
            (heads, dim), device=cache.device
        )

        def consume(keys, values):
            nonlocal maximum, denominator, numerator

            if not all(
                isinstance(t, torch.Tensor)
                for t in (keys, values)
            ):
                raise TypeError("Invalid storage adapter output")

            if (
                keys.ndim != 3
                or values.shape != keys.shape
                or keys.shape[0] != cache.kv_heads
                or keys.shape[-1] != dim
                or keys.shape[1] < 1
            ):
                raise ValueError("Invalid KV block geometry")

            for t in (keys, values):
                if (
                    t.dtype != cache.dtype
                    or t.device != cache.device
                ):
                    raise ValueError("KV block dtype/device mismatch")

            k = keys.index_select(0, head_map).float()
            v = values.index_select(0, head_map).float()

            scores = torch.einsum(
                "hd,htd->ht", q, k
            ) * scale

            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(maximum, block_max)

            old_weight = torch.exp(maximum - new_max)
            probabilities = torch.exp(
                scores - new_max[:, None]
            )

            denominator = (
                denominator * old_weight
                + probabilities.sum(dim=-1)
            )

            numerator = (
                numerator * old_weight[:, None]
                + torch.einsum(
                    "ht,htd->hd", probabilities, v
                )
            )

            maximum = new_max

        remaining = length

        for block_id in blocks:
            count = min(remaining, cache.block_size)

            k, v = cache.storage.read(
                layer, block_id, 0, count
            )
            consume(k, v)

            remaining -= count

        if remaining:
            raise RuntimeError("Incomplete KV block table")

        # A decode token must also attend to itself.
        consume(
            new_keys[index, :, None, :],
            new_values[index, :, None, :],
        )

        outputs.append(
            (numerator / denominator[:, None])
            .to(dtype=cache.dtype)
        )

    return torch.stack(outputs, dim=0)
