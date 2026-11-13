"""Optimized CUDA decode attention over V3.2 block KV storage.

The existing block allocator and transactional writes are unchanged.
CPU and custom storage adapters retain the reference implementation.
"""

import math
import os

import torch
import torch.nn.functional as F

from llm_runtime.v3_2_block_attention import (
    block_decode_attention as reference_attention,
)
from llm_runtime.v3_2_kv_storage import TorchKVBlockStorage


_parity_checked = False


def block_decode_attention_fast(
    cache, *, layer, request_ids,
    query, new_keys, new_values, scale=None,
):
    global _parity_checked

    # Preserve compatibility with CPU tests and custom adapters.
    if (
        not query.is_cuda
        or not isinstance(cache.storage, TorchKVBlockStorage)
    ):
        return reference_attention(
            cache,
            layer=layer,
            request_ids=request_ids,
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            scale=scale,
        )

    ids = tuple(request_ids)

    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid request IDs")

    if any(type(rid) is not int or rid < 1 for rid in ids):
        raise ValueError("Invalid request ID")

    if type(layer) is not int or not 0 <= layer < cache.layers:
        raise ValueError("Invalid layer")

    if cache._pending is not None:
        raise RuntimeError("KV transaction in progress")

    if query.ndim != 3:
        raise ValueError("Invalid query dimensions")

    batch, query_heads, head_dim = query.shape

    if batch != len(ids) or head_dim != cache.head_dim:
        raise ValueError("Query geometry mismatch")

    kv_heads = cache.kv_heads

    if query_heads % kv_heads != 0:
        raise ValueError("Invalid GQA head geometry")

    expected = (batch, kv_heads, head_dim)

    if (
        tuple(new_keys.shape) != expected
        or tuple(new_values.shape) != expected
    ):
        raise ValueError("New KV geometry mismatch")

    for tensor in (query, new_keys, new_values):
        if (
            tensor.dtype != cache.dtype
            or tensor.device != cache.device
        ):
            raise ValueError("KV dtype/device mismatch")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid attention scale")

    storage = cache.storage
    outputs = []

    for index, rid in enumerate(ids):
        length = cache._committed.get(rid, 0)
        block_ids = cache.manager.block_table(rid)

        expected_blocks = (
            length + cache.block_size - 1
        ) // cache.block_size

        if len(block_ids) != expected_blocks:
            raise RuntimeError("Inconsistent KV block table")

        # Gather all historical blocks in one operation per
        # K/V tensor, rather than launching individual kernels
        # and Python softmax updates for every block.
        if length:
            block_indices = torch.as_tensor(
                block_ids,
                dtype=torch.long,
                device=cache.device,
            )

            past_k = (
                storage.keys[layer]
                .index_select(0, block_indices)
                .transpose(0, 1)
                .reshape(kv_heads, -1, head_dim)
                [:, :length, :]
            )

            past_v = (
                storage.values[layer]
                .index_select(0, block_indices)
                .transpose(0, 1)
                .reshape(kv_heads, -1, head_dim)
                [:, :length, :]
            )

            keys = torch.cat(
                (past_k, new_keys[index, :, None, :]),
                dim=1,
            )

            values = torch.cat(
                (past_v, new_values[index, :, None, :]),
                dim=1,
            )

        else:
            keys = new_keys[index, :, None, :]
            values = new_values[index, :, None, :]

        # One-token decode query attends to all previous tokens
        # plus itself. No triangular causal mask is needed.
        q = query[index][None, :, None, :]

        out = F.scaled_dot_product_attention(
            q,
            keys[None, :, :, :],
            values[None, :, :, :],
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )

        outputs.append(out[0, :, 0, :])

    result = torch.stack(outputs, dim=0)

    # One real-GPU numerical parity check per process.
    # This is intentionally outside the steady-state hot path.
    if (
        not _parity_checked
        and os.getenv("V32_ATTN_PARITY_CHECK", "1") == "1"
    ):
        reference = reference_attention(
            cache,
            layer=layer,
            request_ids=ids,
            query=query,
            new_keys=new_keys,
            new_values=new_values,
            scale=scale,
        )

        torch.testing.assert_close(
            result.float(),
            reference.float(),
            rtol=0.03,
            atol=0.03,
        )

        error = (
            result.float() - reference.float()
        ).abs().max().item()

        print(
            f"AB_FAST_ATTENTION_PARITY_OK "
            f"max_abs_error={error:.6f} "
            f"context={cache._committed.get(ids[0], 0)}",
            flush=True,
        )

        _parity_checked = True

    return result
