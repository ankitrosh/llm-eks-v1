import pytest
import torch
import torch.nn.functional as F

from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_block_attention import (
    block_decode_attention,
)


def make_cache(dtype=torch.float32):
    return BlockKVCache(
        layers=2,
        capacity_blocks=12,
        block_size=4,
        kv_heads=2,
        head_dim=8,
        max_context_tokens=32,
        dtype=dtype,
    )


def populate(cache, lengths):
    receipt = cache.prepare_batch(lengths)
    original = {}

    for rid, length in lengths.items():
        for layer in range(cache.layers):
            k = torch.randn(
                1, 2, length, 8, dtype=cache.dtype
            )
            v = torch.randn(
                1, 2, length, 8, dtype=cache.dtype
            )

            original[(rid, layer)] = (k, v)
            cache.stage_write(rid, layer, k, v)

    cache.commit_batch(receipt)
    return original


def dense_reference(q, old_k, old_v, new_k, new_v):
    k = torch.cat(
        (old_k, new_k[:, None, :]), dim=1
    )
    v = torch.cat(
        (old_v, new_v[:, None, :]), dim=1
    )

    # Two query heads per KV head.
    k = k.repeat_interleave(2, dim=0)
    v = v.repeat_interleave(2, dim=0)

    result = F.scaled_dot_product_attention(
        q[None, :, None, :],
        k[None],
        v[None],
        is_causal=False,
    )

    return result[0, :, 0, :]


@pytest.mark.parametrize("lengths", [
    {1: 3, 2: 9},
    {1: 7, 2: 4},
    {1: 1, 2: 15},
])
def test_block_attention_matches_dense(lengths):
    torch.manual_seed(42)
    torch.set_num_threads(1)

    cache = make_cache()
    original = populate(cache, lengths)

    ids = tuple(lengths)
    q = torch.randn(2, 4, 8)
    new_k = torch.randn(2, 2, 8)
    new_v = torch.randn(2, 2, 8)

    before = cache.snapshot()

    # Catch accidental materialization of whole requests.
    cache.read_request = lambda rid: (
        (_ for _ in ()).throw(
            AssertionError("Full KV materialization forbidden")
        )
    )

    actual = block_decode_attention(
        cache,
        layer=0,
        request_ids=ids,
        query=q,
        new_keys=new_k,
        new_values=new_v,
    )

    expected = torch.stack([
        dense_reference(
            q[index],
            original[(rid, 0)][0][0],
            original[(rid, 0)][1][0],
            new_k[index],
            new_v[index],
        )
        for index, rid in enumerate(ids)
    ])

    torch.testing.assert_close(
        actual, expected,
        rtol=1e-5,
        atol=1e-5,
    )

    assert cache.snapshot() == before


def test_decode_token_attends_to_itself():
    cache = make_cache()

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 2, 8)
    v = torch.randn(1, 2, 8)

    # No historical KV: only the current token exists.
    result = block_decode_attention(
        cache,
        layer=0,
        request_ids=(1,),
        query=q,
        new_keys=k,
        new_values=v,
    )

    expected = v.repeat_interleave(2, dim=1)

    torch.testing.assert_close(result, expected)


def test_bfloat16_matches_dense():
    torch.manual_seed(42)

    cache = make_cache(torch.bfloat16)
    original = populate(cache, {1: 7})

    q = torch.randn(1, 4, 8).bfloat16()
    k = torch.randn(1, 2, 8).bfloat16()
    v = torch.randn(1, 2, 8).bfloat16()

    actual = block_decode_attention(
        cache,
        layer=0,
        request_ids=(1,),
        query=q,
        new_keys=k,
        new_values=v,
    )

    expected = dense_reference(
        q[0],
        original[(1, 0)][0][0],
        original[(1, 0)][1][0],
        k[0],
        v[0],
    )

    torch.testing.assert_close(
        actual[0], expected,
        rtol=2e-2,
        atol=2e-2,
    )


def test_reject_pending_transaction():
    cache = make_cache()
    receipt = cache.prepare_batch({1: 3})

    with pytest.raises(RuntimeError):
        block_decode_attention(
            cache,
            layer=0,
            request_ids=(1,),
            query=torch.randn(1, 4, 8),
            new_keys=torch.randn(1, 2, 8),
            new_values=torch.randn(1, 2, 8),
        )

    cache.rollback_batch(receipt)


def test_reject_duplicate_requests():
    cache = make_cache()

    with pytest.raises(ValueError):
        block_decode_attention(
            cache,
            layer=0,
            request_ids=(1, 1),
            query=torch.randn(2, 4, 8),
            new_keys=torch.randn(2, 2, 8),
            new_values=torch.randn(2, 2, 8),
        )
