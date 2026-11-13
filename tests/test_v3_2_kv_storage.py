import pytest
import torch

from llm_runtime.v3_2_kv_storage import (
    BlockKVCache,
    TorchKVBlockStorage,
)


def make(**kwargs):
    return BlockKVCache(
        layers=2,
        capacity_blocks=6,
        block_size=4,
        kv_heads=2,
        head_dim=3,
        max_context_tokens=16,
        **kwargs,
    )


def tensors(count, value, dtype=torch.float32):
    return (
        torch.full(
            (1, 2, count, 3), float(value), dtype=dtype
        ),
        torch.full(
            (1, 2, count, 3), float(value + 100),
            dtype=dtype,
        ),
    )


def stage(store, receipt):
    for allocation in receipt.allocations:
        count = (
            allocation.new_tokens - allocation.old_tokens
        )
        for layer in range(store.layers):
            store.stage_write(
                allocation.request_id,
                layer,
                *tensors(
                    count,
                    allocation.request_id * 10 + layer,
                ),
            )


def test_incremental_roundtrip():
    store = make()

    r = store.prepare_batch({1: 3})
    stage(store, r)
    store.commit_batch(r)

    r = store.prepare_batch({1: 7})
    stage(store, r)
    store.commit_batch(r)

    assert len(store.manager.block_table(1)) == 2

    keys, values = store.read_request(1)[0]

    torch.testing.assert_close(
        keys[:, :, :3, :], tensors(3, 10)[0]
    )
    torch.testing.assert_close(
        keys[:, :, 3:, :], tensors(4, 10)[0]
    )
    torch.testing.assert_close(
        values[:, :, :3, :], tensors(3, 10)[1]
    )

    assert store.release_request(1)
    assert store.snapshot()["allocated_blocks"] == 0


def test_request_isolation_and_reuse():
    store = make()

    r = store.prepare_batch({1: 3, 2: 5})
    stage(store, r)
    store.commit_batch(r)

    assert store.read_request(1)[0][0][0, 0, 0, 0] == 10
    assert store.read_request(2)[0][0][0, 0, 0, 0] == 20

    store.release_request(1)

    r = store.prepare_batch({3: 2})
    stage(store, r)
    store.commit_batch(r)

    assert store.read_request(3)[0][0][0, 0, 0, 0] == 30
    assert store.read_request(2)[0][0][0, 0, 0, 0] == 20


def test_rollback_preserves_committed_kv():
    store = make()

    r = store.prepare_batch({1: 3})
    stage(store, r)
    store.commit_batch(r)

    before = store.read_request(1)

    r = store.prepare_batch({1: 8, 2: 4})
    stage(store, r)
    store.rollback_batch(r)

    after = store.read_request(1)

    for (ak, av), (bk, bv) in zip(before, after):
        torch.testing.assert_close(ak, bk)
        torch.testing.assert_close(av, bv)

    assert store.snapshot()["allocated_blocks"] == 1


def test_missing_layer_rejected():
    store = make()
    r = store.prepare_batch({1: 3})

    store.stage_write(1, 0, *tensors(3, 10))

    with pytest.raises(ValueError, match="Missing"):
        store.commit_batch(r)

    with pytest.raises(RuntimeError):
        store.read_request(1)

    store.rollback_batch(r)
    assert store.snapshot()["allocated_blocks"] == 0


def test_capacity_rejection_is_atomic():
    store = BlockKVCache(
        layers=2,
        capacity_blocks=2,
        block_size=4,
        kv_heads=2,
        head_dim=3,
        max_context_tokens=16,
    )

    with pytest.raises(RuntimeError):
        store.prepare_batch({1: 5, 2: 5})

    assert store.snapshot()["allocated_blocks"] == 0


def test_physical_memory_accounting():
    store = make()
    expected = 2 * 6 * 4 * 2 * 3 * 4 * 2

    assert store.snapshot()["physical_storage_bytes"] == expected
    assert store.snapshot()["estimated_storage_bytes"] == expected


def test_custom_storage_adapter():
    store = make(
        storage_adapter=(
            "examples.v3_2_custom_kv_storage:"
            "CountingKVStorage"
        )
    )

    r = store.prepare_batch({1: 5})
    stage(store, r)
    store.commit_batch(r)

    assert store.storage.write_calls == 4
    assert store.read_request(1)[0][0].shape == (1, 2, 5, 3)


def test_read_returns_independent_tensors():
    store = make()

    r = store.prepare_batch({1: 3})
    stage(store, r)
    store.commit_batch(r)

    first = store.read_request(1)
    first[0][0].zero_()

    assert store.read_request(1)[0][0].sum() > 0


def test_bfloat16_storage():
    store = make(dtype=torch.bfloat16)
    r = store.prepare_batch({1: 3})

    for layer in range(2):
        store.stage_write(
            1, layer,
            *tensors(3, 10 + layer, torch.bfloat16),
        )

    store.commit_batch(r)

    assert store.read_request(1)[0][0].dtype == torch.bfloat16


def test_cannot_release_inflight_reservation():
    store = make()
    r = store.prepare_batch({1: 3})
    stage(store, r)

    with pytest.raises(RuntimeError):
        store.release_request(1)

    store.rollback_batch(r)

    with pytest.raises(ValueError):
        store.commit_batch(r)


def test_storage_bounds():
    storage = TorchKVBlockStorage(
        layers=1,
        capacity_blocks=1,
        block_size=4,
        kv_heads=2,
        head_dim=3,
    )

    keys, values = tensors(2, 1)

    with pytest.raises(ValueError):
        storage.write(0, 1, 0, keys[0], values[0])


def test_real_qwen3_cache_roundtrip():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    torch.set_num_threads(1)

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 2,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).eval()

    store = BlockKVCache(
        layers=2,
        capacity_blocks=8,
        block_size=2,
        kv_heads=2,
        head_dim=16,
        max_context_tokens=12,
    )

    with torch.inference_mode():
        out = model(
            torch.tensor([[2, 5, 8]]),
            use_cache=True,
        )

        r = store.prepare_batch({1: 3})

        for i, layer in enumerate(
            out.past_key_values.layers
        ):
            store.stage_write(
                1, i, layer.keys, layer.values
            )

        store.commit_batch(r)

        for (k, v), layer in zip(
            store.read_request(1),
            out.past_key_values.layers,
        ):
            torch.testing.assert_close(k, layer.keys)
            torch.testing.assert_close(v, layer.values)

        token = int(out.logits[0, -1].argmax())

        out = model(
            torch.tensor([[token]]),
            past_key_values=out.past_key_values,
            use_cache=True,
        )

        r = store.prepare_batch({1: 4})

        for i, layer in enumerate(
            out.past_key_values.layers
        ):
            store.stage_write(
                1, i,
                layer.keys[:, :, -1:, :],
                layer.values[:, :, -1:, :],
            )

        store.commit_batch(r)

        for (k, v), layer in zip(
            store.read_request(1),
            out.past_key_values.layers,
        ):
            torch.testing.assert_close(k, layer.keys)
            torch.testing.assert_close(v, layer.values)
