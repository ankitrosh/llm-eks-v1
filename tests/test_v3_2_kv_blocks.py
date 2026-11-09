import pytest

from llm_runtime.v3_2_kv_blocks import (
    KVBlockManager,
    KVCapacityError,
)


def manager(blocks=4, size=4, context=16):
    return KVBlockManager(
        block_size=size,
        capacity_blocks=blocks,
        max_context_tokens=context,
    )


def test_growing_chunks_and_boundary():
    m = manager()

    r = m.prepare_batch({1: 3})
    assert len(r.allocations[0].block_ids) == 1
    m.commit_batch(r)

    r = m.prepare_batch({1: 4})
    m.commit_batch(r)
    assert len(m.block_table(1)) == 1

    r = m.prepare_batch({1: 5})
    m.commit_batch(r)

    assert len(m.block_table(1)) == 2
    assert m.snapshot()["fragmented_token_slots"] == 3


def test_atomic_capacity_rejection():
    m = manager(blocks=2)
    before = m.snapshot()

    with pytest.raises(KVCapacityError):
        m.prepare_batch({1: 8, 2: 4})

    assert m.snapshot() == before


def test_rollback_reuses_all_pages():
    m = manager()

    first = m.prepare_batch({1: 8, 2: 4})
    blocks = [
        a.block_ids for a in first.allocations
    ]

    m.rollback_batch(first)
    assert m.snapshot()["free_blocks"] == 4

    again = m.prepare_batch({1: 8, 2: 4})

    assert blocks == [
        a.block_ids for a in again.allocations
    ]

    m.commit_batch(again)


def test_growth_rollback_preserves_committed():
    m = manager()

    a = m.prepare_batch({1: 4})
    m.commit_batch(a)
    original = m.block_table(1)

    more = m.prepare_batch({1: 9})
    m.rollback_batch(more)

    assert m.block_table(1) == original
    assert m.snapshot()["committed_tokens"] == 4


def test_release_and_reuse():
    m = manager(blocks=2)

    a = m.prepare_batch({1: 8})
    m.commit_batch(a)

    assert m.snapshot()["free_blocks"] == 0
    assert m.release_request(1)
    assert not m.release_request(1)

    b = m.prepare_batch({2: 8})
    m.commit_batch(b)

    assert m.snapshot()["requests"] == 1


def test_pending_rejects_release():
    m = manager()
    a = m.prepare_batch({1: 2})

    with pytest.raises(RuntimeError):
        m.release_request(1)

    with pytest.raises(RuntimeError):
        m.prepare_batch({2: 3})

    m.rollback_batch(a)


def test_receipts_are_one_shot():
    m = manager()

    a = m.prepare_batch({1: 3})
    m.commit_batch(a)

    with pytest.raises(ValueError):
        m.commit_batch(a)

    b = m.prepare_batch({1: 4})

    with pytest.raises(ValueError):
        m.rollback_batch(a)

    m.commit_batch(b)


@pytest.mark.parametrize(
    "bad", [0, -1, True, 2.5]
)
def test_invalid_capacity_params(bad):
    with pytest.raises(ValueError):
        KVBlockManager(
            block_size=bad,
            capacity_blocks=4,
            max_context_tokens=16,
        )


def test_reject_decrease_and_overcontext():
    m = manager()

    a = m.prepare_batch({1: 4})
    m.commit_batch(a)

    with pytest.raises(ValueError):
        m.prepare_batch({1: 4})

    with pytest.raises(ValueError):
        m.prepare_batch({1: 17})

    assert not m.snapshot()["pending_batch"]


def test_stage_local_isolation():
    stage0 = manager()
    stage1 = manager()

    a = stage0.prepare_batch({1: 4})
    stage0.commit_batch(a)

    assert stage0.snapshot()["requests"] == 1
    assert stage1.snapshot()["requests"] == 0

    b = stage1.prepare_batch({1: 4})
    stage1.commit_batch(b)

    stage0.release_request(1)

    assert stage1.snapshot()["requests"] == 1
