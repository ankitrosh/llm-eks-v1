"""Prevent KV release while a stage is writing another request."""

from threading import Event, RLock, Thread

import torch

from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_concurrent_nrank_runtime import (
    ConcurrentNRankCPURuntime,
)


def test_cleanup_waits_for_stage_transaction():
    cache = BlockKVCache(
        layers=1,
        capacity_blocks=4,
        block_size=4,
        kv_heads=2,
        head_dim=8,
        max_context_tokens=16,
    )

    first = cache.prepare_batch({1: 2})
    values = torch.zeros(1, 2, 2, 8)
    cache.stage_write(1, 0, values, values)
    cache.commit_batch(first)

    class Planner:
        def drain_releases(self):
            return (1,)

    runtime = object.__new__(ConcurrentNRankCPURuntime)
    runtime.planner = Planner()
    runtime.caches = (cache,)
    runtime._stage_locks = (RLock(),)

    entered = Event()
    proceed = Event()
    finished = Event()
    errors = []

    def stage_worker():
        try:
            with runtime._stage_locks[0]:
                receipt = cache.prepare_batch({2: 2})
                entered.set()
                try:
                    if not proceed.wait(5):
                        raise TimeoutError("Stage not released")
                finally:
                    cache.rollback_batch(receipt)
        except Exception as exc:
            errors.append(exc)

    def cleanup_worker():
        try:
            runtime._cleanup()
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    stage = Thread(target=stage_worker)
    cleanup = Thread(target=cleanup_worker)

    try:
        stage.start()
        assert entered.wait(5)
        assert cache.snapshot()["pending_batch"]

        cleanup.start()

        # Cleanup must wait until the stage transaction ends.
        assert not finished.wait(0.1)

    finally:
        proceed.set()
        stage.join(timeout=5)

        if cleanup.ident is not None:
            cleanup.join(timeout=5)

    assert not stage.is_alive()
    assert not cleanup.is_alive()
    assert not errors, errors

    assert cache.snapshot()["allocated_blocks"] == 0
    assert not cache.snapshot()["pending_batch"]
