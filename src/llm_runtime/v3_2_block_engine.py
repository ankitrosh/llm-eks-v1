"""Continuous Qwen3 engine with block-backed decode.

Prefill uses regular Qwen3. Completed prefills migrate to
block storage. Decode uses the tested block attention path.
"""

from math import ceil

from llm_runtime.v3_2_continuous_engine import (
    ContinuousQwen3Engine,
)
from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_qwen3_block_decode import (
    qwen3_block_decode_step,
)


class BlockBackedQwen3Engine(ContinuousQwen3Engine):
    def __init__(
        self, model, config, *,
        block_size=4,
        storage_adapter=None,
        sampler=None,
        stop_token_ids=None,
        batch_policy=None,
    ):
        if getattr(model.config, "model_type", None) != "qwen3":
            raise ValueError("Qwen3 model required")

        if type(block_size) is not int or block_size < 1:
            raise ValueError("Invalid KV block size")

        if any(
            kind != "full_attention"
            for kind in model.config.layer_types
        ):
            raise ValueError("Full-attention Qwen3 required")

        super().__init__(
            model,
            config,
            sampler=sampler,
            stop_token_ids=stop_token_ids,
            batch_policy=batch_policy,
        )

        maximum = config["pipeline"]["memory"][
            "max_total_tokens_per_request"
        ]
        active = self.planner.settings.max_active_requests

        if maximum > model.config.max_position_embeddings:
            raise ValueError("Request context exceeds model limit")

        # Conservative capacity: enough pages for all resident
        # requests at their configured maximum context.
        blocks = active * ceil(maximum / block_size)

        self.block_cache = BlockKVCache(
            layers=len(model.model.layers),
            capacity_blocks=blocks,
            block_size=block_size,
            kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            max_context_tokens=maximum,
            dtype=model.model.embed_tokens.weight.dtype,
            device=self.device,
            storage_adapter=storage_adapter,
        )

    def _migrate_completed_prefills(self, samples, caches):
        completed = {
            rid: cache
            for rid, cache in caches.items()
            if samples[rid] is not None
        }

        if not completed:
            return caches

        with self._lock:
            receipt = self.block_cache.prepare_batch({
                rid: cache.get_seq_length()
                for rid, cache in completed.items()
            })

            try:
                for rid, cache in completed.items():
                    for layer_idx, layer in enumerate(cache.layers):
                        self.block_cache.stage_write(
                            rid,
                            layer_idx,
                            layer.keys,
                            layer.values,
                        )

                self.block_cache.commit_batch(receipt)

            except Exception:
                if self.block_cache._pending is not None:
                    self.block_cache.rollback_batch(receipt)
                raise

            # Completed prefills now live exclusively in blocks.
            # Incomplete chunked prefills keep their DynamicCache.
            for rid in completed:
                self._caches.pop(rid, None)
                caches.pop(rid)

        return caches

    def _initial_prefill(self, plan):
        samples, caches, forwards, fallback = (
            super()._initial_prefill(plan)
        )

        caches = self._migrate_completed_prefills(
            samples, caches
        )
        return samples, caches, forwards, fallback

    def _partial_prefill(self, plan):
        samples, caches, forwards, fallback = (
            super()._partial_prefill(plan)
        )

        caches = self._migrate_completed_prefills(
            samples, caches
        )
        return samples, caches, forwards, fallback

    def _decode(self, plan):
        ids = tuple(item.request_id for item in plan.items)
        tokens = tuple(item.tokens[0] for item in plan.items)

        logits = qwen3_block_decode_step(
            self.model,
            self.block_cache,
            ids,
            tokens,
        )

        samples = {
            rid: self._sample(logits[index], True)
            for index, rid in enumerate(ids)
        }

        # Block-backed decode commits KV internally.
        # There is no DynamicCache update to return.
        return samples, {}, 1, 0

    def _release(self):
        for rid in self.planner.drain_releases():
            self._caches.pop(rid, None)
            self.block_cache.release_request(rid)

    def snapshot(self):
        with self._lock:
            state = super().snapshot()
            state["kv_backend"] = "block_reference"
            state["kv_blocks"] = self.block_cache.snapshot()
            state["cache_request_ids"] = tuple(sorted(
                set(self._caches)
                | set(self.block_cache._committed)
            ))
            return state
