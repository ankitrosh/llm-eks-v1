"""CPU correctness-first continuous Qwen3 execution engine."""

from threading import Lock, RLock

import torch

from llm_runtime.v3_2_batch_planner import ContinuousBatchPlanner
from llm_runtime.v3_2_qwen3_padding import prepare_prefill
from llm_runtime.v3_2_qwen3_cache_batch import (
    batched_decode_step,
    split_caches,
)


class ContinuousQwen3Engine:
    def __init__(
        self, model, config, *,
        sampler=None, stop_token_ids=None,
    ):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.planner = ContinuousBatchPlanner(config)
        self.sampler = sampler or (
            lambda logits: int(logits.argmax().item())
        )

        if stop_token_ids is None:
            eos = model.config.eos_token_id
            stop_token_ids = (
                () if eos is None
                else (eos,) if isinstance(eos, int)
                else tuple(eos)
            )

        self.stop_tokens = set(stop_token_ids)
        self._lock = RLock()
        self._step_lock = Lock()
        self._next_id = 1
        self._caches = {}
        self._generated = {}
        self._finished = {}
        self._fatal = None
        self._closed = False
        self._forwards = 0
        self._batched_forwards = 0
        self._fallback_forwards = 0

        layer_types = getattr(
            model.config, "layer_types", None
        )
        if layer_types and any(
            x != "full_attention" for x in layer_types
        ):
            raise ValueError(
                "Only full-attention Qwen3 supported"
            )

    def _check_healthy(self):
        if self._fatal is not None:
            raise RuntimeError(
                "Inference engine unhealthy; restart required"
            ) from self._fatal
        if self._closed:
            raise RuntimeError("Inference engine closed")

    def _release(self):
        for rid in self.planner.drain_releases():
            self._caches.pop(rid, None)

    def submit(self, prompt_tokens, max_new_tokens):
        with self._lock:
            self._check_healthy()
            rid = self._next_id

            self.planner.submit(
                rid, prompt_tokens, max_new_tokens
            )

            self._next_id += 1
            self._generated[rid] = []
            return rid

    def cancel(self, request_id):
        with self._lock:
            self._check_healthy()
            cancelled = self.planner.cancel(request_id)
            if cancelled:
                self._generated.pop(request_id, None)
                self._release()
            return cancelled

    def result(self, request_id):
        with self._lock:
            return self._finished.get(request_id)

    def snapshot(self):
        with self._lock:
            state = self.planner.snapshot()
            return {
                **state,
                "healthy": self._fatal is None,
                "cache_request_ids": tuple(
                    sorted(self._caches)
                ),
                "model_forwards": self._forwards,
                "batched_forwards": self._batched_forwards,
                "sequential_prefill_forwards": (
                    self._fallback_forwards
                ),
            }

    def _sample(self, logits, enabled):
        if not enabled:
            return None
        token = self.sampler(logits)
        if type(token) is not int or token < 0:
            raise ValueError(
                "Sampler must return a nonnegative Python int"
            )
        return token

    def _initial_prefill(self, plan):
        items = plan.items
        batch = prepare_prefill(
            [x.tokens for x in items],
            pad_token_id=(
                self.model.config.pad_token_id or 0
            ),
            device=self.device,
        )

        output = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=True,
        )

        caches = split_caches(
            output.past_key_values,
            tuple(len(x.tokens) for x in items),
            self.model.config,
        )

        samples = {}
        updated = {}

        for index, item in enumerate(items):
            samples[item.request_id] = self._sample(
                output.logits[index, -1, :],
                item.emits_token,
            )
            updated[item.request_id] = caches[index]

        return samples, updated, 1, 0

    def _partial_prefill(self, plan):
        samples = {}
        updated = {}

        for item in plan.items:
            rid = item.request_id
            old = self._caches.get(rid)

            if item.position and old is None:
                raise RuntimeError(
                    "Missing request KV cache"
                )

            ids = torch.tensor(
                [item.tokens],
                dtype=torch.long,
                device=self.device,
            )

            kwargs = {
                "input_ids": ids,
                "use_cache": True,
            }

            if old is not None:
                if old.get_seq_length() != item.position:
                    raise RuntimeError(
                        "KV cache position mismatch"
                    )

                end = item.position + len(item.tokens)
                positions = torch.arange(
                    item.position, end,
                    device=self.device,
                )

                kwargs.update({
                    "past_key_values": old,
                    "attention_mask": torch.ones(
                        (1, end),
                        dtype=torch.long,
                        device=self.device,
                    ),
                    "position_ids": positions.unsqueeze(0),
                    "cache_position": positions,
                })

            output = self.model(**kwargs)
            cache = output.past_key_values

            if cache.get_seq_length() != (
                item.position + len(item.tokens)
            ):
                raise RuntimeError("Incorrect prefill KV length")

            updated[rid] = cache
            samples[rid] = self._sample(
                output.logits[0, -1, :],
                item.emits_token,
            )

        return samples, updated, len(plan.items), len(plan.items)

    def _decode(self, plan):
        caches = []
        tokens = []

        for item in plan.items:
            cache = self._caches[item.request_id]

            if cache.get_seq_length() != item.position:
                raise RuntimeError(
                    "Decode KV cache position mismatch"
                )

            caches.append(cache)
            tokens.append(item.tokens[0])

        logits, updated_caches = batched_decode_step(
            self.model, caches, tokens
        )

        return (
            {
                item.request_id: self._sample(
                    logits[index], True
                )
                for index, item in enumerate(plan.items)
            },
            {
                item.request_id: updated_caches[index]
                for index, item in enumerate(plan.items)
            },
            1,
            0,
        )

    def advance(self):
        """Execute one planned batch; return its BatchUpdate."""
        with self._step_lock:
            with self._lock:
                self._check_healthy()
                plan = self.planner.next_batch()

            if plan is None:
                return None

            try:
                with torch.inference_mode():
                    if plan.phase == "decode":
                        output = self._decode(plan)
                    elif all(
                        item.position == 0
                        for item in plan.items
                    ):
                        output = self._initial_prefill(plan)
                    else:
                        output = self._partial_prefill(plan)

                samples, caches, forwards, fallback = output

                with self._lock:
                    self._caches.update(caches)
                    stop_ids = {
                        rid for rid, token in samples.items()
                        if token in self.stop_tokens
                    }

                    update = self.planner.complete_batch(
                        samples, stop_ids=stop_ids
                    )

                    for rid, token in update.emitted.items():
                        self._generated[rid].append(token)

                    for rid in update.finished:
                        self._finished[rid] = tuple(
                            self._generated.pop(rid)
                        )

                    for rid in update.cancelled:
                        self._generated.pop(rid, None)

                    self._release()

                    self._forwards += forwards
                    self._fallback_forwards += fallback

                    if len(plan.items) > 1 and forwards == 1:
                        self._batched_forwards += 1

                    return update

            except Exception as exc:
                with self._lock:
                    self._fatal = exc
                raise
