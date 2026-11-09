"""V3.2 reference N-rank continuous inference runtime.

Gate 7C.1:
- Continuous planner controls request lifecycle.
- V3.1 stage executors handle single-request prefill.
- V3.2 block caches own KV for subsequent decode.
- V3.2 microbatch transport handles batched stage execution.
- Single outstanding transport batch for correctness isolation.

Not yet a distributed/NCCL serving runtime.
"""

import torch

from llm_runtime.v3_pipeline_scheduler import StageTask
from llm_runtime.v3_2_batch_planner import ContinuousBatchPlanner
from llm_runtime.v3_2_microbatch_transport import (
    CPUMicrobatchTransport,
)
from llm_runtime.v3_2_qwen3_stage_batch import (
    stage_block_decode,
    validate_stage_alignment,
)


class NRankCPUContinuousRuntime:
    def __init__(
        self,
        stages,
        caches,
        config,
        *,
        batch_policy=None,
        stop_token_ids=(),
    ):
        self.stages = tuple(stages)
        self.caches = tuple(caches)

        if len(self.stages) < 2 or (
            len(self.stages) != len(self.caches)
        ):
            raise ValueError("Invalid stage/cache layout")

        for rank, (stage, cache) in enumerate(
            zip(self.stages, self.caches)
        ):
            if (
                stage.rank != rank
                or stage.layout.world_size != len(self.stages)
                or cache.layers != len(stage.model.owned_layers)
            ):
                raise ValueError("Stage layout mismatch")

        self.planner = ContinuousBatchPlanner(
            config,
            batch_policy=batch_policy,
        )

        self.stop_token_ids = tuple(stop_token_ids)
        self._next_id = 0
        self._prompts = {}
        self._generated = {}
        self._finished = set()
        self._cancelled = set()
        self._failed = False

        self._prefill_limit = config["scheduler"][
            "max_batch_tokens"
        ]

        workers = []

        for stage, cache in zip(self.stages, self.caches):

            def run(packet, stage=stage, cache=cache):
                positions = tuple(
                    cache._committed.get(rid)
                    for rid in packet.request_ids
                )

                if positions != packet.positions:
                    raise RuntimeError(
                        "Stage-local KV position mismatch"
                    )

                return stage_block_decode(
                    stage,
                    cache,
                    packet.request_ids,
                    packet.payload,
                )

            workers.append(run)

        self.transport = CPUMicrobatchTransport(
            workers,
            max_pending=1,
        )

    def submit(self, prompt, max_new_tokens):
        if self._failed:
            raise RuntimeError("Runtime failed; restart required")

        prompt = tuple(prompt)

        if (
            not prompt
            or any(
                type(token) is not int or token < 0
                for token in prompt
            )
        ):
            raise ValueError("Invalid prompt")

        if len(prompt) > self._prefill_limit:
            raise ValueError(
                "Gate 7C.1 requires unchunked prefill"
            )

        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise ValueError("Invalid output-token count")

        rid = self._next_id + 1

        self.planner.submit(
            rid,
            prompt,
            max_new_tokens=max_new_tokens,
        )

        self._next_id = rid
        self._prompts[rid] = prompt
        self._generated[rid] = []
        return rid

    def _prefill_one(self, rid, prompt):
        payload = torch.tensor(
            [prompt],
            dtype=torch.long,
            device=self.stages[0].device,
        )

        for stage, cache in zip(
            self.stages, self.caches
        ):
            with torch.inference_mode():
                result = stage(
                    StageTask(rid, 0, 0, payload),
                    None,
                )

            payload = result.payload
            dynamic = result.cache_state

            receipt = cache.prepare_batch({
                rid: len(prompt)
            })

            try:
                for local_index, global_index in enumerate(
                    stage.model.owned_layers
                ):
                    layer = dynamic.layers[global_index]

                    cache.stage_write(
                        rid,
                        local_index,
                        layer.keys,
                        layer.values,
                    )

                cache.commit_batch(receipt)

            except Exception:
                if cache._pending is not None:
                    cache.rollback_batch(receipt)
                raise

        return payload[0, -1, :].float()

    def _cleanup(self):
        for rid in self.planner.drain_releases():
            for cache in self.caches:
                cache.release_request(rid)

    def advance(self):
        if self._failed:
            raise RuntimeError("Runtime failed; restart required")

        plan = self.planner.next_batch()

        if plan is None:
            self._cleanup()
            return None

        ids = tuple(
            item.request_id for item in plan.items
        )

        try:
            if plan.phase == "prefill":
                logits = []

                for item in plan.items:
                    rid = item.request_id
                    tokens = tuple(item.tokens)

                    if (
                        rid in self.caches[0]._committed
                        or tokens != self._prompts[rid]
                    ):
                        raise RuntimeError(
                            "Chunked/repeated prefill is not "
                            "supported in Gate 7C.1"
                        )

                    logits.append(
                        self._prefill_one(rid, tokens)
                    )

                output = torch.stack(logits)

            elif plan.phase == "decode":
                positions = validate_stage_alignment(
                    self.caches,
                    ids,
                )

                tokens = tuple(
                    item.tokens[0]
                    for item in plan.items
                )

                sequence = self.transport.submit(
                    ids, positions, tokens
                )

                output = self.transport.wait(sequence)

            else:
                raise RuntimeError(
                    f"Unknown batch phase: {plan.phase}"
                )

            samples = {
                rid: int(output[index].argmax())
                for index, rid in enumerate(ids)
            }

            update = self.planner.complete_batch(
                samples,
                stop_ids=self.stop_token_ids,
            )

            for rid, token in samples.items():
                self._generated[rid].append(token)

            self._finished.update(update.finished)
            self._cleanup()
            return update

        except Exception:
            # KV may already have advanced on some stages.
            # Do not resume or reuse this runtime.
            self._failed = True
            self.transport.close()
            raise

    def cancel(self, request_id):
        if self._failed:
            raise RuntimeError("Runtime failed; restart required")

        changed = self.planner.cancel(request_id)

        if changed:
            self._cancelled.add(request_id)
            self._cleanup()

        return changed

    def result(self, request_id):
        if request_id not in self._finished:
            return None

        if request_id in self._cancelled:
            return None

        return tuple(self._generated[request_id])

    def snapshot(self):
        return {
            "failed": self._failed,
            "finished": tuple(sorted(self._finished)),
            "cancelled": tuple(sorted(self._cancelled)),
            "transport": self.transport.snapshot(),
            "kv_blocks": tuple(
                cache.snapshot() for cache in self.caches
            ),
        }

    def close(self):
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
