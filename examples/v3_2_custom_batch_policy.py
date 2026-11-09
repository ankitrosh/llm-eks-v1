"""Example of a user-provided scheduling adapter."""

from llm_runtime.v3_2_policy_api import BatchDecision


class ShortPrefillFirst:
    def select_batch(self, context):
        prefill = [
            c for c in context.candidates
            if c.phase == "prefill"
        ]
        decode = [
            c for c in context.candidates
            if c.phase == "decode"
        ]

        if prefill and decode:
            chosen = (
                decode if context.last_phase == "prefill"
                else prefill
            )
        else:
            chosen = prefill or decode

        if not chosen:
            raise ValueError("No eligible requests")

        phase = chosen[0].phase

        if phase == "prefill":
            chosen = sorted(
                chosen, key=lambda c: c.ready_tokens
            )

        limit = min(
            context.max_requests_per_batch,
            context.max_batch_tokens,
        )

        return BatchDecision(
            phase=phase,
            request_ids=tuple(
                c.request_id for c in chosen[:limit]
            ),
        )
