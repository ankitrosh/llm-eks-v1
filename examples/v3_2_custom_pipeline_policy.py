"""Example user-defined pipeline scheduling policy."""

from llm_runtime.v3_2_pipeline_policy import PipelineDecision


class PreferStageZero:
    def select_pipeline(self, context):
        candidate = min(
            context.candidates,
            key=lambda c: c.stage,
        )
        return PipelineDecision(
            candidate.microbatch_id,
            candidate.stage,
        )
