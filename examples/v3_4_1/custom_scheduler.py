"""Example user-owned vLLM scheduling policy."""

from llm_runtime.extensions.contracts import PluginSpec


def create_policy(context):
    if context.model["num_layers"] < 1:
        raise ValueError("Invalid model layout")

    return {
        "max_num_seqs": 2,
        "max_num_batched_tokens": 384,
        "scheduling_policy": "fcfs",
    }


def register_extensions(registry):
    registry.register(
        PluginSpec(
            kind="scheduler",
            name="user-capacity-policy",
            supported_backends=frozenset({"vllm"}),
            factory=create_policy,
        )
    )
