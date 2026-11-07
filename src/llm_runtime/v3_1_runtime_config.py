"""V3.1 user configuration and memory-aware partition resolution.

CPU-only. Does not load models, allocate CUDA memory, or deploy Pods.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from llm_runtime.v3_1_partition_planner import (
    StagePlacement,
    plan_contiguous_layers,
)
from llm_runtime.providers.v3_1_qwen3_layout import (
    Qwen3NRankLayout,
)


@dataclass(frozen=True)
class V31ResolvedPlan:
    model_id: str
    num_ranks: int
    partition_mode: str
    layout: Qwen3NRankLayout
    placements: tuple[StagePlacement, ...]
    transport_preference: str
    codec: str

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "world_size": self.num_ranks,
            "partition_mode": self.partition_mode,
            "layer_boundaries": list(
                self.layout.boundaries
            ),
            "communication": {
                "backend": self.transport_preference,
                "codec": self.codec,
            },
            "stages": [
                {
                    "rank": stage.rank,
                    "start_layer": stage.start,
                    "end_layer": stage.end,
                    "layer_count": stage.layer_count,
                    "gpu_memory_gib": stage.gpu_memory_gib,
                    "budget_gib": stage.budget_gib,
                    "estimated_gib": stage.estimated_gib,
                    "budget_fraction": stage.budget_fraction,
                }
                for stage in self.placements
            ],
        }


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def resolve_v31_config(config):
    """Resolve a user configuration into a Qwen3 N-rank layout.

    The caller supplies memory estimates. Automatic estimation from
    checkpoint tensors will be added as a separate discovery layer.

    This returns a proposed placement, not a guarantee of GPU fit.
    """

    config = _mapping(config, "config")
    model = _mapping(config.get("model"), "model")
    pipeline = _mapping(config.get("pipeline"), "pipeline")
    memory = _mapping(
        pipeline.get("memory"), "pipeline.memory"
    )
    communication = _mapping(
        config.get("communication", {}), "communication"
    )

    model_id = model.get("id", "unspecified")

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model.id must be a nonempty string")

    num_layers = model.get("num_layers")

    if type(num_layers) is not int or num_layers < 1:
        raise ValueError(
            "model.num_layers must be a positive integer"
        )

    weights = model.get("layer_weights_gib")

    if not isinstance(weights, (list, tuple)):
        raise ValueError(
            "model.layer_weights_gib must be a sequence"
        )

    if len(weights) != num_layers:
        raise ValueError(
            "layer_weights_gib length must equal num_layers"
        )

    num_ranks = pipeline.get("num_ranks")

    if type(num_ranks) is not int or num_ranks < 2:
        raise ValueError(
            "pipeline.num_ranks must be an integer >= 2"
        )

    if num_ranks > num_layers:
        raise ValueError(
            "Cannot assign more ranks than transformer layers"
        )

    partition_mode = pipeline.get("partition", "auto")

    if partition_mode not in ("auto", "manual"):
        raise ValueError(
            "pipeline.partition must be auto or manual"
        )

    counts = pipeline.get("layer_counts")

    if partition_mode == "manual":
        if counts is None:
            raise ValueError(
                "Manual partition requires layer_counts"
            )
    elif counts is not None:
        raise ValueError(
            "layer_counts cannot be set when partition=auto"
        )

    gpu_memory = memory.get("gpu_memory_gib")

    if isinstance(gpu_memory, (int, float)):
        gpu_memory = [gpu_memory] * num_ranks

    if not isinstance(gpu_memory, (list, tuple)):
        raise ValueError(
            "gpu_memory_gib must be a number or sequence"
        )

    if len(gpu_memory) != num_ranks:
        raise ValueError(
            "GPU memory entries must match num_ranks"
        )

    backend = communication.get("backend", "auto")

    if backend not in ("auto", "nccl", "gloo_staged"):
        raise ValueError(
            f"Unsupported communication backend: {backend}"
        )

    codec = communication.get("codec", "identity")

    if codec != "identity":
        raise ValueError(
            "Only identity codec is supported in Phase 1"
        )

    placements = plan_contiguous_layers(
        layer_weights_gib=weights,
        gpu_memory_gib=gpu_memory,
        gpu_reserve_gib=memory.get(
            "reserve_gib", 2.0
        ),
        kv_bytes_per_layer_per_token=memory.get(
            "kv_bytes_per_layer_per_token", 0
        ),
        max_context_tokens=memory.get(
            "max_total_tokens_per_request", 1
        ),
        max_active_requests=memory.get(
            "max_active_requests", 1
        ),
        first_stage_gib=memory.get(
            "first_stage_gib", 0.0
        ),
        last_stage_gib=memory.get(
            "last_stage_gib", 0.0
        ),
        manual_layer_counts=(
            counts if partition_mode == "manual" else None
        ),
    )

    layout = Qwen3NRankLayout.from_placements(
        num_layers, placements
    )

    if layout.world_size != num_ranks:
        raise RuntimeError(
            "Resolved layout does not match requested ranks"
        )

    return V31ResolvedPlan(
        model_id=model_id,
        num_ranks=num_ranks,
        partition_mode=partition_mode,
        layout=layout,
        placements=placements,
        transport_preference=backend,
        codec=codec,
    )
