"""Compile V3.1 model placement and communication into one launch spec.

CPU-only. Uses existing checkpoint discovery, partition planner,
and configuration resolver. Does not launch workers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from llm_runtime.v3_1_runtime_config import (
    V31ResolvedPlan,
    resolve_v31_config,
)
from llm_runtime.v3_1_checkpoint_profile import (
    resolve_v31_checkpoint_config,
)


FORMAT_VERSION = 1


@dataclass(frozen=True)
class V31LaunchSpec:
    """Validated rank layout and communication configuration."""

    plan: V31ResolvedPlan
    plan_sha256: str
    checkpoint_profile: object | None = None

    @property
    def layout(self):
        """The actual Qwen3NRankLayout used by stage factories."""
        return self.plan.layout

    @property
    def world_size(self):
        return self.plan.num_ranks

    def for_rank(self, rank):
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError("Invalid rank")

        placement = self.plan.placements[rank]

        return {
            "rank": rank,
            "world_size": self.world_size,
            "start_layer": placement.start,
            "end_layer": placement.end,
            "layer_count": placement.layer_count,
            "gpu_memory_gib": placement.gpu_memory_gib,
            "budget_gib": placement.budget_gib,
            "estimated_gib": placement.estimated_gib,
            "communication": dict(
                self.plan.as_dict()["communication"]
            ),
            "plan_sha256": self.plan_sha256,
        }

    def as_dict(self):
        result = {
            "format_version": FORMAT_VERSION,
            "model_id": self.plan.model_id,
            "world_size": self.world_size,
            "partition_mode": self.plan.partition_mode,
            "layer_boundaries": list(
                self.layout.boundaries
            ),
            "communication": dict(
                self.plan.as_dict()["communication"]
            ),
            "ranks": [
                self.for_rank(rank)
                for rank in range(self.world_size)
            ],
            "plan_sha256": self.plan_sha256,
        }

        if self.checkpoint_profile is not None:
            result["checkpoint_profile"] = (
                self.checkpoint_profile.as_dict()
            )

        return result


def compile_v31_launch_spec(
    config,
    *,
    gpu_memory_gib=None,
):
    """Resolve a single user configuration into all rank assignments.

    gpu_memory_gib is an optional offline planning override for
    checkpoint mode. Deployment-time GPU discovery comes later.
    """

    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be an object")

    model = config.get("model")

    if not isinstance(model, Mapping):
        raise ValueError("model must be an object")

    if "checkpoint_dir" in model:
        plan, profile = resolve_v31_checkpoint_config(
            config,
            gpu_memory_gib=gpu_memory_gib,
        )
    else:
        if gpu_memory_gib is not None:
            raise ValueError(
                "GPU memory override requires checkpoint mode"
            )

        plan = resolve_v31_config(config)
        profile = None

    boundaries = tuple(plan.layout.boundaries)

    if (
        len(boundaries) != plan.num_ranks + 1
        or boundaries[0] != 0
        or any(
            b <= a
            for a, b in zip(boundaries, boundaries[1:])
        )
    ):
        raise ValueError("Invalid resolved layer boundaries")

    for placement in plan.placements:
        if (
            placement.start != boundaries[placement.rank]
            or placement.end != boundaries[placement.rank + 1]
        ):
            raise ValueError(
                "Planner and layout assignments disagree"
            )

    # This identifies the resolved launch plan.
    # It is NOT a checksum of checkpoint weight contents.
    identity = {
        "format_version": FORMAT_VERSION,
        "plan": plan.as_dict(),
    }

    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    digest = hashlib.sha256(canonical).hexdigest()

    return V31LaunchSpec(
        plan=plan,
        plan_sha256=digest,
        checkpoint_profile=profile,
    )
