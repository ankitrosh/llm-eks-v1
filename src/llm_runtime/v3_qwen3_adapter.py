"""V3 adapter wrapping the validated two-rank Qwen3 loader.

No torch/CUDA import occurs until create_stage().
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from .v3_extension_contracts import (
    StageInputs,
    StageOutputs,
    StageSpec,
)


@dataclass
class LoadedStage:
    spec: StageSpec
    module: Any


class Qwen3Adapter:
    """Bridge V3 stage configuration to the existing V2 loader."""

    model_type = "qwen3"

    def __init__(
        self,
        num_layers: int,
        *,
        stage_factory: Callable | None = None,
    ):
        if type(num_layers) is not int or num_layers < 2:
            raise ValueError("Qwen3 requires at least two layers")

        self.num_layers = num_layers
        self._stage_factory = stage_factory
        self._planned_stages = None

    def validate_partition(self, stages):
        """Validate the complete partition before constructing ranks."""

        if len(stages) != 2:
            raise ValueError(
                "Current Qwen3 runtime supports exactly 2 ranks"
            )

        a, b = stages

        if not all(isinstance(s, StageSpec) for s in stages):
            raise TypeError("Expected StageSpec objects")

        valid = (
            a.rank == 0
            and b.rank == 1
            and a.first_stage is True
            and a.final_stage is False
            and b.first_stage is False
            and b.final_stage is True
            and a.start_layer == 0
            and 0 < a.end_layer < self.num_layers
            and b.start_layer == a.end_layer
            and b.end_layer == self.num_layers
        )

        if not valid:
            raise ValueError(
                "Invalid Qwen3 two-rank partition: "
                "expected [0, split), [split, num_layers)"
            )

        self._planned_stages = (a, b)
        return self._planned_stages

    @staticmethod
    def _default_factory():
        """Import the original GPU loader only when needed."""

        try:
            module = import_module("qwen3_stage_loader")
        except ModuleNotFoundError as exc:
            if exc.name != "qwen3_stage_loader":
                raise

            # Source-checkout import path.
            module = import_module("gpu.qwen3_stage_loader")

        return module.Qwen3Stage

    def create_stage(
        self,
        spec: StageSpec,
        *,
        checkpoint: str,
        device: str,
        dtype: str,
    ) -> LoadedStage:
        if self._planned_stages is None:
            raise RuntimeError(
                "Call validate_partition() before create_stage()"
            )

        if spec not in self._planned_stages:
            raise ValueError(
                "Stage does not belong to validated partition"
            )

        if device not in {"cuda", "cuda:0"}:
            raise ValueError(
                "V2 Qwen3Stage currently requires CUDA"
            )

        if dtype not in {"bfloat16", "bf16"}:
            raise ValueError(
                "V2 Qwen3Stage currently requires BF16"
            )

        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("checkpoint must be a local directory")

        # The V2 loader expects downloaded model files.
        if (
            self._stage_factory is None
            and not Path(checkpoint).is_dir()
        ):
            raise ValueError(
                "Checkpoint directory not found: " + checkpoint
            )

        split = self._planned_stages[0].end_layer

        factory = (
            self._stage_factory
            if self._stage_factory is not None
            else self._default_factory()
        )

        module = factory(checkpoint, spec.rank, split).load()

        if not callable(getattr(module, "forward_local", None)):
            raise TypeError("Stage missing forward_local()")

        if not callable(getattr(module, "make_cache", None)):
            raise TypeError("Stage missing make_cache()")

        # Ensure the actual loader honored the requested range.
        if (
            getattr(module, "start", None) != spec.start_layer
            or getattr(module, "stop", None) != spec.end_layer
        ):
            raise RuntimeError(
                "Loaded stage range does not match planned range"
            )

        return LoadedStage(spec=spec, module=module)

    def create_cache(self, stage: LoadedStage):
        return stage.module.make_cache()

    def forward_stage(
        self,
        stage: LoadedStage,
        inputs: StageInputs,
    ) -> StageOutputs:
        if not isinstance(stage, LoadedStage):
            raise TypeError("Expected LoadedStage")

        if not isinstance(inputs, StageInputs):
            raise TypeError("Expected StageInputs")

        position = inputs.positions

        if type(position) is not int or position < 0:
            raise ValueError(
                "V2 Qwen3 requires integer start_position >= 0"
            )

        output = stage.module.forward_local(
            inputs.payload,
            position,
            inputs.cache_state,
        )

        metadata = {
            "rank": stage.spec.rank,
            "start_layer": stage.spec.start_layer,
            "end_layer": stage.spec.end_layer,
        }

        if stage.spec.final_stage:
            return StageOutputs(
                logits=output,
                cache_state=inputs.cache_state,
                metadata=metadata,
            )

        return StageOutputs(
            hidden_states=output,
            cache_state=inputs.cache_state,
            metadata=metadata,
        )

    def replace_attention(
        self,
        stage: LoadedStage,
        *,
        global_layer_idx: int,
        module: Any,
    ) -> None:
        raise NotImplementedError(
            "Qwen3 attention replacement is not wired yet. "
            "The V2 loader supports eager/SDPA selection; "
            "per-layer replacement requires model-specific "
            "module installation and weight validation."
        )
