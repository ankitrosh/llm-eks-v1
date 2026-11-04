"""V3 extension contracts, without PyTorch dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class StageSpec:
    rank: int
    start_layer: int
    end_layer: int
    first_stage: bool
    final_stage: bool


@dataclass
class StageInputs:
    payload: Any
    positions: Any = None
    attention_mask: Any = None
    cache_state: Any = None
    request_id: str | None = None


@dataclass
class StageOutputs:
    hidden_states: Any = None
    logits: Any = None
    cache_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    """Model-specific loading and stage execution."""

    def create_stage(
        self,
        spec: StageSpec,
        *,
        checkpoint: str,
        device: str,
        dtype: str,
    ) -> Any:
        ...

    def forward_stage(
        self,
        stage: Any,
        inputs: StageInputs,
    ) -> StageOutputs:
        ...

    def replace_attention(
        self,
        stage: Any,
        *,
        global_layer_idx: int,
        module: Any,
    ) -> None:
        ...


class AttentionExtension(Protocol):
    """Construct a replacement attention module.

    The model adapter is responsible for checking the
    module call signature and weight compatibility.
    """

    def create(
        self,
        *,
        global_layer_idx: int,
        native_module: Any,
        model_config: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> Any:
        ...


class CacheExtension(Protocol):
    """Cache representation and snapshot operations."""

    def create(
        self,
        *,
        num_layers: int,
        options: Mapping[str, Any],
    ) -> Any:
        ...

    def snapshot(self, cache_state: Any) -> Any:
        ...

    def restore(self, snapshot: Any) -> Any:
        ...


class OptimizationExtension(Protocol):
    """Optional optimization, including quantization."""

    def check_compatibility(
        self,
        *,
        model_metadata: Mapping[str, Any],
        device: str,
        dtype: str,
    ) -> list[str]:
        ...

    def apply(
        self,
        stage: Any,
        *,
        options: Mapping[str, Any],
    ) -> Any:
        ...
