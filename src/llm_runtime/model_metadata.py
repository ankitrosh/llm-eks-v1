"""Inspect common decoder-only model configurations without loading weights.

Successful inspection does not guarantee partitionability,
quantization support, optimization compatibility, or GPU fit.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _positive_int(config: Any, name: str, default: Any = None) -> int:
    value = _field(config, name, default)

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Missing or invalid {name}: expected a positive integer"
        )

    return value


def _dtype_name(value: Any) -> str | None:
    return None if value is None else str(value).removeprefix("torch.")


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_type: str
    architectures: tuple[str, ...]

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    dtype: str | None
    checkpoint_quantization: str | None

    task: str = "causal_lm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_transformers_config(
    model_id: str,
    config: Any,
) -> ModelMetadata:
    """Inspect an HF-style config dictionary or object."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")

    model_type = _field(config, "model_type")

    if not isinstance(model_type, str) or not model_type:
        raise ValueError(
            "Missing model_type; a model adapter may be required"
        )

    layers = _positive_int(config, "num_hidden_layers")
    hidden = _positive_int(config, "hidden_size")
    heads = _positive_int(config, "num_attention_heads")

    kv_heads = _positive_int(
        config, "num_key_value_heads", heads
    )

    explicit_head_dim = _field(config, "head_dim")

    if explicit_head_dim is None:
        if hidden % heads:
            raise ValueError(
                "Cannot infer head_dim: hidden_size not divisible by heads"
            )
        head_dim = hidden // heads
    else:
        head_dim = _positive_int(config, "head_dim")

    architectures = _field(config, "architectures") or ()

    if not isinstance(architectures, (tuple, list)) or not all(
        isinstance(architecture, str)
        for architecture in architectures
    ):
        raise ValueError(
            "architectures must be a list of strings"
        )

    quant_config = _field(config, "quantization_config")
    quant_method = None

    if quant_config is not None:
        quant_method = _field(quant_config, "quant_method")

        if quant_method is None:
            quant_method = "present_format_unidentified"
        else:
            quant_method = str(
                getattr(quant_method, "value", quant_method)
            )

    dtype = (
        _field(config, "dtype")
        or _field(config, "torch_dtype")
    )

    return ModelMetadata(
        model_id=model_id,
        model_type=model_type,
        architectures=tuple(architectures),
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        dtype=_dtype_name(dtype),
        checkpoint_quantization=quant_method,
    )
