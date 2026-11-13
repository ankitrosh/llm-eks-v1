"""Load validated model artifacts into plugin contexts."""

import json
from pathlib import Path
from types import MappingProxyType

from .contracts import PluginContext


class ArtifactValidationError(ValueError):
    """Model metadata artifacts are inconsistent or invalid."""


def _read_object(path):
    result = json.loads(Path(path).read_text())

    if not isinstance(result, dict):
        raise ArtifactValidationError(
            f"Expected JSON object in {path}"
        )

    return result


def _freeze(value):
    """Make nested metadata read-only for plugin factories."""
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze(item)
            for key, item in value.items()
        })

    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)

    return value


def load_plugin_context(
    *,
    model_capabilities_path,
    model_layout_path,
    backend,
    runtime=None,
    require_verified=True,
):
    """Validate model artifacts and construct PluginContext.

    runtime describes requested deployment parameters.
    It is not a claim about actual GPU/runtime capabilities.
    """

    model = _read_object(model_capabilities_path)
    layout = _read_object(model_layout_path)

    for artifact, kind in (
        (model, "model_capabilities"),
        (layout, "logical_checkpoint_layout"),
    ):
        if artifact.get("schema_version") != 1:
            raise ArtifactValidationError(
                "Unsupported artifact schema version"
            )

        if artifact.get("kind") != kind:
            raise ArtifactValidationError(
                f"Unexpected artifact kind: {artifact.get('kind')}"
            )

    for key in ("model_id", "architecture"):
        if not model.get(key) or model[key] != layout.get(key):
            raise ArtifactValidationError(
                f"Model/layout {key} mismatch"
            )

    if model.get("num_layers") != layout.get("num_layers"):
        raise ArtifactValidationError(
            "Layer-count mismatch"
        )

    count = model["num_layers"]
    layers = layout.get("layers")

    if type(count) is not int or count < 1:
        raise ArtifactValidationError(
            "Invalid layer count"
        )

    if not isinstance(layers, list) or len(layers) != count:
        raise ArtifactValidationError(
            "Incomplete layer layout"
        )

    container = layout.get("transformer_container")

    if container is not None and (
        not isinstance(container, str) or not container
    ):
        raise ArtifactValidationError(
            "Invalid transformer container"
        )

    paths = set()

    for index, layer in enumerate(layers):
        path = (
            layer.get("module_path")
            if isinstance(layer, dict)
            else None
        )

        if (
            not isinstance(layer, dict)
            or type(layer.get("index")) is not int
            or layer.get("index") != index
            or not isinstance(path, str)
            or not path
            or path in paths
            or (
                container is not None
                and path != f"{container}.{index}"
            )
        ):
            raise ArtifactValidationError(
                f"Invalid layer entry at index {index}"
            )

        paths.add(path)

    model_source = model.get("source", {})
    layout_source = layout.get("source", {})

    if (
        not isinstance(model_source, dict)
        or not isinstance(layout_source, dict)
    ):
        raise ArtifactValidationError(
            "Invalid artifact provenance"
        )

    for key in ("revision", "config_sha256"):
        if (
            not model_source.get(key)
            or model_source.get(key)
            != layout_source.get(key)
        ):
            raise ArtifactValidationError(
                f"Artifact provenance mismatch: {key}"
            )

    if (
        model_source.get("index_sha256")
        != layout_source.get("index_sha256")
    ):
        raise ArtifactValidationError(
            "Checkpoint index provenance mismatch"
        )

    if require_verified and not model_source.get(
        "index_sha256"
    ):
        raise ArtifactValidationError(
            "Verified artifacts require a checkpoint index"
        )

    if require_verified:
        if layout.get("verification") != (
            "checkpoint_weight_index"
        ):
            raise ArtifactValidationError(
                "Model layout has not been verified "
                "against a checkpoint index"
            )

        if not isinstance(layout.get("checkpoint"), dict):
            raise ArtifactValidationError(
                "Verified layout lacks checkpoint metadata"
            )

    if not isinstance(backend, str) or not backend:
        raise ArtifactValidationError(
            "A backend name is required"
        )

    if runtime is None:
        runtime = {}

    if not isinstance(runtime, dict):
        raise ArtifactValidationError(
            "Runtime configuration must be a mapping"
        )

    deployment = runtime.get("deployment", {})

    if not isinstance(deployment, dict):
        raise ArtifactValidationError(
            "Invalid deployment configuration"
        )

    max_len = deployment.get("max_model_len")
    supported_len = model.get("max_position_embeddings")

    if max_len is not None:
        if type(max_len) is not int or max_len < 1:
            raise ArtifactValidationError(
                "Invalid max_model_len"
            )

        if (
            supported_len is not None
            and max_len > supported_len
        ):
            raise ArtifactValidationError(
                "Requested context exceeds the "
                "model's configured context length"
            )

    return PluginContext(
        model=_freeze(model),
        layout=_freeze(layout),
        runtime=_freeze(runtime),
        backend=backend,
    )
