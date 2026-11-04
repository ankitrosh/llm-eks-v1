"""Resolve attention overrides without importing plugins."""

from copy import deepcopy

from .v3_manifest import ManifestError, validate


def _attention_setting(
    raw,
    *,
    fallback=None,
    location="attention",
):
    result = deepcopy(fallback) if fallback is not None else {}

    result.update({
        key: value
        for key, value in raw.items()
        if key != "layers"
    })

    # Avoid inheriting a custom plugin when switching back
    # to model-native attention.
    if (
        fallback
        and "implementation" in raw
        and raw["implementation"]
            != fallback.get("implementation", "model_native")
        and "plugin" not in raw
    ):
        result.pop("plugin", None)

    if result.get("implementation", "model_native") != "custom":
        result.pop("plugin", None)

    options = result.get("options", {})

    if not isinstance(options, dict):
        raise ManifestError(
            f"{location}.options must be a mapping"
        )

    return {
        "implementation": result.get(
            "implementation", "model_native"
        ),
        "backend": result.get("backend", "auto"),
        "plugin": result.get("plugin"),
        "options": deepcopy(options),
    }


def resolve_attention_plan(manifest):
    """Build a static per-rank, per-layer attention plan."""

    if manifest is None:
        manifest = {}

    checked = validate(manifest)

    n = checked["num_layers"]

    if n is None or not checked["stages"]:
        raise ManifestError(
            "Cannot resolve attention without model layer "
            "count and partition; inspect config.json first"
        )

    attention = manifest.get("attention") or {}

    if "default" in attention:
        extras = set(attention) - {"default", "overrides"}

        if extras:
            raise ManifestError(
                "Use attention.default instead of sibling "
                f"keys: {sorted(extras)}"
            )

        default = attention.get("default") or {}
    else:
        default = {
            key: value
            for key, value in attention.items()
            if key != "overrides"
        }

    base = _attention_setting(
        default, location="attention.default"
    )

    layers = [deepcopy(base) for _ in range(n)]

    for i, override in enumerate(
        attention.get("overrides", [])
    ):
        selected = _attention_setting(
            override,
            fallback=base,
            location=f"attention.overrides[{i}]",
        )

        for layer in override["layers"]:
            layers[layer] = deepcopy(selected)

    stages = []

    for stage in checked["stages"]:
        segments = []

        start = stage["start_layer"]
        end = stage["end_layer"]

        for layer in range(start, end):
            current = layers[layer]

            same_as_previous = (
                bool(segments)
                and all(
                    segments[-1][key] == current[key]
                    for key in (
                        "implementation",
                        "backend",
                        "plugin",
                        "options",
                    )
                )
            )

            if same_as_previous:
                segments[-1]["end_layer"] = layer + 1
            else:
                segments.append({
                    "start_layer": layer,
                    "end_layer": layer + 1,
                    **deepcopy(current),
                    "execution": (
                        "DECLARED_NOT_IMPORTED"
                        if current["implementation"] == "custom"
                        else "NOT_CONNECTED_TO_RUNTIME"
                    ),
                })

        stages.append({
            **stage,
            "attention": segments,
        })

    return {
        "model_id": checked["model_id"],
        "status": "STATIC_PLAN_ONLY",
        "layer_count_source": "UNVERIFIED_MANIFEST_HINT",
        "stages": stages,
        "not_run": [
            "No custom Python plugin was imported or executed",
            "No attention implementation was installed",
            "No model weights or GPU hardware were used",
        ],
    }
