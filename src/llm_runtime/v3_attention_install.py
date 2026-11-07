"""Install per-layer attention extensions through adapter contracts."""

from .v3_plugin_registry import load_plugin


class AttentionInstallationError(ValueError):
    pass


def install_attention_extensions(
    adapter, stage, attention_plan, model_config
):
    spec = stage.spec

    matches = [
        s for s in attention_plan["stages"]
        if s["rank"] == spec.rank
    ]
    if len(matches) != 1:
        raise AttentionInstallationError(
            "Expected one attention plan for rank"
        )

    getter = getattr(adapter, "get_native_attention", None)
    pending = []

    for segment in matches[0]["attention"]:
        if not (
            spec.start_layer <= segment["start_layer"]
            < segment["end_layer"] <= spec.end_layer
        ):
            raise AttentionInstallationError(
                "Segment outside loaded stage"
            )

        if segment["implementation"] != "custom":
            continue

        if not callable(getter):
            raise AttentionInstallationError(
                "Adapter lacks get_native_attention() "
                "for custom attention"
            )

        extension = load_plugin(
            segment["plugin"], "attention"
        )

        for idx in range(
            segment["start_layer"],
            segment["end_layer"],
        ):
            native = getter(
                stage, global_layer_idx=idx
            )
            replacement = extension.create(
                global_layer_idx=idx,
                native_module=native,
                model_config=model_config,
                options=segment.get("options", {}),
            )

            if not callable(
                getattr(replacement, "forward", None)
            ):
                raise AttentionInstallationError(
                    f"Invalid attention module at layer {idx}"
                )

            pending.append((idx, replacement))

    # Construct and validate replacements before installation.
    for idx, replacement in pending:
        adapter.replace_attention(
            stage,
            global_layer_idx=idx,
            module=replacement,
        )

    return [idx for idx, _ in pending]
