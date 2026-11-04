"""Explicitly loaded V3 extensions. No implicit imports."""

import importlib
import inspect
import re


class PluginError(ValueError):
    pass


PLUGIN_PATTERN = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r":[A-Za-z_]\w*"
)

REQUIRED_METHODS = {
    "attention": ("create",),
    "cache": ("create", "snapshot", "restore"),
    "optimization": ("check_compatibility", "apply"),
    "model": (
        "create_stage",
        "forward_stage",
        "replace_attention",
    ),
}


def validate_reference(reference):
    """Check syntax only. Never import the plugin."""

    if (
        not isinstance(reference, str)
        or not PLUGIN_PATTERN.fullmatch(reference)
    ):
        raise PluginError(
            "Plugin reference must be package.module:ClassName"
        )

    return reference


def collect_attention_plugins(attention_plan):
    """Find custom attention declarations in a static plan."""

    found = []

    for stage in attention_plan["stages"]:
        for segment in stage["attention"]:
            if segment["implementation"] != "custom":
                continue

            reference = validate_reference(segment["plugin"])

            found.append({
                "rank": stage["rank"],
                "start_layer": segment["start_layer"],
                "end_layer": segment["end_layer"],
                "reference": reference,
                "status": "DECLARED_NOT_LOADED",
            })

    return found


def load_plugin(reference, kind):
    """Load trusted user code only when explicitly requested."""

    validate_reference(reference)

    if kind not in REQUIRED_METHODS:
        raise PluginError(f"Unknown plugin kind: {kind}")

    module_name, class_name = reference.split(":", 1)

    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise PluginError(
            f"Cannot load plugin {reference}: {exc}"
        ) from exc

    if not inspect.isclass(cls):
        raise PluginError(
            f"{reference} must refer to a class"
        )

    try:
        instance = cls()
    except Exception as exc:
        raise PluginError(
            f"Cannot initialize {reference}: {exc}"
        ) from exc

    for method in REQUIRED_METHODS[kind]:
        if not callable(getattr(instance, method, None)):
            raise PluginError(
                f"{reference} missing method: {method}"
            )

    if kind == "attention":
        try:
            inspect.signature(instance.create).bind(
                global_layer_idx=0,
                native_module=None,
                model_config={},
                options={},
            )
        except (TypeError, ValueError) as exc:
            raise PluginError(
                "Attention.create does not accept "
                "the required keyword arguments"
            ) from exc

    return instance


def inspect_loaded_plugin(reference, kind):
    """Check Python interface, not tensor correctness."""

    load_plugin(reference, kind)

    return {
        "reference": reference,
        "kind": kind,
        "status": "INTERFACE_CHECK_PASSED",
        "methods": list(REQUIRED_METHODS[kind]),
        "not_run": [
            "Plugin forward execution",
            "Tensor shape and numerical correctness",
            "Model integration",
            "Cache compatibility",
            "Distributed execution",
        ],
    }
