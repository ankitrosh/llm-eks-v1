"""Registry of executable model adapters."""

from importlib import import_module


class UnsupportedAdapterError(ValueError):
    pass


ADAPTERS = {
    "qwen3": {
        "class": (
            "llm_runtime.v3_qwen3_adapter:Qwen3Adapter"
        ),
        "execution_status": "V2_TWO_RANK_GPU_WRAPPER",
        "supported_devices": ["cuda"],
        "supported_dtypes": ["bfloat16"],
        "supported_rank_counts": [2],
    },
}


def list_adapters():
    """Return capability declarations without loading adapters."""

    return {
        name: {
            key: value
            for key, value in entry.items()
            if key != "class"
        }
        for name, entry in ADAPTERS.items()
    }


def create_adapter(model_type, *, num_layers, **kwargs):
    entry = ADAPTERS.get(model_type)

    if entry is None:
        raise UnsupportedAdapterError(
            f"No executable adapter for {model_type!r}. "
            "Implement a model adapter or choose a "
            "registered architecture."
        )

    reference = entry["class"]
    module_name, class_name = reference.split(":", 1)

    cls = getattr(import_module(module_name), class_name)

    return cls(num_layers=num_layers, **kwargs)
