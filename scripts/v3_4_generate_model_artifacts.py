"""CPU-only checkpoint metadata generator with external inspectors."""

import argparse
import importlib
import json
from pathlib import Path

from llm_runtime.inspection.registry import default_registry
from llm_runtime.inspection.model_layout import (
    load_json,
    sha256_file,
)


def load_inspectors(registry, modules):
    if len(modules) != len(set(modules)):
        raise ValueError("Duplicate inspector module")

    for name in modules:
        if not name or not name.strip():
            raise ValueError("Invalid inspector module")

        module = importlib.import_module(name)
        register = getattr(module, "register_inspectors", None)

        if not callable(register):
            raise ValueError(
                f"{name} must define register_inspectors(registry)"
            )

        register(registry)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--revision", default="unspecified")
    parser.add_argument(
        "--inspector-module",
        action="append",
        default=[],
        help="Trusted Python module defining register_inspectors(registry)",
    )

    args = parser.parse_args()

    config = load_json(args.config)

    index_path = (
        args.index
        or args.config.parent / "model.safetensors.index.json"
    )

    if args.index and not args.index.is_file():
        parser.error(f"Index not found: {args.index}")

    index = load_json(index_path) if index_path.exists() else None

    registry = default_registry()
    load_inspectors(registry, args.inspector_module)

    layout, capabilities = registry.inspect(
        config, args.model_id, index
    )

    provenance = {
        "revision": args.revision,
        "config_sha256": sha256_file(args.config),
        "index_sha256": (
            sha256_file(index_path) if index is not None else None
        ),
    }

    layout["source"] = provenance
    capabilities["source"] = provenance

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in (
        ("model_layout.json", layout),
        ("model_capabilities.json", capabilities),
    ):
        destination = args.output_dir / name
        temporary = destination.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(destination)

        print(f"WROTE {destination}")

    print(
        f"MODEL_LAYOUT_OK layers={layout['num_layers']} "
        f"verification={layout['verification']}"
    )


if __name__ == "__main__":
    main()
