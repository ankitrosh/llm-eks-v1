"""CPU-only checkpoint metadata generator."""

import argparse
import json
from pathlib import Path

from llm_runtime.inspection.registry import generate_layout
from llm_runtime.inspection.model_layout import (
    load_json,
    sha256_file,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=Path, required=True
    )
    parser.add_argument(
        "--model-id", required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    parser.add_argument(
        "--index", type=Path
    )
    parser.add_argument(
        "--revision", default="unspecified"
    )

    args = parser.parse_args()

    config = load_json(args.config)

    index_path = (
        args.index
        or args.config.parent / "model.safetensors.index.json"
    )

    if args.index and not args.index.is_file():
        parser.error(f"Index not found: {args.index}")

    index = (
        load_json(index_path)
        if index_path.exists()
        else None
    )

    layout, capabilities = generate_layout(
        config, args.model_id, index
    )

    provenance = {
        "revision": args.revision,
        "config_sha256": sha256_file(args.config),
        "index_sha256": (
            sha256_file(index_path)
            if index is not None
            else None
        ),
    }

    layout["source"] = provenance
    capabilities["source"] = provenance

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    for name, data in (
        ("model_layout.json", layout),
        ("model_capabilities.json", capabilities),
    ):
        destination = args.output_dir / name
        temporary = destination.with_suffix(
            ".json.tmp"
        )

        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True)
            + "\n"
        )
        temporary.replace(destination)

        print(f"WROTE {destination}")

    print(
        f"MODEL_LAYOUT_OK layers={layout['num_layers']} "
        f"verification={layout['verification']}"
    )


if __name__ == "__main__":
    main()
