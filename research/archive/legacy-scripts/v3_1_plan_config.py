"""Resolve V3.1 placement from JSON/YAML configuration."""

import argparse
import json
from pathlib import Path

from llm_runtime.v3_1_runtime_config import (
    resolve_v31_config,
)
from llm_runtime.v3_1_checkpoint_profile import (
    resolve_v31_checkpoint_config,
)


def read_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        return json.loads(text)

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Install PyYAML to read YAML configuration"
            ) from exc

        return yaml.safe_load(text)

    raise ValueError("Expected JSON or YAML configuration")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument(
        "--gpu-memory-gib",
        nargs="+",
        type=float,
        help=(
            "GPU memory per rank, or one value shared by "
            "all ranks. Used for offline checkpoint planning."
        ),
    )

    args = parser.parse_args()

    config = read_config(args.config)

    if "checkpoint_dir" in config.get("model", {}):
        gpu_memory = args.gpu_memory_gib

        if gpu_memory is not None and len(gpu_memory) == 1:
            gpu_memory = gpu_memory[0]

        plan, profile = resolve_v31_checkpoint_config(
            config,
            gpu_memory_gib=gpu_memory,
        )

        result = plan.as_dict()
        result["checkpoint_profile"] = profile.as_dict()

    else:
        if args.gpu_memory_gib is not None:
            raise ValueError(
                "--gpu-memory-gib currently requires checkpoint mode"
            )

        result = resolve_v31_config(config).as_dict()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
