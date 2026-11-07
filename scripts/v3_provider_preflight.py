#!/usr/bin/env python3
"""Lightweight V3 BYOM configuration preflight."""

import argparse
import inspect
import json
from pathlib import Path


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(
            f"{name} must be a positive integer"
        )


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object")

    provider = data.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("provider must be an object")

    path = provider.get("factory_path")

    if (
        not isinstance(path, str)
        or path.count(":") != 1
        or not all(path.split(":"))
    ):
        raise ValueError(
            "provider.factory_path must be module:object"
        )

    kwargs = provider.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError(
            "provider.kwargs must be an object"
        )

    pipeline = data.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be an object")

    world_size = pipeline.get("world_size", 2)

    if type(world_size) is not int or world_size != 2:
        raise ValueError(
            "Current NCCL launcher requires exactly two ranks"
        )

    positive_int(
        pipeline.get("max_active_requests", 4),
        "pipeline.max_active_requests",
    )

    positive_int(
        pipeline.get("max_pending", 4),
        "pipeline.max_pending",
    )

    generation = data.get("generation", {})

    if not isinstance(generation, dict):
        raise ValueError(
            "generation must be an object"
        )

    if "prompts" in generation:
        prompts = generation["prompts"]

        if (
            not isinstance(prompts, list)
            or not prompts
            or not all(
                isinstance(p, str) and p.strip()
                for p in prompts
            )
        ):
            raise ValueError(
                "generation.prompts must contain "
                "nonempty text strings"
            )

    for key in (
        "max_new_tokens",
        "warmup_new_tokens",
    ):
        if key in generation:
            positive_int(
                generation[key],
                f"generation.{key}",
            )

    return path, kwargs


def preflight(
    config_path,
    check_import=False,
    check_checkpoint=False,
):
    data = json.loads(
        Path(config_path).read_text()
    )

    factory_path, kwargs = validate(data)

    print("PASS: Configuration valid")
    print(f"Provider: {factory_path}")

    if check_checkpoint:
        checkpoint = kwargs.get("checkpoint_dir")

        if (
            not isinstance(checkpoint, str)
            or not Path(checkpoint).is_dir()
        ):
            raise ValueError(
                "Checkpoint directory is not accessible"
            )

        print("PASS: Checkpoint directory exists")

    if check_import:
        from llm_runtime.v3_execution_factory import (
            resolve_factory,
        )

        try:
            target = resolve_factory(factory_path)
            inspect.signature(target).bind(**kwargs)
        except Exception as exc:
            raise ValueError(
                f"Provider import/signature failed: {exc}"
            ) from exc

        print("PASS: Provider import and constructor signature")
    else:
        print(
            "SKIP: Provider import; use --check-import "
            "in the model runtime environment"
        )

    print(
        "NOTE: This check does not instantiate stages, "
        "load weights, or test GPU execution."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--check-import",
        action="store_true",
    )

    parser.add_argument(
        "--check-checkpoint",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        preflight(
            args.config,
            check_import=args.check_import,
            check_checkpoint=args.check_checkpoint,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
