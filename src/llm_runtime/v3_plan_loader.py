"""Validate and apply a compiled V3 runtime plan.

CPU-only validation. Does not import PyTorch or load weights.
"""

import hashlib
import json
import os
from pathlib import Path


class PlanLoadError(ValueError):
    pass


REQUIRED_ENV = {
    "LLM_STAGE_IMPL",
    "LLM_ATTENTION_BACKEND",
    "LLM_SDPA_PREFIX_MASK",
    "LLM_PREFIX_CACHE_ENABLED",
    "LLM_PREFIX_CACHE_MAX_MIB",
    "LLM_PREFIX_CACHE_MAX_ENTRIES",
    "LLM_PREFIX_CACHE_MIN_TOKENS",
}

OPTIONAL_ENV = {"LLM_GPTQ_BACKEND"}


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


def read_json(path):
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise PlanLoadError(
            f"Cannot read JSON: {path}"
        ) from exc

    if not isinstance(result, dict):
        raise PlanLoadError("Expected a JSON object")

    return result


def load_runtime_plan(
    plan_path,
    model_dir,
    *,
    rank,
    world_size=2,
):
    """Check plan integrity and local checkpoint configuration."""

    plan = read_json(plan_path)

    if plan.get("plan_version") != 1:
        raise PlanLoadError("Unsupported plan version")

    expected_hash = plan.get("plan_sha256")
    unsigned = {
        k: v for k, v in plan.items()
        if k != "plan_sha256"
    }

    if expected_hash != digest(unsigned):
        raise PlanLoadError("Runtime plan checksum mismatch")

    if plan.get("status") != (
        "STATICALLY_VALIDATED_NOT_EXECUTED"
    ):
        raise PlanLoadError("Unexpected runtime plan status")

    checkpoint = read_json(
        Path(model_dir) / "config.json"
    )

    model = plan.get("model")
    pipeline = plan.get("pipeline")

    if not isinstance(model, dict):
        raise PlanLoadError("Invalid plan model")

    if not isinstance(pipeline, dict):
        raise PlanLoadError("Invalid plan pipeline")

    if model.get("metadata_sha256") != digest(checkpoint):
        raise PlanLoadError(
            "Local checkpoint config differs from compiled plan"
        )

    if (
        checkpoint.get("model_type") != "qwen3"
        or checkpoint.get("tie_word_embeddings") is not False
    ):
        raise PlanLoadError(
            "Current execution requires untied Qwen3"
        )

    n = checkpoint.get("num_hidden_layers")

    if type(n) is not int or n < 2:
        raise PlanLoadError("Invalid checkpoint layer count")

    if model.get("layers") != n:
        raise PlanLoadError("Model layer count mismatch")

    if (
        type(rank) is not int
        or rank not in (0, 1)
        or world_size != 2
        or pipeline.get("num_ranks") != 2
    ):
        raise PlanLoadError(
            "Current runtime supports exactly two ranks"
        )

    split = pipeline.get("split")

    if type(split) is not int or not 0 < split < n:
        raise PlanLoadError("Invalid partition boundary")

    expected_stages = [
        {
            "rank": 0,
            "start_layer": 0,
            "end_layer": split,
        },
        {
            "rank": 1,
            "start_layer": split,
            "end_layer": n,
        },
    ]

    if pipeline.get("stages") != expected_stages:
        raise PlanLoadError(
            "Stage assignments inconsistent with split"
        )

    env = plan.get("runtime_environment")

    if not isinstance(env, dict):
        raise PlanLoadError("Invalid runtime environment")

    if not REQUIRED_ENV.issubset(env):
        raise PlanLoadError("Required runtime settings missing")

    if set(env) - REQUIRED_ENV - OPTIONAL_ENV:
        raise PlanLoadError(
            "Unknown runtime environment settings"
        )

    if any(
        not isinstance(value, str)
        for value in env.values()
    ):
        raise PlanLoadError(
            "Runtime environment values must be strings"
        )

    if env["LLM_STAGE_IMPL"] != "v3_adapter":
        raise PlanLoadError("Invalid stage implementation")

    backend = env["LLM_ATTENTION_BACKEND"]
    mask = env["LLM_SDPA_PREFIX_MASK"]

    if backend not in {"eager", "sdpa"}:
        raise PlanLoadError("Invalid attention backend")

    if mask not in {"explicit", "lower_right"}:
        raise PlanLoadError("Invalid prefix mask")

    if mask == "lower_right" and backend != "sdpa":
        raise PlanLoadError(
            "Lower-right mask requires SDPA"
        )

    if env["LLM_PREFIX_CACHE_ENABLED"] not in {"0", "1"}:
        raise PlanLoadError("Invalid prefix cache flag")

    for key in (
        "LLM_PREFIX_CACHE_MAX_MIB",
        "LLM_PREFIX_CACHE_MAX_ENTRIES",
        "LLM_PREFIX_CACHE_MIN_TOKENS",
    ):
        value = env[key]

        if not value.isdecimal() or int(value) < 1:
            raise PlanLoadError(
                f"Invalid positive integer: {key}"
            )

    checkpoint_quant = checkpoint.get(
        "quantization_config"
    )

    quant_backend = env.get("LLM_GPTQ_BACKEND")

    if checkpoint_quant is not None:
        if quant_backend not in {"marlin", "tritonv2"}:
            raise PlanLoadError(
                "Quantized checkpoint requires GPTQ backend"
            )
    elif quant_backend is not None:
        raise PlanLoadError(
            "Unquantized checkpoint cannot specify GPTQ backend"
        )

    return plan


def configure_serving_from_plan(args, env=None):
    """Apply validated plan before stage construction.

    Leaves existing legacy behavior unchanged if no plan is set.
    """

    if env is None:
        env = os.environ

    path = env.get("LLM_RUNTIME_PLAN")

    if not path:
        return None

    raw_world_size = getattr(
        args, "world_size", None
    )

    if raw_world_size is None:
        try:
            raw_world_size = int(
                env.get("WORLD_SIZE", "2")
            )
        except ValueError as exc:
            raise PlanLoadError(
                "Invalid WORLD_SIZE"
            ) from exc

    plan = load_runtime_plan(
        path,
        args.model_dir,
        rank=args.rank,
        world_size=raw_world_size,
    )

    # Apply configuration only after all validation passes.
    env.update(plan["runtime_environment"])
    args.split = plan["pipeline"]["split"]

    return plan
