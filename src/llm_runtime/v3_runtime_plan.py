"""Compile V3 manifests into validated Qwen3 runtime plans.

Static only: no model weights, CUDA, or plugin imports.
"""

from copy import deepcopy
from hashlib import sha256
import json

from .v3_manifest import ManifestError, validate
from .v3_attention_plan import resolve_attention_plan


class RuntimePlanError(ManifestError):
    pass


def mapping(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimePlanError(f"{name} must be a mapping")
    return value


def allowed(section, keys, name):
    unknown = set(section) - set(keys)
    if unknown:
        raise RuntimePlanError(
            f"Unimplemented {name} options: {sorted(unknown)}"
        )


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise RuntimePlanError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def supported_gptq(quant):
    """Mirror the existing Qwen3 loader's GPTQ constraints."""
    return (
        isinstance(quant, dict)
        and quant.get("quant_method") == "gptq"
        and quant.get("checkpoint_format") == "gptq"
        and quant.get("bits") == 8
        and quant.get("group_size") == 128
        and quant.get("sym") is True
        and quant.get("desc_act") is False
        and quant.get("pack_dtype") == "int32"
        and quant.get("lm_head") is False
        and quant.get("hyb_act") is False
        and isinstance(quant.get("meta", {}), dict)
        and quant.get("meta", {}).get("v2") is False
    )


def compile_runtime_plan(manifest, checkpoint_config):
    if not isinstance(manifest, dict):
        raise RuntimePlanError("Manifest must be a mapping")

    cfg = mapping(checkpoint_config, "checkpoint config")

    if cfg.get("model_type") != "qwen3":
        raise RuntimePlanError(
            "Executable adapter currently supports Qwen3 only"
        )

    if "Qwen3ForCausalLM" not in (
        cfg.get("architectures") or []
    ):
        raise RuntimePlanError(
            "Expected Qwen3ForCausalLM architecture"
        )

    if cfg.get("tie_word_embeddings") is not False:
        raise RuntimePlanError(
            "Current loader requires untied embeddings"
        )

    n = positive_int(
        cfg.get("num_hidden_layers"),
        "num_hidden_layers", 2
    )

    conf = deepcopy(manifest)
    model = mapping(conf.get("model"), "model")

    allowed(
        model,
        {
            "id", "revision", "adapter",
            "architecture", "num_layers", "dtype",
        },
        "model",
    )

    if (
        model.get("num_layers") is not None
        and model["num_layers"] != n
    ):
        raise RuntimePlanError(
            "Manifest layer count differs from checkpoint"
        )

    if model.get("adapter", "auto") not in ("auto", "qwen3"):
        raise RuntimePlanError("Unsupported model adapter")

    if model.get("architecture", "auto") not in (
        "auto", "qwen3"
    ):
        raise RuntimePlanError("Unsupported architecture")

    if model.get("dtype", "bfloat16") not in (
        "bfloat16", "bf16"
    ):
        raise RuntimePlanError("Qwen3 loader requires BF16")

    model["num_layers"] = n
    conf["model"] = model

    pipeline = mapping(conf.get("pipeline"), "pipeline")

    allowed(
        pipeline,
        {"num_ranks", "partition", "stages"},
        "pipeline",
    )

    checked = validate(conf)

    if checked["num_ranks"] != 2:
        raise RuntimePlanError(
            "N-rank execution is not available until V3.2"
        )

    if len(checked["stages"]) != 2:
        raise RuntimePlanError("Invalid two-stage partition")

    attention = mapping(conf.get("attention"), "attention")

    if attention.get("overrides"):
        raise RuntimePlanError(
            "Per-layer attention overrides are not "
            "connected to execution yet"
        )

    attention_plan = resolve_attention_plan(conf)

    settings = {
        (segment["implementation"], segment["backend"])
        for stage in attention_plan["stages"]
        for segment in stage["attention"]
    }

    if len(settings) != 1:
        raise RuntimePlanError(
            "Current loader requires one attention backend"
        )

    for stage in attention_plan["stages"]:
        for segment in stage["attention"]:
            if segment["options"]:
                raise RuntimePlanError(
                    "Attention options are not wired yet"
                )

    implementation, att_backend = settings.pop()

    if implementation not in ("auto", "model_native"):
        raise RuntimePlanError(
            "Custom attention execution not implemented"
        )

    runtime = mapping(conf.get("runtime"), "runtime")

    allowed(
        runtime,
        {
            "device", "distributed_backend",
            "attention_backend", "sdpa_prefix_mask",
        },
        "runtime",
    )

    if runtime.get("device", "cuda") not in (
        "auto", "cuda"
    ):
        raise RuntimePlanError(
            "Current Qwen3 loader requires CUDA"
        )

    if runtime.get("distributed_backend", "nccl") not in (
        "auto", "nccl"
    ):
        raise RuntimePlanError(
            "Current serving engine requires NCCL"
        )

    alternate = runtime.get("attention_backend", "auto")

    if (
        att_backend not in ("auto", "eager", "sdpa")
        or alternate not in ("auto", "eager", "sdpa")
    ):
        raise RuntimePlanError(
            "Only eager and SDPA are executable"
        )

    if (
        att_backend != "auto"
        and alternate != "auto"
        and att_backend != alternate
    ):
        raise RuntimePlanError(
            "Conflicting attention backend settings"
        )

    backend = (
        att_backend if att_backend != "auto"
        else alternate if alternate != "auto"
        else "sdpa"
    )

    mask = runtime.get("sdpa_prefix_mask", "explicit")

    if mask not in ("explicit", "lower_right"):
        raise RuntimePlanError("Unsupported prefix mask")

    if mask == "lower_right" and backend != "sdpa":
        raise RuntimePlanError(
            "Lower-right mask requires SDPA"
        )

    quant = mapping(
        conf.get("quantization"), "quantization"
    )

    allowed(quant, {"backend"}, "quantization")

    requested = quant.get("backend", "auto")
    checkpoint_quant = cfg.get("quantization_config")

    if checkpoint_quant is not None:
        if not supported_gptq(checkpoint_quant):
            raise RuntimePlanError(
                "Checkpoint GPTQ format unsupported"
            )

        if requested not in (
            "auto", "marlin", "tritonv2"
        ):
            raise RuntimePlanError(
                "GPTQ requires Marlin or TritonV2"
            )

        quant_backend = (
            "marlin" if requested == "auto"
            else requested
        )
    else:
        if requested not in ("auto", "none"):
            raise RuntimePlanError(
                "Unquantized checkpoint cannot use GPTQ backend"
            )

        quant_backend = "none"

    cache = mapping(conf.get("cache"), "cache")

    allowed(
        cache,
        {
            "prefix_enabled", "max_mib_per_rank",
            "max_entries", "min_tokens", "implementation",
        },
        "cache",
    )

    if cache.get("implementation", "auto") not in (
        "auto", "model_native"
    ):
        raise RuntimePlanError(
            "Custom cache implementation is not wired yet"
        )

    enabled = cache.get("prefix_enabled", False)

    if type(enabled) is not bool:
        raise RuntimePlanError(
            "prefix_enabled must be boolean"
        )

    max_mib = positive_int(
        cache.get("max_mib_per_rank", 256),
        "max_mib_per_rank"
    )

    entries = positive_int(
        cache.get("max_entries", 4), "max_entries"
    )

    min_tokens = positive_int(
        cache.get("min_tokens", 64), "min_tokens"
    )

    scheduler = mapping(
        conf.get("scheduler"), "scheduler"
    )

    allowed(
        scheduler,
        {"mode", "max_active_requests", "batching"},
        "scheduler",
    )

    batching = mapping(
        scheduler.get("batching"), "scheduler.batching"
    )

    allowed(batching, {"enabled"}, "scheduler.batching")

    if (
        scheduler.get("mode", "sequential") != "sequential"
        or scheduler.get("max_active_requests", 1) != 1
        or batching.get("enabled", False) is not False
    ):
        raise RuntimePlanError(
            "Concurrent scheduling/batching not yet executable"
        )

    deployment = mapping(
        conf.get("deployment"), "deployment"
    )

    allowed(
        deployment,
        {"platform", "gpu_per_rank"},
        "deployment",
    )

    if deployment.get("platform", "eks") not in (
        "eks", "docker", "local"
    ):
        raise RuntimePlanError(
            "Unsupported deployment platform"
        )

    if (
        type(deployment.get("gpu_per_rank", 1)) is not int
        or deployment.get("gpu_per_rank", 1) != 1
    ):
        raise RuntimePlanError(
            "Current runtime requires one GPU per rank"
        )

    validation_options = mapping(
        conf.get("validation"), "validation"
    )

    allowed(
        validation_options,
        {"enabled"},
        "validation",
    )

    if validation_options.get("enabled", True) is not True:
        raise RuntimePlanError(
            "Runtime plan validation cannot be disabled"
        )

    env = {
        "LLM_STAGE_IMPL": "v3_adapter",
        "LLM_ATTENTION_BACKEND": backend,
        "LLM_SDPA_PREFIX_MASK": mask,
        "LLM_PREFIX_CACHE_ENABLED": "1" if enabled else "0",
        "LLM_PREFIX_CACHE_MAX_MIB": str(max_mib),
        "LLM_PREFIX_CACHE_MAX_ENTRIES": str(entries),
        "LLM_PREFIX_CACHE_MIN_TOKENS": str(min_tokens),
    }

    if quant_backend != "none":
        env["LLM_GPTQ_BACKEND"] = quant_backend

    plan = {
        "plan_version": 1,
        "status": "STATICALLY_VALIDATED_NOT_EXECUTED",
        "model": {
            "id": checked["model_id"],
            "revision": model.get("revision", "main"),
            "type": "qwen3",
            "layers": n,
            "dtype": "bfloat16",
            "metadata_sha256": sha256(
                json.dumps(cfg, sort_keys=True).encode()
            ).hexdigest(),
        },
        "pipeline": {
            "num_ranks": 2,
            "partition": checked["partition"],
            "stages": checked["stages"],
            "split": checked["stages"][0]["end_layer"],
        },
        "runtime_environment": env,
        "not_run": [
            "Model weights and tokenizer not loaded",
            "GPU memory feasibility not tested",
            "NCCL inference correctness not tested",
            "Runtime plan not connected to serving launcher yet",
        ],
    }

    plan["plan_sha256"] = sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()

    return plan
