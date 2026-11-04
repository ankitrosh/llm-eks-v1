"""CPU-only V3 manifest validation and partition planning."""

from pathlib import Path
import re

try:
    import yaml
except ImportError as exc:
    raise RuntimeError("Install PyYAML to read manifests") from exc


class ManifestError(ValueError):
    pass


def obj(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"{name} must be a mapping")
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ManifestError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def plugin_check(setting, name):
    impl = setting.get("implementation", "model_native")

    if impl not in {"auto", "model_native", "custom"}:
        raise ManifestError(
            f"{name}.implementation invalid: {impl}"
        )

    plugin = setting.get("plugin")

    if impl == "custom":
        pattern = (
            r"[A-Za-z_]\w*"
            r"(?:\.[A-Za-z_]\w*)*"
            r":[A-Za-z_]\w*"
        )
        if not isinstance(plugin, str) or not re.fullmatch(
            pattern, plugin
        ):
            raise ManifestError(
                f"{name}.plugin needs package.module:ClassName"
            )
    elif plugin is not None:
        raise ManifestError(
            f"{name}.plugin requires implementation: custom"
        )

    backend = setting.get("backend", "auto")

    if backend not in {
        "auto", "eager", "sdpa",
        "flash_attention_2", "pytorch"
    }:
        raise ManifestError(
            f"{name}.backend invalid: {backend}"
        )


def validate(data):
    data = obj(data, "manifest")

    if data.get("schema_version", 1) != 1:
        raise ManifestError("Only schema_version: 1 supported")

    supported = {
        "schema_version", "model", "pipeline",
        "runtime", "attention", "quantization",
        "cache", "scheduler", "deployment", "validation",
    }

    unknown = set(data) - supported
    if unknown:
        raise ManifestError(
            f"Unknown top-level keys: {sorted(unknown)}"
        )

    model = obj(data.get("model"), "model")
    model_id = model.get("id", "Qwen/Qwen3-14B")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ManifestError("model.id must be nonempty")

    n = model.get("num_layers")
    if n is not None:
        integer(n, "model.num_layers", 1)

    pipeline = obj(data.get("pipeline"), "pipeline")
    ranks = integer(
        pipeline.get("num_ranks", 2),
        "pipeline.num_ranks", 1
    )

    if n is not None and ranks > n:
        raise ManifestError(
            "num_ranks cannot exceed num_layers"
        )

    policy = pipeline.get("partition", "equal_layers")

    if policy not in {"manual", "equal_layers"}:
        raise ManifestError(
            "V3.1 supports manual or equal_layers"
        )

    if policy == "equal_layers" and pipeline.get("stages"):
        raise ManifestError(
            "Explicit stages require partition: manual"
        )

    stages = []

    if policy == "manual":
        raw = pipeline.get("stages")

        if not isinstance(raw, list) or len(raw) != ranks:
            raise ManifestError(
                "manual partition needs one stage per rank"
            )

        end = 0

        for rank, item in enumerate(raw):
            item = obj(item, f"pipeline.stages[{rank}]")

            if item.get("rank") != rank:
                raise ManifestError(
                    f"Expected rank {rank}"
                )

            start = integer(
                item.get("start_layer"), "start_layer"
            )
            next_end = integer(
                item.get("end_layer"), "end_layer", 1
            )

            if start != end or next_end <= start:
                raise ManifestError(
                    "Stages must be contiguous and nonempty"
                )

            end = next_end
            stages.append({
                "rank": rank,
                "start_layer": start,
                "end_layer": end,
            })

        if n is not None and end != n:
            raise ManifestError(
                f"Partition ends at {end}; model has {n} layers"
            )

    elif n is not None:
        q, rem = divmod(n, ranks)
        start = 0

        for rank in range(ranks):
            end = start + q + (1 if rank < rem else 0)

            stages.append({
                "rank": rank,
                "start_layer": start,
                "end_layer": end,
            })

            start = end

    attention = obj(data.get("attention"), "attention")

    if "default" in attention:
        base = obj(
            attention["default"], "attention.default"
        )
    else:
        base = {
            k: v for k, v in attention.items()
            if k != "overrides"
        }

    plugin_check(base, "attention.default")

    overrides = attention.get("overrides", [])

    if not isinstance(overrides, list):
        raise ManifestError(
            "attention.overrides must be a list"
        )

    seen = set()

    for i, item in enumerate(overrides):
        item = obj(item, f"attention.overrides[{i}]")
        plugin_check(item, f"attention.overrides[{i}]")

        layers = item.get("layers")

        if not isinstance(layers, list) or not layers:
            raise ManifestError(
                "Override layers must be a nonempty list"
            )

        for layer in layers:
            integer(layer, "override layer")

            if n is not None and layer >= n:
                raise ManifestError(
                    f"Override layer {layer} outside model"
                )

            if layer in seen:
                raise ManifestError(
                    f"Attention override layer {layer} repeated"
                )

            seen.add(layer)

    runtime = obj(data.get("runtime"), "runtime")
    device = runtime.get("device", "auto")
    backend = runtime.get("distributed_backend", "auto")

    if device not in {"auto", "cpu", "cuda"}:
        raise ManifestError("Invalid runtime.device")

    if backend not in {"auto", "gloo", "nccl"}:
        raise ManifestError(
            "Invalid runtime.distributed_backend"
        )

    if device == "cpu" and backend == "nccl":
        raise ManifestError("CPU runtime cannot use NCCL")

    scheduler = obj(data.get("scheduler"), "scheduler")
    mode = scheduler.get("mode", "sequential")

    if mode not in {"sequential", "pipelined"}:
        raise ManifestError("Invalid scheduler.mode")

    integer(
        scheduler.get("max_active_requests", 1),
        "scheduler.max_active_requests", 1
    )

    notices = [
        "No weights loaded; model metadata and "
        "runtime compatibility not verified"
    ]

    if n is None:
        notices.append(
            "Model layer count unknown; inspect checkpoint "
            "metadata before deployment"
        )

    if ranks != 2:
        notices.append(
            "N-rank execution not implemented yet (V3.2)"
        )

    if device == "cpu":
        notices.append(
            "CPU/Gloo inference not implemented yet (V3.2)"
        )

    if mode == "pipelined":
        notices.append(
            "Concurrent request scheduling not implemented "
            "yet (V3.3)"
        )

    if overrides:
        notices.append(
            "Per-layer attention overrides parsed only; "
            "execution pending plugin interface"
        )

    return {
        "model_id": model_id,
        "num_layers": n,
        "num_ranks": ranks,
        "partition": policy,
        "stages": stages,
        "not_run": notices,
    }


def load_manifest(path):
    try:
        data = yaml.safe_load(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(
            f"Unable to read YAML manifest: {exc}"
        ) from exc

    return validate(data)
