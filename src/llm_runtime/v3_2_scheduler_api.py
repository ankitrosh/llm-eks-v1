"""V3.2 public scheduler configuration and construction API."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SchedulerSettings:
    mode: str
    policy: str
    max_active_requests: int
    max_requests_per_batch: int | None
    max_batch_tokens: int | None
    max_queue_requests: int | None
    max_wait_ms: int | None

    def as_dict(self):
        return asdict(self)


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def resolve_scheduler_config(config):
    """Resolve public scheduling settings from model configuration."""
    config = _mapping(config, "config")
    pipeline = _mapping(config.get("pipeline", {}), "pipeline")
    memory = _mapping(pipeline.get("memory", {}), "pipeline.memory")
    options = _mapping(config.get("scheduler", {}), "scheduler")

    active = _integer(
        memory.get("max_active_requests", 1),
        "pipeline.memory.max_active_requests",
        1,
    )

    allowed = {
        "mode",
        "policy",
        "max_requests_per_batch",
        "max_batch_tokens",
        "max_queue_requests",
        "max_wait_ms",
    }

    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"Unknown scheduler settings: {sorted(unknown)}"
        )

    mode = options.get("mode", "interleaved")
    policy = options.get("policy", "fcfs")

    if mode not in ("interleaved", "continuous"):
        raise ValueError("Unsupported scheduler mode")

    if policy not in ("fcfs", "round_robin"):
        raise ValueError("Unsupported scheduler policy")

    batch_keys = {
        "max_requests_per_batch",
        "max_batch_tokens",
        "max_queue_requests",
        "max_wait_ms",
    }

    if mode == "interleaved":
        if policy != "fcfs" or batch_keys.intersection(options):
            raise ValueError(
                "Interleaved mode only supports its existing "
                "FCFS scheduler; batching options require "
                "continuous mode"
            )

        return SchedulerSettings(
            mode="interleaved",
            policy="fcfs",
            max_active_requests=active,
            max_requests_per_batch=1,
            max_batch_tokens=None,
            max_queue_requests=None,
            max_wait_ms=None,
        )

    batch_size = _integer(
        options.get("max_requests_per_batch", min(active, 4)),
        "scheduler.max_requests_per_batch",
        1,
    )

    if batch_size > active:
        raise ValueError(
            "Batch request limit exceeds resident request capacity"
        )

    token_budget = _integer(
        options.get("max_batch_tokens", 1024),
        "scheduler.max_batch_tokens",
        1,
    )

    if token_budget < batch_size:
        raise ValueError(
            "Batch token budget must fit at least one "
            "token per batched request"
        )

    queue_size = _integer(
        options.get("max_queue_requests", 32),
        "scheduler.max_queue_requests",
        0,
    )

    wait_ms = _integer(
        options.get("max_wait_ms", 0),
        "scheduler.max_wait_ms",
        0,
    )

    return SchedulerSettings(
        mode=mode,
        policy=policy,
        max_active_requests=active,
        max_requests_per_batch=batch_size,
        max_batch_tokens=token_budget,
        max_queue_requests=queue_size,
        max_wait_ms=wait_ms,
    )


def create_scheduler(
    config,
    stage_fns,
    sample_fn,
    *,
    stop_fn=None,
    trace_limit=2048,
    on_request_end=None,
):
    """Construct the selected scheduler implementation.

    Continuous mode will be connected to the V3.2 engine
    after its admission and batch-execution tests pass.
    """
    settings = resolve_scheduler_config(config)

    if settings.mode == "continuous":
        raise NotImplementedError(
            "V3.2 continuous batch execution is not implemented yet"
        )

    from llm_runtime.v3_pipeline_scheduler import PipelineScheduler

    return PipelineScheduler(
        stage_fns,
        sample_fn,
        max_active_requests=settings.max_active_requests,
        stop_fn=stop_fn,
        trace_limit=trace_limit,
        on_request_end=on_request_end,
    )
