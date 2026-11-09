"""Validated V3.2 pipeline dispatch policy API.

CPU control plane only. No GPU or transport execution.
"""
from collections import deque
from dataclasses import dataclass
from importlib import import_module
import re


@dataclass(frozen=True)
class PipelineCandidate:
    microbatch_id: int
    stage: int
    request_ids: tuple[int, ...]
    tokens: int


@dataclass(frozen=True)
class PipelineContext:
    candidates: tuple[PipelineCandidate, ...]
    busy_stages: tuple[int, ...]
    in_flight: int
    max_inflight: int


@dataclass(frozen=True)
class PipelineDecision:
    microbatch_id: int
    stage: int


class PipelineBusy(RuntimeError):
    pass


class DownstreamFirst:
    def select_pipeline(self, context):
        return max(context.candidates, key=lambda c: c.stage)


class FillPipeline:
    def select_pipeline(self, context):
        return min(context.candidates, key=lambda c: c.stage)


def load_pipeline_policy(spec):
    if spec is None or spec == "downstream_first":
        return DownstreamFirst()
    if spec == "fill_pipeline":
        return FillPipeline()

    if isinstance(spec, str):
        if spec.count(":") != 1:
            raise ValueError("Use built-in name or module:Class")

        module, name = spec.split(":")
        pattern = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"

        if not re.fullmatch(pattern, module) or not re.fullmatch(
            pattern, name
        ):
            raise ValueError("Invalid pipeline policy import path")

        obj = import_module(module)
        for part in name.split("."):
            obj = getattr(obj, part)
        spec = obj()

    if not callable(getattr(spec, "select_pipeline", None)):
        raise TypeError("Policy needs select_pipeline(context)")

    return spec


def _resolve_config(config):
    settings = config.get("pipeline", {}).get("scheduling", {})

    if not isinstance(settings, dict):
        raise ValueError("pipeline.scheduling must be an object")

    allowed = {
        "policy",
        "max_inflight_microbatches",
        "max_queued_microbatches",
    }

    if set(settings) - allowed:
        raise ValueError("Unknown pipeline scheduling settings")

    inflight = settings.get("max_inflight_microbatches", 2)
    queued = settings.get("max_queued_microbatches", 32)

    for value, name in (
        (inflight, "max_inflight_microbatches"),
        (queued, "max_queued_microbatches"),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive integer")

    return settings.get("policy", "downstream_first"), inflight, queued


class PipelineDispatchController:
    """Single-owner, ordered microbatch dispatch controller."""

    def __init__(self, config, *, stages, policy=None):
        if type(stages) is not int or stages < 1:
            raise ValueError("stages must be positive")

        configured, self.max_inflight, self.max_queued = (
            _resolve_config(config)
        )

        ranks = config.get("pipeline", {}).get("num_ranks")
        if ranks is not None and (
            type(ranks) is not int or ranks != stages
        ):
            raise ValueError("Stage count must match num_ranks")

        self.policy = load_pipeline_policy(
            configured if policy is None else policy
        )

        self.stages = stages
        self._ready = [deque() for _ in range(stages)]
        self._busy = {}
        self._inflight = set()
        self._live_requests = set()
        self._finished = set()
        self._last_id = 0

    def enqueue(self, microbatch_id, request_ids, tokens):
        if type(microbatch_id) is not int or (
            microbatch_id <= self._last_id
        ):
            raise ValueError("Microbatch IDs must increase")

        request_ids = tuple(request_ids)

        if not request_ids or any(
            type(r) is not int or r < 1 for r in request_ids
        ) or len(request_ids) != len(set(request_ids)):
            raise ValueError("Invalid request IDs")

        if type(tokens) is not int or tokens < 1:
            raise ValueError("Invalid token count")

        if self._live_requests.intersection(request_ids):
            raise ValueError("Request already has unfinished batch")

        if len(self._ready[0]) >= self.max_queued:
            raise PipelineBusy("Stage-0 dispatch queue full")

        task = PipelineCandidate(
            microbatch_id, 0, request_ids, tokens
        )

        self._ready[0].append(task)
        self._live_requests.update(request_ids)
        self._last_id = microbatch_id

    def _context(self):
        candidates = []

        for stage, queue in enumerate(self._ready):
            if stage in self._busy or not queue:
                continue

            if stage == 0 and (
                len(self._inflight) >= self.max_inflight
            ):
                continue

            # Only the head is eligible: per-stage FIFO.
            candidates.append(queue[0])

        return PipelineContext(
            tuple(candidates),
            tuple(sorted(self._busy)),
            len(self._inflight),
            self.max_inflight,
        )

    def dispatch(self):
        context = self._context()

        if not context.candidates:
            return None

        selected = self.policy.select_pipeline(context)

        if isinstance(selected, PipelineCandidate):
            selected = PipelineDecision(
                selected.microbatch_id, selected.stage
            )

        if not isinstance(selected, PipelineDecision):
            raise TypeError("Policy must return PipelineDecision")

        if type(selected.microbatch_id) is not int or (
            type(selected.stage) is not int
        ):
            raise ValueError("Invalid pipeline decision")

        eligible = {
            (c.microbatch_id, c.stage): c
            for c in context.candidates
        }

        key = (selected.microbatch_id, selected.stage)

        if key not in eligible:
            raise ValueError("Policy selected ineligible work")

        task = self._ready[selected.stage].popleft()
        self._busy[selected.stage] = task

        if selected.stage == 0:
            self._inflight.add(task.microbatch_id)

        return task

    def complete(self, stage, microbatch_id):
        if type(stage) is not int or stage not in self._busy:
            raise ValueError("Stage is not busy")

        task = self._busy[stage]

        if type(microbatch_id) is not int or (
            task.microbatch_id != microbatch_id
        ):
            raise ValueError("Stage/completion mismatch")

        del self._busy[stage]

        if stage + 1 < self.stages:
            self._ready[stage + 1].append(
                PipelineCandidate(
                    microbatch_id,
                    stage + 1,
                    task.request_ids,
                    task.tokens,
                )
            )
            return False

        self._inflight.remove(microbatch_id)
        self._live_requests.difference_update(task.request_ids)
        self._finished.add(microbatch_id)
        return True

    def snapshot(self):
        return {
            "ready_by_stage": tuple(
                tuple(x.microbatch_id for x in q)
                for q in self._ready
            ),
            "busy_by_stage": tuple(
                self._busy[s].microbatch_id
                if s in self._busy else None
                for s in range(self.stages)
            ),
            "in_flight": tuple(sorted(self._inflight)),
            "completed": tuple(sorted(self._finished)),
        }
