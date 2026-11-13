"""Public, validated V3.2 batch policy extension interface."""

from dataclasses import dataclass
from importlib import import_module
import re


@dataclass(frozen=True)
class BatchCandidate:
    request_id: int
    phase: str
    ready_tokens: int


@dataclass(frozen=True)
class BatchContext:
    candidates: tuple[BatchCandidate, ...]
    max_requests_per_batch: int
    max_batch_tokens: int
    last_phase: str | None


@dataclass(frozen=True)
class BatchDecision:
    phase: str
    request_ids: tuple[int, ...]


def load_batch_policy(spec):
    """Accept a Python policy object or module:Class reference."""
    if spec is None:
        return None

    if isinstance(spec, str):
        if spec.count(":") != 1:
            raise ValueError(
                "Policy must use module:Class notation"
            )

        module_name, class_name = spec.split(":")

        valid = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"

        if (
            not re.fullmatch(valid, module_name)
            or not re.fullmatch(valid, class_name)
        ):
            raise ValueError("Invalid policy import path")

        obj = import_module(module_name)

        for component in class_name.split("."):
            obj = getattr(obj, component)

        spec = obj()

    if not callable(getattr(spec, "select_batch", None)):
        raise TypeError(
            "Batch policy must implement select_batch(context)"
        )

    return spec


def validate_batch_decision(decision, context):
    """Reject unsafe plugin decisions before model execution."""
    if not isinstance(decision, BatchDecision):
        raise TypeError("Policy must return BatchDecision")

    if decision.phase not in ("prefill", "decode"):
        raise ValueError("Invalid execution phase")

    ids = decision.request_ids

    if not isinstance(ids, tuple) or not ids:
        raise ValueError("Policy must select request IDs")

    if any(type(rid) is not int for rid in ids):
        raise ValueError("Request IDs must be integers")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request IDs")

    if len(ids) > context.max_requests_per_batch:
        raise ValueError("Batch request limit exceeded")

    # Each selected request requires at least one token slot.
    if len(ids) > context.max_batch_tokens:
        raise ValueError("Batch token budget exceeded")

    candidates = {
        candidate.request_id: candidate
        for candidate in context.candidates
    }

    for rid in ids:
        candidate = candidates.get(rid)

        if candidate is None:
            raise ValueError(
                f"Request {rid} is not active"
            )

        if candidate.phase != decision.phase:
            raise ValueError(
                "Cannot combine prefill and decode in this batch"
            )

        if candidate.ready_tokens < 1:
            raise ValueError("Request has no schedulable tokens")

    return decision
