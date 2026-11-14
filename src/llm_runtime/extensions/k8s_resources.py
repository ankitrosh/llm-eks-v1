"""Validated Kubernetes resource overrides for vLLM workers."""

from copy import deepcopy
from decimal import Decimal
import re

from .contracts import CompatibilityError


SECTIONS = {"requests", "limits"}
RESOURCES = {"cpu", "memory", "nvidia.com/gpu"}


def quantity(name, value):
    if name == "nvidia.com/gpu":
        if type(value) is int:
            value = str(value)
        if not isinstance(value, str) or not re.fullmatch(
            r"[1-9][0-9]*", value
        ):
            raise CompatibilityError(
                "GPU count must be a positive integer"
            )
        return int(value)

    if not isinstance(value, str):
        raise CompatibilityError(
            f"{name} must be a Kubernetes quantity string"
        )

    if name == "cpu":
        if re.fullmatch(r"[1-9][0-9]*m", value):
            return int(value[:-1])

        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
            millicores = Decimal(value) * 1000
            if millicores >= 1 and millicores == int(millicores):
                return int(millicores)

        raise CompatibilityError(
            "CPU must be positive cores or millicores"
        )

    if name == "memory":
        match = re.fullmatch(r"([0-9]+)(Mi|Gi)", value)
        if not match or int(match.group(1)) < 1:
            raise CompatibilityError(
                "Memory must be a positive Mi or Gi quantity"
            )

        multiplier = 1024 ** (2 if match.group(2) == "Mi" else 3)
        return int(match.group(1)) * multiplier

    raise CompatibilityError(f"Unsupported resource: {name}")


def apply_resource_overrides(deployment, overrides):
    """Apply optional resource changes without altering the input."""

    if not isinstance(overrides, dict) or not overrides:
        raise CompatibilityError(
            "Resource overrides must be a nonempty object"
        )

    if set(overrides) - SECTIONS:
        raise CompatibilityError("Unknown resource section")

    result = deepcopy(deployment)
    pod = result["spec"]["template"]["spec"]
    containers = pod["containers"]

    if len(containers) != 1 or containers[0]["name"] != "vllm":
        raise CompatibilityError("Expected one vLLM container")

    container = containers[0]
    resources = container["resources"]

    original_gpu = quantity(
        "nvidia.com/gpu",
        resources["limits"]["nvidia.com/gpu"],
    )

    for section, values in overrides.items():
        if not isinstance(values, dict) or not values:
            raise CompatibilityError(
                f"{section} must be a nonempty object"
            )

        if set(values) - RESOURCES:
            raise CompatibilityError(
                f"Unsupported resource in {section}"
            )

        for name, value in values.items():
            quantity(name, value)
            resources.setdefault(section, {})[name] = str(value)

    requests = resources.get("requests", {})
    limits = resources.get("limits", {})

    for name in ("cpu", "memory"):
        if name in requests and name in limits:
            if quantity(name, requests[name]) > quantity(name, limits[name]):
                raise CompatibilityError(
                    f"{name} request exceeds limit"
                )

    requested_gpu = quantity(
        "nvidia.com/gpu",
        requests.get("nvidia.com/gpu"),
    )
    limited_gpu = quantity(
        "nvidia.com/gpu",
        limits.get("nvidia.com/gpu"),
    )

    if requested_gpu != limited_gpu:
        raise CompatibilityError(
            "GPU request and limit must match"
        )

    if requested_gpu != original_gpu:
        raise CompatibilityError(
            "GPU override conflicts with the execution plan. "
            "Change GPU topology through the deployment planner."
        )

    return result
