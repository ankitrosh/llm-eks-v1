"""Translate selected user policies into vLLM launch arguments."""

from copy import deepcopy

from .contracts import CompatibilityError


FLAGS = {
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "scheduling_policy": "--scheduling-policy",
    "scheduler_cls": "--scheduler-cls",
    "attention_backend": "--attention-backend",
}


def validate_policy(policy):
    if not isinstance(policy, dict):
        raise CompatibilityError(
            "A vLLM policy factory must return a dictionary"
        )

    unknown = set(policy) - set(FLAGS)

    if unknown:
        raise CompatibilityError(
            f"Unsupported vLLM policy overrides: {sorted(unknown)}"
        )

    for key in ("max_num_seqs", "max_num_batched_tokens"):
        if key in policy:
            value = policy[key]
            if type(value) is not int or value < 1:
                raise CompatibilityError(
                    f"{key} must be a positive integer"
                )

    if "scheduling_policy" in policy:
        if policy["scheduling_policy"] not in (
            "fcfs", "priority"
        ):
            raise CompatibilityError(
                "Unsupported scheduling policy"
            )

    for key in ("scheduler_cls", "attention_backend"):
        if key in policy:
            value = policy[key]
            if not isinstance(value, str) or not value.strip():
                raise CompatibilityError(
                    f"{key} must be a nonempty string"
                )

    return policy


def apply_vllm_policy(deployment, policy, runtime_image=None):
    """Patch a generated vLLM GPU Deployment.

    Custom runtime classes must be installed in the supplied
    container image. Existence is verified separately.
    """
    validate_policy(policy)

    if "scheduler_cls" in policy or "attention_backend" in policy:
        if not runtime_image:
            raise CompatibilityError(
                "Custom runtime classes/backends require an "
                "explicit vLLM image containing their implementation"
            )

    result = deepcopy(deployment)

    container = result["spec"]["template"]["spec"][
        "containers"
    ][0]

    if container.get("name") != "vllm":
        raise CompatibilityError(
            "Expected a vLLM worker container"
        )

    command = list(container["command"])

    if command[:2] != ["vllm", "serve"]:
        raise CompatibilityError(
            "Unsupported vLLM launch command"
        )

    for key, value in policy.items():
        flag = FLAGS[key]

        if command.count(flag) > 1:
            raise CompatibilityError(
                f"Duplicate existing CLI flag: {flag}"
            )

        if flag in command:
            index = command.index(flag)

            if index + 1 >= len(command):
                raise CompatibilityError(
                    f"Missing value for {flag}"
                )

            del command[index:index + 2]

        command.extend([flag, str(value)])

    container["command"] = command

    if runtime_image:
        container["image"] = runtime_image

    return result
