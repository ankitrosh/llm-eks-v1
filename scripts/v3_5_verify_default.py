"""Offline validation of the default Qwen3-14B deployment."""

import json
from pathlib import Path

ROOT = Path("infrastructure/profiles/v3-3/qwen3-14b")
DEPLOY = Path("deploy/v3_3/rendered")


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_deployment(name):
    document = load(DEPLOY / f"{name}.json")
    require(document["kind"] == "Deployment", name)
    require(document["metadata"]["namespace"] == "llm", name)
    require(document["spec"]["replicas"] == 0,
            f"{name} must have zero replicas")
    return document


def main():
    capabilities = load(ROOT / "model_capabilities.json")
    layout = load(ROOT / "model_layout.json")
    plan = load(ROOT / "execution_plan.json")

    require(
        capabilities["model_id"] == "Qwen/Qwen3-14B",
        "Unexpected default model",
    )
    require(
        capabilities["num_layers"] == 40,
        "Unexpected model layer count",
    )
    require(
        layout["verification"] == "checkpoint_weight_index",
        "Model layout is not checkpoint-index verified",
    )
    require(
        plan["backend"] == "vllm"
        and plan["pipeline_parallel_size"] == 2
        and plan["tensor_parallel_size"] == 1,
        "Unexpected execution topology",
    )
    require(
        len(plan["nodes"]) == 2,
        "Expected two distributed workers",
    )

    for rank in (0, 1):
        name = f"vllm-rank{rank}"
        deployment = check_deployment(name)
        pod = deployment["spec"]["template"]["spec"]

        require(len(pod["containers"]) == 1, name)
        container = pod["containers"][0]
        command = container["command"]

        require(
            container["image"] == plan["image"],
            f"{name}: image mismatch",
        )
        require(
            container["resources"]["requests"]["nvidia.com/gpu"] == "1",
            f"{name}: incorrect GPU request",
        )
        require(
            container["resources"]["limits"]["nvidia.com/gpu"] == "1",
            f"{name}: incorrect GPU limit",
        )
        require(
            "--node-rank" in command
            and command[command.index("--node-rank") + 1] == str(rank),
            f"{name}: rank mismatch",
        )
        require(
            "--pipeline-parallel-size" in command
            and command[
                command.index("--pipeline-parallel-size") + 1
            ] == "2",
            f"{name}: pipeline mismatch",
        )
        require(
            any(
                volume.get("persistentVolumeClaim", {}).get(
                    "claimName"
                ) == f"model-files-qwen3-{rank}"
                for volume in pod["volumes"]
            ),
            f"{name}: model PVC mismatch",
        )

        print(f"PASS: GPU worker rank {rank}")

    check_deployment("cpu-coordinator")
    recovery = check_deployment("cpu-recovery")

    recovery_env = {
        item["name"]: item.get("value")
        for item in recovery["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
    }
    require(
        recovery_env.get("AUTO_RECOVER") == "false",
        "Automatic recovery must be disabled",
    )

    state = load(DEPLOY / "recovery-state.json")
    recovery_state = json.loads(state["data"]["state"])
    require(
        recovery_state["mode"] == "PAUSED",
        "Recovery state must start paused",
    )

    print("PASS: CPU coordinator and recovery")
    print("PASS: All deployment replicas are zero")
    print("PASS: Recovery initially paused")
    print("PASS: Pinned execution image matches worker manifests")
    print("PASS: No cluster changes")
    print("V35_DEFAULT_WORKFLOW_PASS")


if __name__ == "__main__":
    main()
