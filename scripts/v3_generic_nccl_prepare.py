#!/usr/bin/env python3
"""Prepare isolated two-rank V3 BYOM Kubernetes smoke test.

Only generates YAML. Does not create AWS or Kubernetes resources.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "deploy/manifests/v3-qwen3-14b-benchmark.yaml"
OUTPUT = ROOT / "deploy/manifests/v3-byom-qwen3-smoke.yaml"
CONFIGMAP_OUTPUT = Path("/tmp/v3-byom-qwen3-smoke-code-configmap.yaml")

NAME = "v3-byom-qwen3-smoke"
SERVICE = NAME + "-headless"
CONFIGMAP_NAME = NAME + "-code"

RENDEZVOUS_PORT = 29678
CONTROL_PORT = 29679

CONFIG_PATH = ROOT / "examples/v3_byom_qwen3.json"

SOURCE_PATHS = [
    "src/llm_runtime/__init__.py",
    "src/llm_runtime/v3_execution_factory.py",
    "src/llm_runtime/v3_pipeline_scheduler.py",
    "src/llm_runtime/v3_ordered_rank_transport.py",
    "src/llm_runtime/providers/__init__.py",
    "src/llm_runtime/providers/v3_qwen3_partition.py",
    "src/llm_runtime/providers/v3_qwen3_process_factory.py",
    "src/llm_runtime/providers/v3_qwen3_stage_only_factory.py",
    "scripts/v3_provider_preflight.py",
    "scripts/v3_generic_nccl_runner.py",
    "examples/v3_byom_qwen3.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    if config_path != CONFIG_PATH.resolve():
        raise ValueError(
            "This Qwen3 smoke-test generator uses "
            "examples/v3_byom_qwen3.json only. "
            "Custom models need an adapted deployment profile."
        )

    if not BASE.is_file():
        raise FileNotFoundError(BASE)

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)

    from v3_generic_nccl_runner import load_config

    config = load_config(CONFIG_PATH)

    if config["pipeline"]["world_size"] != 2:
        raise ValueError("Exactly two ranks required")

    provider_path = config["provider"]["factory_path"]

    if provider_path != (
        "llm_runtime.providers.v3_qwen3_stage_only_factory:"
        "Qwen3StageOnlyFactory"
    ):
        raise ValueError(
            "This Kubernetes profile is specifically for "
            "the validated Qwen3 stage-only provider"
        )

    # This deployment mounts the retained rank-specific Qwen3 PVCs.
    kwargs = config["provider"]["kwargs"]

    if kwargs.get("checkpoint_dir") != "/models/qwen3-14b":
        raise ValueError("Unexpected checkpoint directory")

    if kwargs.get("split_at") != 20:
        raise ValueError("Expected tested 20/20 split")

    resources = [
        item
        for item in yaml.safe_load_all(BASE.read_text())
        if item is not None
    ]

    base_service = next(
        item for item in resources
        if item.get("kind") == "Service"
    )

    base_jobs = {}

    for item in resources:
        if item.get("kind") != "Job":
            continue

        name = item["metadata"]["name"]

        for rank in (0, 1):
            if name.endswith(f"-rank{rank}"):
                base_jobs[rank] = item

    if set(base_jobs) != {0, 1}:
        raise ValueError(
            "Baseline manifest must contain "
            "separate rank0 and rank1 Jobs"
        )

    # Build a ConfigMap with the actual current source files.
    data = {}
    items = []

    for i, relative_path in enumerate(SOURCE_PATHS):
        source = ROOT / relative_path

        if not source.is_file():
            raise FileNotFoundError(source)

        key = f"source_{i}"
        data[key] = source.read_text()

        items.append({
            "key": key,
            "path": relative_path,
        })

    size = sum(
        len(value.encode("utf-8"))
        for value in data.values()
    )

    if size >= 800_000:
        raise ValueError(
            f"Projected source size too large: {size}"
        )

    service = deepcopy(base_service)

    service["metadata"]["name"] = SERVICE
    service["metadata"]["namespace"] = "llm"
    service["spec"]["selector"] = {"app": NAME}

    for port in service["spec"]["ports"]:
        if port["name"] == "rendezvous":
            number = RENDEZVOUS_PORT
        else:
            number = CONTROL_PORT

        port["port"] = number
        port["targetPort"] = number

    jobs = []

    for rank in (0, 1):
        job = deepcopy(base_jobs[rank])

        job["metadata"]["name"] = f"{NAME}-rank{rank}"
        job["metadata"]["namespace"] = "llm"

        job["spec"]["completions"] = 1
        job["spec"]["parallelism"] = 1
        job["spec"]["backoffLimit"] = 0
        job["spec"]["activeDeadlineSeconds"] = 2700
        job["spec"].pop("completionMode", None)

        template = job["spec"]["template"]

        template["metadata"]["labels"] = {
            "app": NAME,
            "rank": str(rank),
        }

        pod = template["spec"]

        pod["hostname"] = f"{NAME}-rank{rank}"
        pod["subdomain"] = SERVICE
        pod["nodeSelector"] = {"pool": "gpu"}

        affinity = pod["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]

        affinity[0]["labelSelector"]["matchLabels"] = {
            "app": NAME
        }

        container = pod["containers"][0]

        container["command"] = [
            "python",
            "-u",
            "/workspace/scripts/v3_generic_nccl_runner.py",
        ]

        container["args"] = [
            "--config",
            "/workspace/examples/v3_byom_qwen3.json",
        ]

        if str(
            container["resources"]["limits"]["nvidia.com/gpu"]
        ) != "1":
            raise ValueError(
                "Expected one GPU per pod"
            )

        envs = container["env"]

        env_values = {
            "RANK": str(rank),
            "WORLD_SIZE": "2",
            "MASTER_ADDR": (
                f"{NAME}-rank0.{SERVICE}.llm.svc.cluster.local"
            ),
            "MASTER_PORT": str(RENDEZVOUS_PORT),
            "V3_CTRL_PORT": str(CONTROL_PORT),
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
        }

        for env in envs:
            key = env.get("name")

            if key in env_values:
                env.pop("valueFrom", None)
                env["value"] = env_values.pop(key)

        for key, value in env_values.items():
            envs.append({
                "name": key,
                "value": value,
            })

        code_volume = next(
            volume for volume in pod["volumes"]
            if volume["name"] == "test-code"
        )

        code_volume["projected"]["sources"] = [{
            "configMap": {
                "name": CONFIGMAP_NAME,
                "items": items,
            }
        }]

        code_mount = next(
            mount for mount in container["volumeMounts"]
            if mount["name"] == "test-code"
        )

        if code_mount["mountPath"] != "/workspace":
            raise ValueError(
                "Unexpected source mount path"
            )

        model_volume = next(
            volume for volume in pod["volumes"]
            if volume["name"] == "model-files"
        )

        claim = model_volume[
            "persistentVolumeClaim"
        ]["claimName"]

        expected_claim = f"model-files-qwen3-{rank}"

        if claim != expected_claim:
            raise ValueError(
                f"Unexpected model PVC: {claim}"
            )

        jobs.append(job)

    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": CONFIGMAP_NAME,
            "namespace": "llm",
        },
        "data": data,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        yaml.safe_dump_all(
            [service, *jobs],
            sort_keys=False,
        )
    )

    CONFIGMAP_OUTPUT.write_text(
        yaml.safe_dump(
            configmap,
            sort_keys=False,
        )
    )

    print("PASS: Generic runner deployment prepared")
    print("PASS: Original benchmark preserved")
    print("PASS: Two rank-specific GPU Jobs prepared")
    print("PASS: Existing Qwen3 PVCs retained")
    print(f"Manifest: {OUTPUT}")
    print(f"ConfigMap: {CONFIGMAP_OUTPUT}")
    print(f"Source bytes: {size}")


if __name__ == "__main__":
    main()
