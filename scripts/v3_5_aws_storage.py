"""Read-only Kubernetes model PVC discovery."""

import argparse
import copy
import os
import re
from pathlib import Path

import yaml

from v3_5_aws_doctor import kubectl_client


REFERENCE_MODEL = "Qwen/Qwen3-14B"
REFERENCE_PVCS = (
    "model-files-qwen3-0",
    "model-files-qwen3-1",
)


def discover_storage(config, pvc_items, rank0=None, rank1=None):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Model configuration missing")

    if not isinstance(pvc_items, list):
        raise ValueError("Invalid Kubernetes PVC response")

    existing0 = model.get("pvc_rank0")
    existing1 = model.get("pvc_rank1")

    selected0 = rank0 if rank0 is not None else existing0
    selected1 = rank1 if rank1 is not None else existing1

    if bool(selected0) != bool(selected1):
        raise ValueError(
            "Both rank PVCs must be specified together"
        )

    if not selected0:
        if model.get("id") != REFERENCE_MODEL:
            raise ValueError(
                "Automatic PVC selection supports only "
                "the Qwen3-14B reference model; "
                "provide both PVC names explicitly"
            )

        selected0, selected1 = REFERENCE_PVCS

    for name in (selected0, selected1):
        if (
            not isinstance(name, str)
            or not re.fullmatch(
                r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?",
                name,
            )
            or len(name) > 253
        ):
            raise ValueError("Invalid PVC name")

    if selected0 == selected1:
        raise ValueError("Ranks must use distinct PVCs")

    found = {}

    for item in pvc_items:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}
        name = metadata.get("name")

        if name in (selected0, selected1):
            found[name] = item

    for name in (selected0, selected1):
        item = found.get(name)

        if item is None:
            raise ValueError(
                f"Required model PVC missing: {name}"
            )

        phase = (item.get("status") or {}).get("phase")

        if phase != "Bound":
            raise ValueError(
                f"Model PVC not Bound: {name}"
            )

        modes = (item.get("spec") or {}).get(
            "accessModes", []
        )

        if not isinstance(modes, list) or not any(
            mode in (
                "ReadWriteOnce",
                "ReadWriteMany",
                "ReadWriteOncePod",
            )
            for mode in modes
        ):
            raise ValueError(
                f"Unsupported PVC access mode: {name}"
            )

    result = copy.deepcopy(config)

    result["model"]["pvc_rank0"] = selected0
    result["model"]["pvc_rank1"] = selected1

    result["storage_discovery"] = {
        "pvc_rank0_bound": True,
        "pvc_rank1_bound": True,
        "checkpoint_contents_verified": False,
        "node_volume_placement_verified": False,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            ".llm-eks/deployment.g5.discovered.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".llm-eks/deployment.storage.yaml"
        ),
    )
    parser.add_argument("--pvc-rank0")
    parser.add_argument("--pvc-rank1")

    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        parser.error(
            "Output already exists; refusing to overwrite"
        )

    try:
        config = yaml.safe_load(args.config.read_text())

        if not isinstance(config, dict):
            raise ValueError("Invalid configuration")

        aws = config.get("aws") or {}

        region = aws.get("region")
        account = aws.get("account_id")
        cluster = aws.get("cluster_name")
        namespace = aws.get("namespace")

        if not (
            isinstance(region, str) and region
            and isinstance(account, str)
            and re.fullmatch(r"\d{12}", account)
            and isinstance(cluster, str) and cluster
            and isinstance(namespace, str) and namespace
        ):
            raise ValueError(
                "AWS target is incomplete"
            )

        context = (
            f"arn:aws:eks:{region}:{account}:"
            f"cluster/{cluster}"
        )

        kube = kubectl_client(context)

        response = kube([
            "get", "pvc", "-n", namespace
        ])

        discovered = discover_storage(
            config,
            response.get("items"),
            args.pvc_rank0,
            args.pvc_rank1,
        )

        args.output.parent.mkdir(
            parents=True, exist_ok=True, mode=0o700
        )

        old_umask = os.umask(0o077)
        try:
            with args.output.open("x") as stream:
                yaml.safe_dump(
                    discovered,
                    stream,
                    sort_keys=False,
                )
        finally:
            os.umask(old_umask)

    except (
        ValueError, RuntimeError, OSError,
        yaml.YAMLError,
    ) as exc:
        parser.error(str(exc))

    print(
        "Rank 0 PVC:",
        discovered["model"]["pvc_rank0"],
    )
    print(
        "Rank 1 PVC:",
        discovered["model"]["pvc_rank1"],
    )
    print("Both PVCs: Bound")
    print("Checkpoint contents: NOT VERIFIED")
    print("Volume placement: NOT VERIFIED")
    print("Saved:", args.output)
    print("No Kubernetes resources modified")
    print("V35_AWS_STORAGE_DISCOVERY_PASS")


if __name__ == "__main__":
    main()
