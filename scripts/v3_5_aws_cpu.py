"""Read-only EKS CPU node-group discovery."""

import argparse
import copy
import os
import re
from pathlib import Path

import yaml

from v3_5_aws_configure import aws_json


def discover_cpu(config, api, requested=None):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Invalid configuration schema")

    aws = config.get("aws")
    deploy = config.get("deployment")

    if not isinstance(aws, dict) or not isinstance(deploy, dict):
        raise ValueError("Missing AWS or deployment configuration")

    region = aws.get("region")
    cluster = aws.get("cluster_name")
    account = aws.get("account_id")

    if not all(
        isinstance(x, str) and x and x != "auto"
        for x in (region, cluster, account)
    ):
        raise ValueError("Complete AWS discovery first")

    if not re.fullmatch(r"\d{12}", account):
        raise ValueError("Invalid AWS account ID")

    identity = api(["sts", "get-caller-identity"], region)
    if identity.get("Account") != account:
        raise ValueError("Active AWS account mismatch")

    response = api(
        ["eks", "list-nodegroups", "--cluster-name", cluster],
        region,
    )
    names = response.get("nodegroups")

    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name
        for name in names
    ):
        raise ValueError("Invalid EKS node-group listing")

    eligible = {}
    candidates = []

    for name in sorted(set(names)):
        response = api(
            [
                "eks", "describe-nodegroup",
                "--cluster-name", cluster,
                "--nodegroup-name", name,
            ],
            region,
        )
        group = response.get("nodegroup")

        if not isinstance(group, dict):
            raise ValueError("Invalid node-group description")

        scaling = group.get("scalingConfig") or {}
        labels = group.get("labels") or {}

        if (
            group.get("status") != "ACTIVE"
            or group.get("capacityType") != "ON_DEMAND"
            or type(scaling.get("maxSize")) is not int
            or scaling["maxSize"] < 1
            or labels.get("pool") == "gpu"
            or bool((group.get("health") or {}).get("issues"))
        ):
            continue

        instance_types = group.get("instanceTypes")

        # A custom launch template may hide the instance type.
        # Never guess such a node group's hardware.
        if (
            not isinstance(instance_types, list)
            or not instance_types
            or any(
                not isinstance(t, str) or not t
                for t in instance_types
            )
        ):
            continue

        descriptions = api(
            [
                "ec2", "describe-instance-types",
                "--instance-types",
                *sorted(set(instance_types)),
            ],
            region,
        ).get("InstanceTypes")

        if not isinstance(descriptions, list):
            raise ValueError("Invalid EC2 instance-type response")

        by_type = {
            item["InstanceType"]: item
            for item in descriptions
            if isinstance(item, dict)
            and isinstance(item.get("InstanceType"), str)
        }

        if not all(t in by_type for t in instance_types):
            continue

        valid = True

        for instance_type in instance_types:
            info = by_type[instance_type]
            vcpus = (info.get("VCpuInfo") or {}).get(
                "DefaultVCpus"
            )
            memory = (info.get("MemoryInfo") or {}).get(
                "SizeInMiB"
            )

            if (
                type(vcpus) is not int
                or vcpus < 2
                or type(memory) is not int
                or memory < 2048
                or any(
                    info.get(key) is not None
                    for key in (
                        "GpuInfo",
                        "FpgaInfo",
                        "InferenceAcceleratorInfo",
                        "NeuronInfo",
                    )
                )
            ):
                valid = False
                break

        if valid:
            candidates.append(name)
            eligible[name] = {
                "instance_types": instance_types,
                "desired_size": scaling.get("desiredSize"),
                "max_size": scaling["maxSize"],
            }

    selected = requested or deploy.get("cpu_nodegroup")

    if selected and selected not in candidates:
        raise ValueError(
            "Requested CPU node group is not an eligible candidate"
        )

    if not selected and len(candidates) == 1:
        selected = candidates[0]

    result = copy.deepcopy(config)
    result["deployment"]["cpu_nodegroup"] = selected or None
    result["deployment"]["cpu_node_selector"] = (
        {"eks.amazonaws.com/nodegroup": selected}
        if selected else None
    )

    discovery = result.setdefault("aws_discovery", {})
    discovery["cpu_nodegroup_candidates"] = candidates
    discovery["cpu_nodegroup_details"] = (
        eligible.get(selected) if selected else None
    )

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".llm-eks/deployment.storage.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".llm-eks/deployment.cpu.yaml"),
    )
    parser.add_argument("--nodegroup")
    parser.add_argument("--profile")
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        parser.error("Output exists; refusing to overwrite")

    try:
        config = yaml.safe_load(args.config.read_text())

        result = discover_cpu(
            config,
            lambda command, region: aws_json(
                command, region, args.profile
            ),
            args.nodegroup,
        )

        args.output.parent.mkdir(
            mode=0o700, parents=True, exist_ok=True
        )

        old_umask = os.umask(0o077)
        try:
            with args.output.open("x") as stream:
                yaml.safe_dump(
                    result, stream, sort_keys=False
                )
        finally:
            os.umask(old_umask)

    except (
        OSError, ValueError, RuntimeError, yaml.YAMLError
    ) as exc:
        parser.error(str(exc))

    group = result["deployment"]["cpu_nodegroup"]
    details = result["aws_discovery"][
        "cpu_nodegroup_details"
    ]

    print(
        "Eligible CPU groups:",
        len(result["aws_discovery"]["cpu_nodegroup_candidates"]),
    )
    print("Selected CPU node group:", group or "NONE")

    if details:
        print("CPU instance types:", details["instance_types"])
        print(
            "Desired/max nodes:",
            details["desired_size"], "/", details["max_size"],
        )

    print("Live CPU Pod scheduling: NOT VERIFIED")
    print("Saved:", args.output)
    print("No AWS or Kubernetes resources modified")
    print("V35_CPU_DISCOVERY_PASS")


if __name__ == "__main__":
    main()
