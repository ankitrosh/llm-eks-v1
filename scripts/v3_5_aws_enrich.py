"""Read-only GPU node group and quota discovery."""

import argparse
import copy
import os
from pathlib import Path

import yaml

from v3_5_aws_configure import aws_json


def enrich(config, api, requested_nodegroup=None):
    if not isinstance(config, dict) or config.get(
        "schema_version"
    ) != 1:
        raise ValueError("Invalid deployment config")

    aws = config.get("aws")
    deploy = config.get("deployment")
    runtime = config.get("runtime")

    if not all(
        isinstance(x, dict)
        for x in (aws, deploy, runtime)
    ):
        raise ValueError("Missing configuration sections")

    region, cluster, account = (
        aws.get(x)
        for x in ("region", "cluster_name", "account_id")
    )

    if any(
        x in (None, "", "auto")
        for x in (region, cluster, account)
    ):
        raise ValueError(
            "Run AWS configure and select a cluster first"
        )

    if (
        runtime.get("backend") != "vllm"
        or (
            runtime.get("pipeline_parallelism"),
            runtime.get("tensor_parallelism"),
        ) != (2, 1)
    ):
        raise ValueError(
            "Only two-worker PP=2, TP=1 topology is supported"
        )

    kind = deploy.get("capacity_type")

    if kind not in ("ON_DEMAND", "SPOT"):
        raise ValueError(
            "Only ON_DEMAND and SPOT are supported"
        )

    instance = deploy.get("gpu_instance_type")

    if (
        not isinstance(instance, str)
        or not instance.startswith("g5.")
    ):
        raise ValueError(
            "This discovery workflow supports G5 only"
        )

    identity = api(
        ["sts", "get-caller-identity"], region
    )

    if identity.get("Account") != account:
        raise ValueError(
            "Active AWS account does not match local configuration"
        )

    response = api(
        [
            "ec2", "describe-instance-types",
            "--instance-types", instance,
        ],
        region,
    )

    types = response.get("InstanceTypes", [])

    if (
        len(types) != 1
        or types[0].get("InstanceType") != instance
    ):
        raise ValueError(
            "GPU instance type was not verified"
        )

    info = types[0]
    vcpus = info.get("VCpuInfo", {}).get(
        "DefaultVCpus"
    )
    gpus = sum(
        gpu.get("Count", 0)
        for gpu in info.get("GpuInfo", {}).get(
            "Gpus", []
        )
    )

    if (
        type(vcpus) is not int
        or vcpus <= 0
        or gpus != 1
    ):
        raise ValueError(
            "Expected one GPU and valid vCPU count"
        )

    required = 2 * vcpus

    response = api(
        [
            "eks", "list-nodegroups",
            "--cluster-name", cluster,
        ],
        region,
    )

    names = response.get("nodegroups")

    if (
        not isinstance(names, list)
        or any(not isinstance(x, str) for x in names)
    ):
        raise ValueError(
            "Invalid EKS node group list"
        )

    groups = {}

    for name in sorted(set(names)):
        response = api(
            [
                "eks", "describe-nodegroup",
                "--cluster-name", cluster,
                "--nodegroup-name", name,
            ],
            region,
        )

        group = response.get("nodegroup", {})

        if not isinstance(group, dict):
            raise ValueError(
                "Invalid node group description"
            )

        groups[name] = group

    def matches(group):
        return (
            group.get("status") == "ACTIVE"
            and group.get("capacityType") == kind
            and instance in (
                group.get("instanceTypes") or []
            )
        )

    candidates = [
        name
        for name, group in groups.items()
        if matches(group)
    ]

    if (
        requested_nodegroup
        and requested_nodegroup not in candidates
    ):
        raise ValueError(
            "Requested node group is not a verified matching group"
        )

    selected = requested_nodegroup or (
        candidates[0] if len(candidates) == 1
        else None
    )

    quota_name = (
        "Running On-Demand G and VT instances"
        if kind == "ON_DEMAND"
        else "All G and VT Spot Instance Requests"
    )

    quota = None
    quota_state = "UNAVAILABLE"

    try:
        response = api(
            [
                "service-quotas",
                "list-service-quotas",
                "--service-code", "ec2",
            ],
            region,
        )

        quotas = response.get("Quotas", [])

        quota = next(
            (
                item for item in quotas
                if item.get("QuotaName") == quota_name
            ),
            None,
        )

        if (
            quota is not None
            and type(quota.get("Value"))
            in (float, int)
        ):
            quota_state = (
                "BELOW_REQUIRED_TOTAL"
                if quota["Value"] < required
                else "LIMIT_ENOUGH_USAGE_UNCHECKED"
            )
        else:
            quota = None

    except RuntimeError:
        # Missing Service Quotas permission does not
        # invalidate the discovered node group.
        pass

    result = copy.deepcopy(config)
    output = result["deployment"]

    output["gpu_nodegroup"] = selected

    output["gpu_node_selector"] = (
        {"eks.amazonaws.com/nodegroup": selected}
        if selected else None
    )

    scaling = (
        groups[selected].get("scalingConfig") or {}
        if selected else {}
    )

    result["aws_discovery"] = {
        "instance_vcpus": vcpus,
        "instance_gpus": gpus,
        "required_vcpus_for_two_nodes": required,
        "candidate_gpu_nodegroups": candidates,
        "quota_name": quota_name,
        "quota_code": (
            quota.get("QuotaCode") if quota else None
        ),
        "quota_value": (
            quota.get("Value") if quota else None
        ),
        "quota_status": quota_state,
        "nodegroup_max_size": scaling.get("maxSize"),
        "nodegroup_desired_size": scaling.get(
            "desiredSize"
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".llm-eks/deployment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".llm-eks/deployment.discovered.yaml"
        ),
    )
    parser.add_argument("--nodegroup")
    parser.add_argument("--profile")

    args = parser.parse_args()

    if (
        args.output.is_symlink()
        or args.output.exists()
    ):
        parser.error(
            "Output already exists; refusing to overwrite"
        )

    try:
        config = yaml.safe_load(
            args.config.read_text()
        )

        discovered = enrich(
            config,
            lambda command, region: aws_json(
                command, region, args.profile
            ),
            args.nodegroup,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
            mode=0o700,
        )

        old_umask = os.umask(0o077)

        try:
            with args.output.open(
                "x", encoding="utf-8"
            ) as stream:
                yaml.safe_dump(
                    discovered,
                    stream,
                    sort_keys=False,
                )
        finally:
            os.umask(old_umask)

    except (
        ValueError, OSError, RuntimeError
    ) as exc:
        parser.error(str(exc))

    details = discovered["aws_discovery"]

    print(
        "GPU instance vCPUs:",
        details["instance_vcpus"],
    )
    print(
        "Required vCPUs for two nodes:",
        details["required_vcpus_for_two_nodes"],
    )
    print(
        "Matching node groups:",
        len(details["candidate_gpu_nodegroups"]),
    )
    print(
        "Node group:",
        "selected"
        if discovered["deployment"]["gpu_nodegroup"]
        else "selection required",
    )
    print(
        "Quota assessment:",
        details["quota_status"],
    )
    print(
        "Node count desired/max:",
        details["nodegroup_desired_size"],
        "/",
        details["nodegroup_max_size"],
    )
    print(
        "PVCs, device plugin, available quota usage "
        "and node readiness: NOT VERIFIED"
    )
    print("Saved:", args.output)
    print("No AWS or Kubernetes changes performed")
    print("V35_AWS_GPU_DISCOVERY_PASS")


if __name__ == "__main__":
    main()
