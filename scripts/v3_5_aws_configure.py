"""Read-only AWS discovery for a local deployment configuration."""

import argparse
import copy
import json
import os
import re
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML required: python3 -m pip install '.[config]'"
    ) from exc


def aws_json(command, region, profile=None):
    argv = [
        "aws", *command,
        "--region", region,
        "--output", "json",
        "--no-cli-pager",
    ]
    if profile:
        argv.extend(["--profile", profile])

    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=45
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "AWS CLI unavailable or timed out"
        ) from exc

    if result.returncode:
        # Avoid printing AWS stderr or session-specific details.
        raise RuntimeError(
            f"AWS discovery failed for {command[0]} "
            f"{command[1]}; check credentials, region "
            "and IAM permissions"
        )

    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "AWS CLI did not return valid JSON"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError("Unexpected AWS CLI response")

    return value


def default_region(override=None, profile=None):
    if override:
        return override

    for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if os.getenv(key):
            return os.environ[key]

    command = ["aws", "configure", "get", "region"]
    if profile:
        command.extend(["--profile", profile])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Could not inspect AWS CLI region"
        ) from exc

    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    raise ValueError(
        "No AWS region configured; pass --region"
    )


def discover(template, region, requested_cluster, api):
    if (
        not isinstance(template, dict)
        or template.get("schema_version") != 1
    ):
        raise ValueError(
            "Unsupported configuration template"
        )

    if not isinstance(template.get("aws"), dict):
        raise ValueError(
            "Configuration lacks AWS settings"
        )

    config = copy.deepcopy(template)
    settings = config["aws"]

    original_account = settings.get("account_id")

    identity = api(
        ["sts", "get-caller-identity"], region
    )
    account = identity.get("Account")

    if (
        not isinstance(account, str)
        or not re.fullmatch(r"\d{12}", account)
    ):
        raise ValueError(
            "AWS identity returned an invalid account ID"
        )

    if original_account not in (None, "auto", account):
        raise ValueError(
            "Configured AWS account differs from active identity"
        )

    result = api(["eks", "list-clusters"], region)
    clusters = result.get("clusters")

    if (
        not isinstance(clusters, list)
        or any(
            not isinstance(item, str) or not item
            for item in clusters
        )
    ):
        raise ValueError(
            "Invalid EKS cluster discovery result"
        )

    clusters = sorted(set(clusters))

    selected = (
        requested_cluster
        or settings.get("cluster_name")
    )

    if selected and selected not in clusters:
        raise ValueError(
            "Requested EKS cluster is not available "
            "in this region"
        )

    # Select automatically only when unambiguous.
    if not selected and len(clusters) == 1:
        selected = clusters[0]

    if selected:
        detail = api(
            ["eks", "describe-cluster", "--name", selected],
            region,
        )

        if (
            detail.get("cluster", {}).get("status")
            != "ACTIVE"
        ):
            raise ValueError(
                "Selected EKS cluster is not ACTIVE"
            )

    settings["account_id"] = account
    settings["region"] = region
    settings["cluster_name"] = selected or None

    return config, len(clusters)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--template",
        type=Path,
        default=Path(
            "examples/aws/deployment.example.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".llm-eks/deployment.yaml"),
    )
    parser.add_argument("--region")
    parser.add_argument("--cluster")
    parser.add_argument(
        "--profile",
        help="Optional configured AWS CLI profile",
    )

    args = parser.parse_args()

    if args.output.is_symlink() or args.output.exists():
        parser.error(
            "Output already exists; refusing to overwrite"
        )

    try:
        template = yaml.safe_load(
            args.template.read_text()
        )

        settings = (
            template.get("aws", {})
            if isinstance(template, dict)
            else {}
        )

        template_region = settings.get("region")

        region = default_region(
            args.region or (
                template_region
                if template_region not in (None, "auto")
                else None
            ),
            args.profile,
        )

        config, count = discover(
            template,
            region,
            args.cluster,
            lambda command, selected_region: aws_json(
                command, selected_region, args.profile
            ),
        )

        args.output.parent.mkdir(
            parents=True, exist_ok=True, mode=0o700
        )

        old_umask = os.umask(0o077)
        try:
            with args.output.open(
                "x", encoding="utf-8"
            ) as stream:
                yaml.safe_dump(
                    config, stream, sort_keys=False
                )
        finally:
            os.umask(old_umask)

    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print("AWS identity and region discovered")
    print("No credentials saved")
    print("EKS clusters found:", count)
    print(
        "EKS cluster:",
        "selected"
        if config["aws"]["cluster_name"]
        else "selection required",
    )
    print(
        "Model PVCs and GPU node selection "
        "still require verification"
    )
    print("Saved local configuration:", args.output)
    print("No AWS or Kubernetes resources changed")
    print("V35_AWS_CONFIG_DISCOVERY_PASS")


if __name__ == "__main__":
    main()
