"""Public, non-deploying CLI for the LLM inference framework."""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def yaml_module():
    try:
        import yaml
        return yaml
    except ImportError as exc:
        raise ValueError(
            "PyYAML required. Install with: "
            "pip install '.[config]'"
        ) from exc


def read_config(path):
    yaml = yaml_module()
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    return data


def exclusive_write(path, text):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(
            f"Output exists; refusing overwrite: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_umask = os.umask(0o077)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    finally:
        os.umask(old_umask)


def write_config(path, config):
    yaml = yaml_module()
    exclusive_write(
        path,
        yaml.safe_dump(config, sort_keys=False),
    )


def write_json(path, obj):
    exclusive_write(
        path,
        json.dumps(obj, indent=2) + "\n",
    )


def repo_scripts():
    root = Path.cwd().resolve()
    scripts = root / "scripts"

    if not (
        scripts / "v3_5_render_discovered.py"
    ).is_file():
        raise ValueError(
            "Reference commands require the repository checkout "
            "as the current directory"
        )

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    return root


def select_mode(config, requested):
    if requested != "auto":
        return requested
    return "plugin" if "plugins" in config else "reference"


def plugin_backend(config, allow_custom_code):
    if not allow_custom_code:
        raise ValueError(
            "Custom Python imports require --allow-custom-code"
        )

    from llm_runtime.extensions import PluginRegistry
    from llm_runtime.extensions.contracts import PluginContext
    from llm_runtime.extensions.external import (
        load_external_extensions
    )

    settings = config.get("plugins")
    if not isinstance(settings, dict):
        raise ValueError("Missing plugins configuration")

    modules = settings.get("modules")
    selected = settings.get("selected_backend")
    runtime = config.get("runtime", {})
    model = config.get("model", {})
    layout = config.get("model_layout", {})

    if (
        not isinstance(modules, list)
        or not modules
        or any(not isinstance(x, str) or not x for x in modules)
        or not isinstance(selected, str)
        or not selected
        or not isinstance(runtime, dict)
        or not isinstance(model, dict)
        or not isinstance(layout, dict)
        or not isinstance(runtime.get("backend"), str)
    ):
        raise ValueError(
            "Invalid plugin modules, selected backend, "
            "or context"
        )

    # This is an explicit, trusted-code operation.
    # The repository root is needed for local example modules.
    root = str(Path.cwd().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    context = PluginContext(
        model=model,
        layout=layout,
        runtime=runtime,
        backend=runtime["backend"],
    )

    registry = PluginRegistry()
    load_external_extensions(registry, modules)

    backend = registry.build(
        "backend", selected, context
    )
    return backend


def auto_reference_instance(config, api):
    """Choose g5.xlarge only if one matching group is evident."""
    if config.get("model", {}).get("id") != "Qwen/Qwen3-14B":
        return False

    aws = config["aws"]
    region = aws["region"]
    cluster = aws["cluster_name"]

    response = api([
        "eks", "list-nodegroups",
        "--cluster-name", cluster,
    ], region)

    matches = []

    for name in response.get("nodegroups", []):
        response = api([
            "eks", "describe-nodegroup",
            "--cluster-name", cluster,
            "--nodegroup-name", name,
        ], region)

        group = response.get("nodegroup", {})
        scaling = group.get("scalingConfig") or {}

        if (
            group.get("status") == "ACTIVE"
            and group.get("capacityType") == "ON_DEMAND"
            and group.get("instanceTypes") == ["g5.xlarge"]
            and type(scaling.get("maxSize")) is int
            and scaling["maxSize"] >= 2
        ):
            matches.append(name)

    if len(matches) == 1:
        config["deployment"]["gpu_instance_type"] = (
            "g5.xlarge"
        )
        return True

    return False


def configure(args):
    template = read_config(args.template)

    if args.aws:
        repo_scripts()

        from v3_5_aws_configure import (
            aws_json, default_region, discover
        )
        from v3_5_aws_enrich import enrich
        from v3_5_aws_cpu import discover_cpu
        from v3_5_aws_storage import discover_storage
        from v3_5_aws_doctor import kubectl_client

        api = lambda command, region: aws_json(
            command, region, args.profile
        )

        region = default_region(
            args.region, args.profile
        )
        config, count = discover(
            template, region, args.cluster, api
        )

        print("EKS clusters found:", count)

        if not config["aws"].get("cluster_name"):
            print(
                "[WARN] Select a cluster with --cluster; "
                "remaining discovery skipped"
            )
        else:
            if args.gpu_instance:
                config["deployment"][
                    "gpu_instance_type"
                ] = args.gpu_instance
            else:
                try:
                    if auto_reference_instance(config, api):
                        print(
                            "[PASS] Unique reference GPU "
                            "instance selected"
                        )
                except (ValueError, RuntimeError, KeyError):
                    print(
                        "[WARN] Automatic GPU instance "
                        "selection unavailable"
                    )

            try:
                config = enrich(config, api)
                print("[PASS] GPU discovery completed")
            except (ValueError, RuntimeError, KeyError):
                print(
                    "[WARN] GPU discovery incomplete; "
                    "review configuration"
                )

            try:
                config = discover_cpu(config, api)
                print("[PASS] CPU discovery completed")
            except (ValueError, RuntimeError, KeyError):
                print(
                    "[WARN] CPU discovery incomplete; "
                    "review configuration"
                )

            aws = config["aws"]
            context = (
                f"arn:aws:eks:{aws['region']}:"
                f"{aws['account_id']}:"
                f"cluster/{aws['cluster_name']}"
            )

            try:
                response = kubectl_client(context)([
                    "get", "pvc", "-n", aws["namespace"]
                ])
                config = discover_storage(
                    config, response.get("items")
                )
                print("[PASS] PVC discovery completed")
            except (ValueError, RuntimeError, OSError, KeyError):
                print(
                    "[WARN] Storage discovery incomplete; "
                    "inspect PVCs or supply explicit names"
                )
    else:
        config = copy.deepcopy(template)
        print("Created template; AWS discovery not requested")

    write_config(args.output, config)
    print("Local configuration:", args.output)
    print("No AWS or Kubernetes resources modified")
    print("CLI_CONFIGURE_PASS")


def reference_docs(config, cpu_override=None):
    repo_scripts()
    from v3_5_render_discovered import build

    return build(config, cpu_override)


def plan(args):
    config = read_config(args.config)
    mode = select_mode(config, args.mode)

    if mode == "reference":
        docs = reference_docs(config, args.cpu_nodegroup)
        metadata = docs["model-metadata.json"]["data"]
        output = json.loads(metadata["execution_plan.json"])
    else:
        backend = plugin_backend(
            config, args.allow_custom_code
        )
        if not callable(getattr(backend, "plan", None)):
            raise ValueError(
                "Selected backend does not provide plan()"
            )
        output = backend.plan()
        if not isinstance(output, dict):
            raise ValueError(
                "Backend plan() must return a JSON mapping"
            )

    write_json(args.output, output)
    print("Selected mode:", mode)
    print("Plan written:", args.output)
    print("No tests or deployments performed")
    print("CLI_PLAN_PASS")


def safe_output_directory(path):
    path = Path(path)
    target = path.resolve()

    if path.exists() or path.is_symlink():
        raise ValueError("Output directory already exists")

    root = Path.cwd().resolve()
    for protected in (
        root / "deploy/v3_3",
        root / "deploy/v3_4_1",
    ):
        protected = protected.resolve()
        if target == protected or protected in target.parents:
            raise ValueError(
                "Cannot overwrite frozen deployment files"
            )


def render(args):
    config = read_config(args.config)
    mode = select_mode(config, args.mode)

    if mode == "reference":
        docs = reference_docs(config, args.cpu_nodegroup)
    else:
        backend = plugin_backend(
            config, args.allow_custom_code
        )

        if not callable(getattr(backend, "render", None)):
            raise ValueError(
                "Selected backend has no render(plan) capability; "
                "provide an appropriate deployment adapter"
            )

        if args.plan:
            execution_plan = json.loads(
                Path(args.plan).read_text()
            )
        else:
            if not callable(getattr(backend, "plan", None)):
                raise ValueError(
                    "Backend must supply plan() or --plan"
                )
            execution_plan = backend.plan()

        docs = backend.render(execution_plan)

    if not isinstance(docs, dict) or not docs:
        raise ValueError(
            "Renderer must return a nonempty mapping "
            "of JSON filenames to objects"
        )

    for filename, document in docs.items():
        if (
            not isinstance(filename, str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json",
                filename,
            )
            or not isinstance(document, dict)
        ):
            raise ValueError(
                "Invalid renderer artifact name or object"
            )

        json.dumps(document)

    target = Path(args.output_dir)
    safe_output_directory(target)

    target.mkdir(parents=True, mode=0o700)

    for filename, document in sorted(docs.items()):
        (target / filename).write_text(
            json.dumps(document, indent=2) + "\n"
        )

    print("Selected mode:", mode)
    print("Generated JSON artifacts:", len(docs))
    print("Output:", target)
    print("No tests or deployments performed")
    print("CLI_RENDER_PASS")


def doctor(args):
    config = read_config(args.config)
    mode = select_mode(config, args.mode)

    if mode == "plugin":
        backend = plugin_backend(
            config, args.allow_custom_code
        )
        print("[PASS] Plugin registered and constructed")

        diagnose = getattr(backend, "diagnose", None)
        if callable(diagnose):
            print("Provider diagnostics:", diagnose())
        else:
            print(
                "[UNVERIFIED] No optional provider "
                "diagnostics supplied"
            )

        print(
            "[UNVERIFIED] Runtime correctness and "
            "deployment readiness not established"
        )
        print("CLI_DOCTOR_COMPLETE")
        return

    repo_scripts()
    from v3_5_aws_doctor import assess, kubectl_client

    client = None
    if args.check_k8s:
        aws = config["aws"]
        context = (
            f"arn:aws:eks:{aws['region']}:"
            f"{aws['account_id']}:"
            f"cluster/{aws['cluster_name']}"
        )
        client = kubectl_client(context)

    findings = assess(config, client)
    for level, code, message in findings:
        print(f"[{level}] {code}: {message}")

    failures = sum(
        level == "FAIL"
        for level, _, _ in findings
    )
    print("Blocking configuration findings:", failures)
    print("CLI_DOCTOR_COMPLETE")

    if failures:
        raise SystemExit(2)


def user_test(args):
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]

    if not command:
        print(
            "No tests selected. Framework regression tests "
            "are not part of the user workflow."
        )
        print(
            "Usage: llm-eks test -- "
            "python -m unittest your_tests"
        )
        print("CLI_TEST_NOT_RUN")
        return

    print("Executing user-selected command only", flush=True)
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="llm-eks",
        description=(
            "Distributed inference planning and deployment "
            "artifact tooling; no implicit deployment or tests"
        ),
    )
    commands = parser.add_subparsers(
        dest="command", required=True
    )

    configure_cmd = commands.add_parser("configure")
    configure_cmd.add_argument(
        "--template",
        default="examples/aws/deployment.example.yaml",
    )
    configure_cmd.add_argument(
        "--output",
        default=".llm-eks/deployment.yaml",
    )
    configure_cmd.add_argument(
        "--aws", action="store_true"
    )
    configure_cmd.add_argument("--region")
    configure_cmd.add_argument("--cluster")
    configure_cmd.add_argument("--profile")
    configure_cmd.add_argument("--gpu-instance")
    configure_cmd.set_defaults(func=configure)

    for name, handler in (
        ("plan", plan),
        ("render", render),
        ("doctor", doctor),
    ):
        command = commands.add_parser(name)
        command.add_argument(
            "--config",
            default=".llm-eks/deployment.yaml",
        )
        command.add_argument(
            "--mode",
            choices=("auto", "reference", "plugin"),
            default="auto",
        )
        command.add_argument(
            "--allow-custom-code",
            action="store_true",
            help="Import explicitly trusted external plugin modules",
        )

        if name in ("plan", "render"):
            command.add_argument("--cpu-nodegroup")

        if name == "plan":
            command.add_argument(
                "--output",
                default=".llm-eks/plan.json",
            )
        elif name == "render":
            command.add_argument("--plan")
            command.add_argument(
                "--output-dir",
                default="test-results/llm-eks-rendered",
            )
        else:
            command.add_argument(
                "--check-k8s",
                action="store_true",
            )

        command.set_defaults(func=handler)

    test_cmd = commands.add_parser("test")
    test_cmd.add_argument(
        "command", nargs=argparse.REMAINDER
    )
    test_cmd.set_defaults(func=user_test)

    args = parser.parse_args(argv)

    try:
        args.func(args)
    except (
        ValueError, OSError, KeyError,
        TypeError, json.JSONDecodeError,
    ) as exc:
        parser.exit(
            2, f"llm-eks: {type(exc).__name__}: {exc}\n"
        )


if __name__ == "__main__":
    main()
