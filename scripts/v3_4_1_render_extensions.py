"""Render selected external extensions into GPU worker manifests."""

import json
import sys
from pathlib import Path

from llm_runtime.extensions import (
    PluginRegistry,
    load_plugin_context,
)
from llm_runtime.extensions.external import (
    load_external_extensions,
)
from llm_runtime.extensions.vllm_policy import (
    apply_vllm_policy,
)
from llm_runtime.extensions.k8s_resources import (
    apply_resource_overrides,
)


def main():
    if len(sys.argv) not in (2, 4) or (
        len(sys.argv) == 4 and sys.argv[2] != "--output-dir"
    ):
        raise SystemExit(
            "Usage: python scripts/v3_4_1_render_extensions.py "
            "extensions.json"
        )

    config = json.loads(Path(sys.argv[1]).read_text())

    allowed = {
        "schema_version",
        "backend",
        "modules",
        "selected_scheduler",
        "runtime_image",
        "resources",
    }

    if set(config) - allowed:
        raise ValueError("Unknown extension configuration fields")

    if (
        config.get("schema_version") != 1
        or config.get("backend") != "vllm"
        or not config.get("selected_scheduler")
    ):
        raise ValueError("Invalid extension configuration")

    root = Path(
        "infrastructure/profiles/v3-3/qwen3-14b"
    )

    context = load_plugin_context(
        model_capabilities_path=root / "model_capabilities.json",
        model_layout_path=root / "model_layout.json",
        backend="vllm",
        runtime={
            "execution_plan": json.loads(
                (root / "execution_plan.json").read_text()
            )
        },
    )

    registry = PluginRegistry()

    load_external_extensions(
        registry,
        config.get("modules", []),
    )

    policy = registry.build(
        "scheduler",
        config["selected_scheduler"],
        context,
    )

    # Validate every output before writing anything.
    source = Path("deploy/v3_3/rendered")
    outputs = {}

    for rank in (0, 1):
        filename = f"vllm-rank{rank}.json"

        deployment = json.loads(
            (source / filename).read_text()
        )

        selected = apply_vllm_policy(
            deployment,
            policy,
            runtime_image=config.get("runtime_image"),
        )
        if "resources" in config:
            selected = apply_resource_overrides(
                selected, config["resources"]
            )
        outputs[filename] = selected

    destination = Path(
        sys.argv[3] if len(sys.argv) == 4
        else "deploy/v3_4_1/rendered"
    )
    destination.mkdir(parents=True, exist_ok=True)

    for filename, deployment in outputs.items():
        (destination / filename).write_text(
            json.dumps(deployment, indent=2) + "\n"
        )

    print("V341_EXTERNAL_EXTENSION_OK")
    print("Selected:", config["selected_scheduler"])
    print("Runtime policy:", policy)
    print("GPU manifests:", len(outputs))
    print("No Kubernetes changes performed")


if __name__ == "__main__":
    main()
