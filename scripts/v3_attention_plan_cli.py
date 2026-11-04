import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

import yaml

from llm_runtime.v3_attention_plan import (
    resolve_attention_plan,
)
from llm_runtime.v3_manifest import ManifestError
from llm_runtime.v3_model_inspector import (
    InspectionError,
    fetch_config,
    inspect_config,
)


parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument(
    "--online",
    action="store_true",
    help="Verify layer count using Hugging Face config.json",
)

args = parser.parse_args()

try:
    manifest = yaml.safe_load(
        Path(args.manifest).read_text()
    ) or {}

    inspected = None

    if args.online:
        model = manifest.setdefault("model", {})
        model_id = model.get("id", "Qwen/Qwen3-14B")
        revision = model.get("revision", "main")

        config = fetch_config(model_id, revision)

        inspected = inspect_config(
            model_id,
            config,
            expected_layers=model.get("num_layers"),
            revision=revision,
        )

        model["num_layers"] = inspected["num_layers"]

    plan = resolve_attention_plan(manifest)

    if inspected:
        plan["layer_count_source"] = (
            "DOWNLOADED_CONFIG_JSON"
        )
        plan["inspected_adapter_status"] = (
            inspected["adapter_status"]
        )
        plan["inspected_checkpoint_status"] = (
            inspected["checkpoint_status"]
        )

except (
    ManifestError,
    InspectionError,
    ValueError,
    OSError,
    yaml.YAMLError,
) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(2)

print("PASS: static per-layer attention plan")
print(json.dumps(plan, indent=2))
