"""Compile a V3 manifest without loading model weights."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from llm_runtime.v3_model_inspector import fetch_config
from llm_runtime.v3_runtime_plan import (
    RuntimePlanError,
    compile_runtime_plan,
)


parser = argparse.ArgumentParser()

parser.add_argument("manifest")

source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--online", action="store_true")
source.add_argument("--config-json")

parser.add_argument("--output", required=True)
parser.add_argument("--overwrite", action="store_true")

args = parser.parse_args()

try:
    manifest = yaml.safe_load(
        Path(args.manifest).read_text()
    )

    if not isinstance(manifest, dict):
        raise RuntimePlanError("Invalid YAML manifest")

    model = manifest.get("model") or {}

    if args.online:
        checkpoint_config = fetch_config(
            model.get("id", "Qwen/Qwen3-14B"),
            model.get("revision", "main"),
        )
    else:
        checkpoint_config = json.loads(
            Path(args.config_json).read_text()
        )

    plan = compile_runtime_plan(
        manifest, checkpoint_config
    )

    output = Path(args.output)

    if output.exists() and not args.overwrite:
        raise RuntimePlanError(
            "Output already exists; use --overwrite"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(plan, indent=2) + "\n"
    )

except (
    RuntimePlanError,
    OSError,
    ValueError,
    yaml.YAMLError,
) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(2)

print("PASS: Runtime plan compiled")
print("Status:", plan["status"])
print("Stages:", plan["pipeline"]["stages"])
print("Runtime environment:")
print(json.dumps(plan["runtime_environment"], indent=2))
print("Saved:", output)
