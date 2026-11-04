import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml

from llm_runtime.v3_attention_plan import (
    resolve_attention_plan,
)
from llm_runtime.v3_plugin_registry import (
    PluginError,
    collect_attention_plugins,
    inspect_loaded_plugin,
)


parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument(
    "--load",
    action="store_true",
    help="Import and inspect trusted custom plugin code",
)

args = parser.parse_args()

try:
    manifest = yaml.safe_load(
        Path(args.manifest).read_text()
    ) or {}

    plan = resolve_attention_plan(manifest)

    declared = collect_attention_plugins(plan)

    print("========== DECLARED PLUGINS ==========")
    print(json.dumps(declared, indent=2))

    if args.load:
        print("========== INTERFACE CHECK ==========")

        references = sorted({
            item["reference"] for item in declared
        })

        for reference in references:
            result = inspect_loaded_plugin(
                reference, "attention"
            )
            print(json.dumps(result, indent=2))

        print("PASS: Plugin interface inspection")
    else:
        print("PASS: Static plugin declarations")
        print("NOT RUN: Plugin imports and execution")

except (
    PluginError,
    ValueError,
    OSError,
    yaml.YAMLError,
) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(2)
