import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from llm_runtime.v3_manifest import (
    ManifestError,
    load_manifest,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "action", choices=["validate", "plan"]
)
parser.add_argument("manifest")
args = parser.parse_args()

try:
    result = load_manifest(args.manifest)
except ManifestError as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(2)

print("PASS: static manifest validation")

if args.action == "plan":
    print(json.dumps({
        "model_id": result["model_id"],
        "stages": result["stages"],
        "not_run": result["not_run"],
    }, indent=2))

    if not result["stages"]:
        print(
            "NOT RUN: model layer count or manual "
            "partition is required"
        )
else:
    print(json.dumps(result, indent=2))
