import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import yaml

from llm_runtime.v3_manifest import (
    ManifestError,
    validate,
)

from llm_runtime.v3_model_inspector import (
    InspectionError,
    fetch_config,
    inspect_config,
)


parser = argparse.ArgumentParser()
parser.add_argument("manifest")

source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--online", action="store_true")
source.add_argument("--config-json")

args = parser.parse_args()

try:
    manifest = yaml.safe_load(
        Path(args.manifest).read_text()
    )

    static = validate(manifest)

    model = manifest.get("model") or {}
    revision = model.get("revision", "main")

    if args.online:
        config = fetch_config(
            static["model_id"],
            revision,
        )
        metadata_source = "huggingface_config_json"
    else:
        config = json.loads(
            Path(args.config_json).read_text()
        )
        metadata_source = "local_config_json"

    result = inspect_config(
        static["model_id"],
        config,
        expected_layers=static["num_layers"],
        revision=revision,
    )

    result["metadata_source"] = metadata_source
    result["manifest_stages"] = static["stages"]

except (
    ManifestError,
    InspectionError,
    OSError,
    ValueError,
    yaml.YAMLError,
) as exc:
    print(f"FAIL: {exc}", file=sys.stderr)
    raise SystemExit(2)

print("PASS: checkpoint metadata inspected")
print("NOTE: runtime execution has NOT been validated")
print(json.dumps(result, indent=2))
