"""CPU-only tests for validated vLLM resource customization."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from llm_runtime.extensions.contracts import CompatibilityError
from llm_runtime.extensions.k8s_resources import (
    apply_resource_overrides,
)

ROOT = Path(__file__).resolve().parents[1]

OVERRIDES = {
    "requests": {
        "cpu": "1500m",
        "memory": "9Gi",
        "nvidia.com/gpu": "1",
    },
    "limits": {
        "cpu": "3",
        "memory": "16Gi",
        "nvidia.com/gpu": "1",
    },
}


def original(rank=0):
    return json.loads(
        (ROOT / f"deploy/v3_3/rendered/vllm-rank{rank}.json")
        .read_text()
    )


class TestV341KubernetesResources(unittest.TestCase):

    def test_valid_overrides(self):
        source = original()
        result = apply_resource_overrides(source, OVERRIDES)

        resources = result["spec"]["template"]["spec"][
            "containers"
        ][0]["resources"]

        self.assertEqual(resources, OVERRIDES)
        self.assertEqual(
            source["spec"]["template"]["spec"][
                "containers"
            ][0]["resources"]["requests"]["cpu"],
            "2",
        )

    def test_memory_request_exceeds_limit(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {"requests": {"memory": "20Gi"}},
            )

    def test_cpu_request_exceeds_limit(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {
                    "requests": {"cpu": "3"},
                    "limits": {"cpu": "2"},
                },
            )

    def test_gpu_topology_change_rejected(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {
                    "requests": {"nvidia.com/gpu": "2"},
                    "limits": {"nvidia.com/gpu": "2"},
                },
            )

    def test_gpu_mismatch_rejected(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {"requests": {"nvidia.com/gpu": "2"}},
            )

    def test_unsupported_resource_rejected(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {"requests": {"unknown-resource": "1"}},
            )

    def test_invalid_cpu_quantity_rejected(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {"requests": {"cpu": "4cores"}},
            )

    def test_invalid_memory_quantity_rejected(self):
        with self.assertRaises(CompatibilityError):
            apply_resource_overrides(
                original(),
                {"limits": {"memory": "0Gi"}},
            )

    def test_renderer_updates_both_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads(
                (ROOT / "examples/v3_4_1/extensions.json")
                .read_text()
            )
            config["resources"] = OVERRIDES

            config_path = root / "extensions.json"
            config_path.write_text(json.dumps(config))
            output = root / "rendered"

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([
                str(ROOT / "src"),
                str(ROOT),
                env.get("PYTHONPATH", ""),
            ])

            command = [
                sys.executable,
                "scripts/v3_4_1_render_extensions.py",
                str(config_path),
                "--output-dir",
                str(output),
            ]

            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode, 0, result.stderr
            )

            for rank in (0, 1):
                deployment = json.loads(
                    (output / f"vllm-rank{rank}.json")
                    .read_text()
                )

                container = deployment["spec"]["template"][
                    "spec"
                ]["containers"][0]

                self.assertEqual(
                    container["resources"], OVERRIDES
                )
                self.assertEqual(
                    deployment["spec"]["replicas"], 0
                )
                self.assertEqual(
                    container["command"][:2],
                    ["vllm", "serve"],
                )

            print("V341_RESOURCE_RENDER_INTEGRATION_PASS")


if __name__ == "__main__":
    unittest.main()
