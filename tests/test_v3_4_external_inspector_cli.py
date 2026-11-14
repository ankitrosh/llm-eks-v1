"""Verify external inspector selection through the public CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestExternalInspectorCLI(unittest.TestCase):

    def test_user_inspector_executes_and_default_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = directory / "config.json"
            output = directory / "artifacts"

            config.write_text(json.dumps({
                "architectures": ["UserDemoForCausalLM"],
                "num_hidden_layers": 2,
            }))

            command = [
                sys.executable,
                "scripts/v3_4_generate_model_artifacts.py",
                "--config", str(config),
                "--model-id", "user/demo",
                "--output-dir", str(output),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([
                str(ROOT / "src"),
                str(ROOT),
                env.get("PYTHONPATH", ""),
            ])

            without_plugin = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(without_plugin.returncode, 0)
            self.assertFalse(output.exists())

            with_plugin = subprocess.run(
                command + [
                    "--inspector-module",
                    "examples.v3_4_1.custom_inspector",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                with_plugin.returncode,
                0,
                with_plugin.stderr,
            )

            layout = json.loads(
                (output / "model_layout.json").read_text()
            )
            capabilities = json.loads(
                (output / "model_capabilities.json").read_text()
            )

            self.assertEqual(layout["num_layers"], 2)
            self.assertEqual(
                layout["layers"][1]["module_path"],
                "transformer.blocks.1",
            )
            self.assertEqual(
                capabilities["architecture"],
                "UserDemoForCausalLM",
            )
            self.assertEqual(
                layout["verification"],
                "config_inferred_unverified",
            )

            print("V34_EXTERNAL_INSPECTOR_CLI_PASS")


if __name__ == "__main__":
    unittest.main()
