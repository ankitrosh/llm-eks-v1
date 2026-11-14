"""Generic CLI interface tests; no AWS or GPU access."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/v3_5_cli/config.yaml"
TEMPLATE = ROOT / "examples/aws/deployment.example.yaml"


class TestPublicCLI(unittest.TestCase):

    def run_cli(self, *args, success=True):
        result = subprocess.run(
            [
                sys.executable, "-m",
                "llm_runtime.cli", *map(str, args),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout + result.stderr

        if success:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)

        return output

    def test_all_five_commands_available(self):
        for command in (
            "configure", "plan", "render",
            "doctor", "test",
        ):
            output = self.run_cli(command, "--help")
            self.assertIn("usage:", output)

    def test_no_implicit_tests(self):
        output = self.run_cli("test")
        self.assertIn("CLI_TEST_NOT_RUN", output)

    def test_explicit_plugin_permission_required(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_cli(
                "plan",
                "--config", CONFIG,
                "--output", Path(directory) / "plan.json",
                success=False,
            )

            self.assertIn(
                "--allow-custom-code", output
            )

    def test_generic_custom_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"

            self.run_cli(
                "plan",
                "--config", CONFIG,
                "--allow-custom-code",
                "--output", target,
            )

            plan = json.loads(target.read_text())
            self.assertEqual(
                plan["model"], "Example/CustomModel"
            )
            self.assertEqual(
                plan["backend"], "sample-runtime"
            )

    def test_generic_custom_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "rendered"

            self.run_cli(
                "render",
                "--config", CONFIG,
                "--allow-custom-code",
                "--output-dir", target,
            )

            manifest = json.loads(
                (
                    target / "example-configmap.json"
                ).read_text()
            )

            self.assertEqual(
                manifest["kind"], "ConfigMap"
            )

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_cli(
                "render",
                "--config", CONFIG,
                "--allow-custom-code",
                "--output-dir", directory,
                success=False,
            )

            self.assertIn(
                "already exists", output
            )

    def test_template_configuration_without_aws(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "deployment.yaml"

            self.run_cli(
                "configure",
                "--template", TEMPLATE,
                "--output", target,
            )

            self.assertTrue(target.is_file())
            self.assertEqual(
                target.stat().st_mode & 0o777,
                0o600,
            )


    def test_custom_diagnostics_can_be_unverified(self):
        output = self.run_cli(
            "doctor",
            "--config", CONFIG,
            "--allow-custom-code",
        )

        self.assertIn("[UNVERIFIED]", output)
        self.assertIn("CLI_DOCTOR_COMPLETE", output)

    def test_explicit_test_command_propagates_exit_code(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m", "llm_runtime.cli",
                "test", "--",
                sys.executable,
                "-c", "raise SystemExit(7)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 7)
        self.assertIn(
            "Executing user-selected command only",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
