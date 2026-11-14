"""Public customization workflow acceptance tests."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/custom_extensions"


def read(path):
    return json.loads(Path(path).read_text())


def run_script(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


class TestCustomExtensions(unittest.TestCase):

    def test_scheduler_and_resource_customization(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = ROOT / "deploy/v3_3/rendered"
            original = {
                rank: (source / f"vllm-rank{rank}.json").read_bytes()
                for rank in (0, 1)
            }

            result = run_script(
                "scripts/v3_4_1_render_extensions.py",
                EXAMPLE / "extensions.json",
                "--output-dir",
                tmp,
            )

            self.assertEqual(
                result.returncode,
                0,
                result.stdout + result.stderr,
            )
            self.assertIn(
                "V341_EXTERNAL_EXTENSION_OK",
                result.stdout,
            )

            for rank in (0, 1):
                manifest = read(
                    Path(tmp) / f"vllm-rank{rank}.json"
                )
                container = manifest["spec"]["template"][
                    "spec"
                ]["containers"][0]

                command = container["command"]
                resources = container["resources"]

                for flag, value in (
                    ("--max-num-seqs", "2"),
                    ("--max-num-batched-tokens", "384"),
                    ("--scheduling-policy", "fcfs"),
                ):
                    self.assertEqual(
                        command[command.index(flag) + 1],
                        value,
                    )

                self.assertEqual(
                    resources["requests"]["cpu"],
                    "2500m",
                )
                self.assertEqual(
                    resources["requests"]["memory"],
                    "9Gi",
                )
                self.assertEqual(
                    resources["limits"]["memory"],
                    "14Gi",
                )
                self.assertEqual(
                    resources["limits"]["nvidia.com/gpu"],
                    "1",
                )
                self.assertEqual(
                    manifest["spec"]["replicas"],
                    0,
                )
                self.assertEqual(
                    (source / f"vllm-rank{rank}.json").read_bytes(),
                    original[rank],
                )

    def test_unknown_scheduler_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = read(EXAMPLE / "extensions.json")
            config["selected_scheduler"] = "not-registered"

            config_path = Path(tmp) / "extensions.json"
            config_path.write_text(json.dumps(config))

            output = Path(tmp) / "rendered"

            result = run_script(
                "scripts/v3_4_1_render_extensions.py",
                config_path,
                "--output-dir",
                output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "Unknown plugin",
                result.stdout + result.stderr,
            )
            self.assertFalse(output.exists())

    def test_external_model_inspector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "inspected"

            base_args = (
                "--config",
                EXAMPLE / "demo_model_config.json",
                "--model-id",
                "User/DemoModel",
                "--revision",
                "demo-only",
            )

            result = run_script(
                "scripts/v3_4_generate_model_artifacts.py",
                *base_args,
                "--output-dir",
                output,
                "--inspector-module",
                "examples.v3_4_1.custom_inspector",
            )

            self.assertEqual(
                result.returncode,
                0,
                result.stdout + result.stderr,
            )

            layout = read(output / "model_layout.json")
            caps = read(output / "model_capabilities.json")

            self.assertEqual(layout["num_layers"], 3)
            self.assertEqual(caps["num_layers"], 3)
            self.assertEqual(
                layout["architecture"],
                "UserDemoForCausalLM",
            )
            self.assertEqual(
                layout["verification"],
                "config_inferred_unverified",
            )

            rejected = run_script(
                "scripts/v3_4_generate_model_artifacts.py",
                *base_args,
                "--output-dir",
                root / "rejected",
            )

            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "Unsupported architecture",
                rejected.stdout + rejected.stderr,
            )

    def test_external_backend_http(self):
        result = run_script(
            "scripts/v3_4_1_test_custom_backend.py"
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )
        self.assertIn(
            "V341_CUSTOM_BACKEND_INTEGRATION_PASS",
            result.stdout,
        )
        self.assertIn(
            "Unsupported streaming rejected",
            result.stdout,
        )

    def test_native_scheduler_mount_evidence(self):
        root = ROOT / "deploy/v3_4_1/scheduler-mount-test"

        configmap = read(root / "scheduler-configmap.json")

        self.assertEqual(
            configmap["metadata"]["name"],
            "v341-user-scheduler",
        )
        self.assertIn(
            "V341_USER_SCHEDULER_INSTANTIATED",
            configmap["data"]["scheduler.py"],
        )

        for rank in (0, 1):
            deployment = read(
                root / f"vllm-rank{rank}.json"
            )

            self.assertEqual(deployment["spec"]["replicas"], 0)

            pod = deployment["spec"]["template"]["spec"]
            container = pod["containers"][0]
            command = container["command"]

            self.assertEqual(
                command[command.index("--scheduler-cls") + 1],
                "user_vllm_ext.scheduler.UserScheduler",
            )

            self.assertTrue(any(
                item["name"] == "PYTHONPATH"
                and item["value"] == "/opt/v341"
                for item in container["env"]
            ))

            self.assertTrue(any(
                volume["name"] == "v341-scheduler"
                and volume["configMap"]["name"]
                == "v341-user-scheduler"
                for volume in pod["volumes"]
            ))


if __name__ == "__main__":
    unittest.main()
