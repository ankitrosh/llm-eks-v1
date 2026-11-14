"""CPU-only tests for the public model planning workflow."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "examples/model_swap/profile.json"
SCRIPT = ROOT / "scripts/v3_5_plan_model_swap.py"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


class TestModelSwap(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

        self.profile = read_json(PROFILE)
        self.caps = read_json(
            ROOT / self.profile["model_capabilities_path"]
        )
        self.layout = read_json(
            ROOT / self.profile["model_layout_path"]
        )

        self.caps_path = self.work / "capabilities.json"
        self.layout_path = self.work / "layout.json"
        self.profile_path = self.work / "profile.json"
        self.out = self.work / "output"

        self.profile["model_capabilities_path"] = str(
            self.caps_path
        )
        self.profile["model_layout_path"] = str(
            self.layout_path
        )

    def run_plan(self, *, success, error=None):
        write_json(self.caps_path, self.caps)
        write_json(self.layout_path, self.layout)
        write_json(self.profile_path, self.profile)

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--profile",
                str(self.profile_path),
                "--output-dir",
                str(self.out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        if success:
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("V35_MODEL_SWAP_PLAN_PASS", output)

            plan_path = self.out / "execution_plan.json"
            self.assertTrue(plan_path.exists())

            return read_json(plan_path)

        self.assertNotEqual(result.returncode, 0, output)
        self.assertFalse(
            (self.out / "execution_plan.json").exists()
        )

        if error:
            self.assertIn(error, output)

    def test_default_checkpoint_plan(self):
        plan = self.run_plan(success=True)

        self.assertEqual(
            plan["pipeline_parallel_size"], 2
        )
        self.assertEqual(len(plan["nodes"]), 2)

    def test_custom_model_mount_path(self):
        self.profile["deployment"]["model_path"] = (
            "/models/my-qwen3-checkpoint"
        )

        plan = self.run_plan(success=True)

        self.assertEqual(
            plan["model"],
            "/models/my-qwen3-checkpoint",
        )

    def test_reject_model_id_mismatch(self):
        self.profile["expected_model_id"] = "Other/Model"

        self.run_plan(
            success=False,
            error="Model ID does not match profile",
        )

    def test_reject_unsupported_architecture(self):
        self.caps["architecture"] = "OtherForCausalLM"
        self.layout["architecture"] = "OtherForCausalLM"

        self.run_plan(
            success=False,
            error="Unsupported model architecture",
        )

    def test_reject_quantized_checkpoint(self):
        self.caps["quantization_config"] = {
            "quant_method": "gptq"
        }

        self.run_plan(
            success=False,
            error="Quantized models are not validated",
        )

    def test_reject_non_bf16_checkpoint(self):
        self.caps["checkpoint_dtype"] = "float16"

        self.run_plan(
            success=False,
            error="Expected BF16 checkpoint metadata",
        )

    def test_reject_invalid_pipeline_topology(self):
        self.profile["deployment"]["pp"] = 4

        self.run_plan(
            success=False,
            error="This workflow currently supports",
        )

    def test_reject_excessive_context_length(self):
        self.profile["deployment"]["max_model_len"] = 50000

        self.run_plan(
            success=False,
            error="Requested context exceeds",
        )

    def test_reject_unverified_checkpoint_index(self):
        self.caps["source"]["index_sha256"] = None
        self.layout["source"]["index_sha256"] = None

        self.run_plan(
            success=False,
            error="Verified artifacts require a checkpoint index",
        )

    def test_reject_single_layer_pipeline(self):
        self.caps["num_layers"] = 1
        self.layout["num_layers"] = 1
        self.layout["layers"] = self.layout["layers"][:1]

        self.run_plan(
            success=False,
            error="Two pipeline stages require at least two layers",
        )


if __name__ == "__main__":
    unittest.main()
