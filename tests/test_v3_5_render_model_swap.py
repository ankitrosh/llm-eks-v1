"""Offline contract tests for the isolated model deployment renderer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "examples/model_swap/profile.json"
PLANNER = ROOT / "scripts/v3_5_plan_model_swap.py"
RENDERER = ROOT / "scripts/v3_5_render_model_swap.py"
DEFAULT = ROOT / "deploy/v3_3/rendered"


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


class TestModelSwapRenderer(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.work = Path(temp.name)
        self.plan_dir = self.work / "plan"
        self.plan = self.plan_dir / "execution_plan.json"
        self.bundle = self.work / "bundle"

        result = subprocess.run(
            [
                sys.executable,
                str(PLANNER),
                "--profile", str(PROFILE),
                "--output-dir", str(self.plan_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )

        self.assertTrue(self.plan.is_file())

    def render(self, *, success=True, error=None, **overrides):
        options = {
            "profile": PROFILE,
            "plan": self.plan,
            "output_dir": self.bundle,
            "namespace": "llm-model-swap",
            "name": "v35-qwen3-test",
            "pvc_rank0": "test-model-0",
            "pvc_rank1": "test-model-1",
        }
        options.update(overrides)

        command = [sys.executable, str(RENDERER)]

        for key, value in options.items():
            command.extend([
                "--" + key.replace("_", "-"),
                str(value),
            ])

        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        if success:
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("V35_MODEL_SWAP_RENDER_PASS", output)
            return

        self.assertNotEqual(result.returncode, 0, output)

        if error:
            self.assertIn(error, output)

    def test_render_isolated_bundle(self):
        self.render()

        files = sorted(self.bundle.glob("*.json"))
        self.assertEqual(len(files), 5)

        docs = {
            path.stem: load(path)
            for path in files
        }

        for document in docs.values():
            self.assertEqual(
                document["metadata"]["namespace"],
                "llm-model-swap",
            )

        for rank in (0, 1):
            doc = docs[f"vllm-rank{rank}"]
            self.assertEqual(doc["spec"]["replicas"], 0)

            template = doc["spec"]["template"]
            self.assertEqual(
                template["metadata"]["labels"],
                {
                    "app": "v35-qwen3-test",
                    "rank": str(rank),
                },
            )

            pod = template["spec"]
            command = pod["containers"][0]["command"]

            self.assertEqual(command[2], "/models/qwen3-14b")
            self.assertEqual(
                command[command.index("--node-rank") + 1],
                str(rank),
            )
            self.assertEqual(
                command[command.index("--master-addr") + 1],
                "v35-qwen3-test-rank0."
                "llm-model-swap.svc.cluster.local",
            )

            self.assertTrue(any(
                volume.get("persistentVolumeClaim", {}).get(
                    "claimName"
                ) == f"test-model-{rank}"
                for volume in pod["volumes"]
            ))

        self.assertEqual(
            docs["vllm-api-service"]["metadata"]["name"],
            "v35-qwen3-test-api",
        )
        self.assertEqual(
            docs["vllm-master-service"]["spec"]["clusterIP"],
            "None",
        )

        data = docs["model-metadata"]["data"]
        caps = json.loads(data["model_capabilities.json"])
        layout = json.loads(data["model_layout.json"])

        self.assertEqual(caps["num_layers"], 40)
        self.assertEqual(
            layout["verification"],
            "checkpoint_weight_index",
        )

    def test_reject_tampered_plan(self):
        plan = load(self.plan)
        plan["model"] = "/models/unexpected"
        save(self.plan, plan)

        self.render(
            success=False,
            error="Execution plan does not match",
        )

        self.assertFalse(self.bundle.exists())

    def test_reject_invalid_resource_name(self):
        self.render(
            success=False,
            name="Invalid_Name",
            error="Invalid Kubernetes resource name",
        )

    def test_reject_reserved_default_name(self):
        self.render(
            success=False,
            name="v33-vllm",
            error="Reserved default deployment name",
        )

    def test_reject_shared_rank_pvc(self):
        self.render(
            success=False,
            pvc_rank0="same-model-pvc",
            pvc_rank1="same-model-pvc",
            error="Separate per-rank PVC names required",
        )

    def test_reject_nonempty_output_directory(self):
        self.bundle.mkdir()
        (self.bundle / "keep.txt").write_text("untouched")

        self.render(
            success=False,
            error="Output directory must be empty",
        )

        self.assertEqual(
            (self.bundle / "keep.txt").read_text(),
            "untouched",
        )

    def test_reject_overwrite_default_manifests(self):
        self.render(
            success=False,
            output_dir=DEFAULT,
            error="Cannot overwrite the default deployment",
        )

    def test_reject_wrong_model_identity(self):
        profile = load(PROFILE)
        profile["expected_model_id"] = "Other/Checkpoint"

        alternate = self.work / "bad-profile.json"
        save(alternate, profile)

        self.render(
            success=False,
            profile=alternate,
            error="Model ID mismatch",
        )

    def test_reject_wrong_topology(self):
        profile = load(PROFILE)
        profile["deployment"]["pp"] = 4

        alternate = self.work / "bad-topology.json"
        save(alternate, profile)

        self.render(
            success=False,
            profile=alternate,
            error="Only the validated two-worker topology",
        )

    def test_reject_quantized_checkpoint(self):
        profile = load(PROFILE)
        caps_path = ROOT / profile["model_capabilities_path"]

        caps = load(caps_path)
        caps["quantization_config"] = {
            "quant_method": "gptq",
        }

        alternate_caps = self.work / "quantized-caps.json"
        save(alternate_caps, caps)

        profile["model_capabilities_path"] = str(
            alternate_caps
        )

        alternate = self.work / "quantized-profile.json"
        save(alternate, profile)

        self.render(
            success=False,
            profile=alternate,
            error="Unsupported Qwen3 checkpoint configuration",
        )


if __name__ == "__main__":
    unittest.main()
