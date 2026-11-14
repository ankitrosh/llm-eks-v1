"""Offline model PVC discovery tests."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_5_aws_storage import discover_storage


def pvc(name, phase="Bound", modes=None):
    return {
        "metadata": {"name": name},
        "spec": {
            "accessModes": modes or ["ReadWriteOnce"],
            "storageClassName": "llm-gp3",
        },
        "status": {
            "phase": phase,
            "capacity": {"storage": "80Gi"},
        },
    }


class TestStorageDiscovery(unittest.TestCase):

    def setUp(self):
        self.config = {
            "schema_version": 1,
            "model": {
                "id": "Qwen/Qwen3-14B",
                "pvc_rank0": None,
                "pvc_rank1": None,
            },
        }
        self.pvcs = [
            pvc("model-files-qwen3-0"),
            pvc("model-files-qwen3-1"),
        ]

    def test_reference_pvcs_selected(self):
        result = discover_storage(
            self.config, self.pvcs
        )

        self.assertEqual(
            result["model"]["pvc_rank0"],
            "model-files-qwen3-0",
        )
        self.assertEqual(
            result["model"]["pvc_rank1"],
            "model-files-qwen3-1",
        )

    def test_unrelated_pvc_not_selected(self):
        result = discover_storage(
            self.config,
            self.pvcs + [pvc("unrelated-volume")],
        )
        self.assertEqual(
            result["model"]["pvc_rank0"],
            "model-files-qwen3-0",
        )

    def test_missing_pvc_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "missing"
        ):
            discover_storage(
                self.config, self.pvcs[:1]
            )

    def test_unbound_pvc_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "not Bound"
        ):
            discover_storage(
                self.config,
                [
                    pvc("model-files-qwen3-0"),
                    pvc("model-files-qwen3-1", "Pending"),
                ],
            )

    def test_custom_pvcs(self):
        result = discover_storage(
            self.config,
            [pvc("custom-a"), pvc("custom-b")],
            "custom-a",
            "custom-b",
        )
        self.assertEqual(
            result["model"]["pvc_rank1"],
            "custom-b",
        )

    def test_shared_pvc_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "distinct"
        ):
            discover_storage(
                self.config,
                [pvc("shared")],
                "shared",
                "shared",
            )

    def test_custom_model_requires_selection(self):
        self.config["model"]["id"] = "Other/Model"

        with self.assertRaisesRegex(
            ValueError, "only"
        ):
            discover_storage(
                self.config, self.pvcs
            )

    def test_only_one_pvc_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "together"
        ):
            discover_storage(
                self.config,
                self.pvcs,
                rank0="model-files-qwen3-0",
            )

    def test_no_mutation(self):
        original = deepcopy(self.config)
        discover_storage(self.config, self.pvcs)
        self.assertEqual(self.config, original)


if __name__ == "__main__":
    unittest.main()
