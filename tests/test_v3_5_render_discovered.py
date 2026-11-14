"""Offline tests for configuration-driven reference rendering."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_5_render_discovered import (
    build,
    validate_output_dir,
)


class TestDiscoveredRenderer(unittest.TestCase):

    def setUp(self):
        self.config = {
            "schema_version": 1,
            "aws": {
                "account_id": "123456789012",
                "region": "us-east-2",
                "cluster_name": "test-cluster",
                "namespace": "llm",
            },
            "model": {
                "id": "Qwen/Qwen3-14B",
                "checkpoint_path": "/models/qwen3-14b",
                "pvc_rank0": "model-files-qwen3-0",
                "pvc_rank1": "model-files-qwen3-1",
            },
            "runtime": {
                "backend": "vllm",
                "pipeline_parallelism": 2,
                "tensor_parallelism": 1,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.85,
            },
            "deployment": {
                "capacity_type": "ON_DEMAND",
                "gpu_instance_type": "g5.xlarge",
                "gpu_nodegroup": "gpu-a10g",
                "gpu_node_selector": {
                    "eks.amazonaws.com/nodegroup": "gpu-a10g",
                },
                "replicas": 0,
                "auto_recovery": False,
            },
            "aws_discovery": {
                "required_vcpus_for_two_nodes": 8,
                "quota_value": 8,
                "nodegroup_max_size": 2,
            },
            "storage_discovery": {
                "pvc_rank0_bound": True,
                "pvc_rank1_bound": True,
            },
        }

    def test_auto_discovered_cpu_nodegroup(self):
        self.config["deployment"].update({
            "cpu_nodegroup": "cpu-workers",
            "cpu_node_selector": {
                "eks.amazonaws.com/nodegroup": "cpu-workers"
            },
        })

        docs = build(self.config)

        for name in (
            "cpu-coordinator.json",
            "cpu-recovery.json",
        ):
            spec = docs[name]["spec"]["template"]["spec"]

            self.assertEqual(
                spec["nodeSelector"],
                {
                    "eks.amazonaws.com/nodegroup":
                        "cpu-workers"
                },
            )

    def test_explicit_cpu_group_matches_discovery(self):
        self.config["deployment"].update({
            "cpu_nodegroup": "cpu-system",
            "cpu_node_selector": {
                "eks.amazonaws.com/nodegroup": "cpu-system"
            },
        })

        docs = build(self.config, "cpu-system")
        self.assertEqual(len(docs), 14)

    def test_conflicting_cpu_override_rejected(self):
        self.config["deployment"].update({
            "cpu_nodegroup": "cpu-system",
            "cpu_node_selector": {
                "eks.amazonaws.com/nodegroup": "cpu-system"
            },
        })

        with self.assertRaisesRegex(
            ValueError, "conflicts"
        ):
            build(self.config, "different-cpu-group")

    def test_invalid_discovered_cpu_selector_rejected(self):
        self.config["deployment"].update({
            "cpu_nodegroup": "cpu-system",
            "cpu_node_selector": {"pool": "gpu"},
        })

        with self.assertRaisesRegex(
            ValueError, "selector mismatch"
        ):
            build(self.config)

    def test_missing_cpu_discovery_requires_override(self):
        with self.assertRaisesRegex(
            ValueError, "not discovered"
        ):
            build(self.config)

    def render(self):
        return build(self.config, "cpu-system")

    def test_all_fourteen_resources_rendered(self):
        docs = self.render()
        self.assertEqual(len(docs), 14)

        for doc in docs.values():
            self.assertTrue(
                doc["metadata"]["name"].startswith("v35-")
            )

    def test_gpu_placement_and_storage(self):
        docs = self.render()

        for rank in (0, 1):
            spec = docs[
                f"vllm-rank{rank}.json"
            ]["spec"]["template"]["spec"]

            self.assertEqual(
                spec["nodeSelector"],
                {
                    "eks.amazonaws.com/nodegroup":
                        "gpu-a10g"
                },
            )
            self.assertEqual(
                spec["volumes"][0][
                    "persistentVolumeClaim"
                ]["claimName"],
                f"model-files-qwen3-{rank}",
            )

    def test_cpu_placement(self):
        docs = self.render()

        for name in (
            "cpu-coordinator.json",
            "cpu-recovery.json",
        ):
            spec = docs[name]["spec"]["template"]["spec"]

            self.assertEqual(
                spec["nodeSelector"],
                {
                    "eks.amazonaws.com/nodegroup":
                        "cpu-system"
                },
            )

    def test_namespace_and_services_rewritten(self):
        self.config["aws"]["namespace"] = "llm-demo"
        docs = self.render()

        for doc in docs.values():
            self.assertEqual(
                doc["metadata"]["namespace"],
                "llm-demo",
            )

        source = docs["cpu-source.json"]["data"]

        self.assertIn(
            'NS = "llm-demo"',
            source["v33_recovery_controller.py"],
        )
        self.assertIn(
            'namespace="llm-demo"',
            source["vllm_engine_controller.py"],
        )

        self.assertIn(
            "v35-recovery.llm-demo.svc.cluster.local",
            source["vllm_coordinator.py"],
        )

        plan = json.loads(
            docs["model-metadata.json"]["data"][
                "execution_plan.json"
            ]
        )

        for node in plan["nodes"]:
            command = node["command"]
            addr = command[
                command.index("--master-addr") + 1
            ]

            self.assertEqual(
                addr,
                "v35-vllm-rank0.llm-demo.svc.cluster.local",
            )

    def test_default_deployments_stay_zero(self):
        docs = self.render()

        for doc in docs.values():
            if doc["kind"] == "Deployment":
                self.assertEqual(doc["spec"]["replicas"], 0)

    def test_recovery_remains_paused(self):
        docs = self.render()

        state = json.loads(
            docs["recovery-state.json"]["data"]["state"]
        )

        self.assertEqual(state["mode"], "PAUSED")

    def test_unsupported_model_rejected(self):
        self.config["model"]["id"] = "Other/Model"

        with self.assertRaisesRegex(
            ValueError, "Qwen3-14B"
        ):
            self.render()

    def test_mismatched_gpu_selector_rejected(self):
        self.config["deployment"]["gpu_node_selector"] = {
            "pool": "cpu"
        }

        with self.assertRaisesRegex(
            ValueError, "selector"
        ):
            self.render()

    def test_missing_storage_evidence_rejected(self):
        self.config["storage_discovery"][
            "pvc_rank1_bound"
        ] = False

        with self.assertRaisesRegex(
            ValueError, "PVC"
        ):
            self.render()

    def test_insufficient_quota_rejected(self):
        self.config["aws_discovery"]["quota_value"] = 4

        with self.assertRaisesRegex(
            ValueError, "quota"
        ):
            self.render()

    def test_unsafe_replicas_rejected(self):
        self.config["deployment"]["replicas"] = 2

        with self.assertRaisesRegex(
            ValueError, "Unsafe"
        ):
            self.render()

    def test_existing_output_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "exists"
        ):
            validate_output_dir(
                ROOT / "deploy/v3_3/rendered"
            )

    def test_frozen_subdirectory_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "frozen"
        ):
            validate_output_dir(
                ROOT / "deploy/v3_3/new-output"
            )

    def test_config_not_mutated(self):
        original = deepcopy(self.config)
        self.render()
        self.assertEqual(self.config, original)


if __name__ == "__main__":
    unittest.main()
