"""Offline pre-deployment doctor tests."""

import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v3_5_aws_doctor.py"

spec = importlib.util.spec_from_file_location(
    "v35_doctor", SCRIPT
)
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


class TestDoctor(unittest.TestCase):

    def setUp(self):
        self.config = {
            "schema_version": 1,
            "aws": {
                "account_id": "123456789012",
                "region": "us-east-2",
                "cluster_name": "test-cluster",
                "namespace": "llm",
            },
            "runtime": {
                "backend": "vllm",
                "pipeline_parallelism": 2,
                "tensor_parallelism": 1,
            },
            "model": {
                "pvc_rank0": "model-0",
                "pvc_rank1": "model-1",
            },
            "deployment": {
                "gpu_nodegroup": "gpu-workers",
                "gpu_node_selector": {
                    "eks.amazonaws.com/nodegroup": "gpu-workers"
                },
                "replicas": 0,
            },
            "aws_discovery": {
                "required_vcpus_for_two_nodes": 8,
                "quota_value": 8,
                "nodegroup_max_size": 2,
                "nodegroup_desired_size": 0,
            },
        }

    @staticmethod
    def status(findings, code):
        return next(
            level
            for level, name, _ in findings
            if name == code
        )

    def test_idle_gpu_group_is_warning(self):
        result = doctor.assess(self.config)
        self.assertEqual(
            self.status(result, "nodegroup_desired"), "WARN"
        )
        self.assertEqual(
            self.status(result, "quota"), "WARN"
        )
        self.assertFalse(
            any(level == "FAIL" for level, _, _ in result)
        )

    def test_insufficient_quota_fails(self):
        self.config["aws_discovery"]["quota_value"] = 4
        self.assertEqual(
            self.status(
                doctor.assess(self.config), "quota"
            ),
            "FAIL",
        )

    def test_missing_pvc_fails(self):
        self.config["model"]["pvc_rank1"] = None
        self.assertEqual(
            self.status(
                doctor.assess(self.config), "pvc_config"
            ),
            "FAIL",
        )

    def test_insufficient_nodegroup_max_fails(self):
        self.config["aws_discovery"][
            "nodegroup_max_size"
        ] = 1
        self.assertEqual(
            self.status(
                doctor.assess(self.config), "nodegroup_max"
            ),
            "FAIL",
        )

    def test_invalid_gpu_selector_fails(self):
        self.config["deployment"]["gpu_node_selector"] = {
            "pool": "cpu"
        }
        self.assertEqual(
            self.status(
                doctor.assess(self.config), "gpu_group"
            ),
            "FAIL",
        )

    def test_ready_nodes_and_bound_pvcs(self):
        def kube(command):
            if command[:2] == ["get", "namespace"]:
                return {"metadata": {"name": "llm"}}

            if command == ["get", "nodes"]:
                return {
                    "items": [
                        {
                            "metadata": {
                                "labels": {
                                    "eks.amazonaws.com/nodegroup":
                                        "gpu-workers"
                                }
                            },
                            "status": {
                                "conditions": [{
                                    "type": "Ready",
                                    "status": "True",
                                }],
                                "allocatable": {
                                    "nvidia.com/gpu": "1"
                                },
                            },
                        }
                        for _ in range(2)
                    ]
                }

            if command[:2] == ["get", "pvc"]:
                return {
                    "items": [
                        {
                            "metadata": {"name": name},
                            "status": {"phase": "Bound"},
                        }
                        for name in ("model-0", "model-1")
                    ]
                }

            raise AssertionError(command)

        result = doctor.assess(self.config, kube)

        for code in (
            "kubernetes", "gpu_nodes", "model_pvcs"
        ):
            self.assertEqual(
                self.status(result, code), "PASS"
            )

    def test_zero_gpu_nodes_warns(self):
        def kube(command):
            if command[:2] == ["get", "namespace"]:
                return {}
            if command == ["get", "nodes"]:
                return {"items": []}
            if command[:2] == ["get", "pvc"]:
                return {"items": []}
            raise AssertionError(command)

        result = doctor.assess(self.config, kube)

        self.assertEqual(
            self.status(result, "gpu_nodes"), "WARN"
        )
        self.assertEqual(
            self.status(result, "model_pvcs"), "FAIL"
        )

    def test_kubernetes_permission_failure(self):
        def denied(command):
            raise RuntimeError("Access denied")

        result = doctor.assess(self.config, denied)

        self.assertEqual(
            self.status(result, "kubernetes"), "FAIL"
        )

    def test_config_remains_unchanged(self):
        original = deepcopy(self.config)
        doctor.assess(self.config)
        self.assertEqual(self.config, original)


if __name__ == "__main__":
    unittest.main()
