"""Offline tests of G5 node-group and quota discovery."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_5_aws_enrich import enrich


class TestGPUDiscovery(unittest.TestCase):

    def setUp(self):
        self.config = {
            "schema_version": 1,
            "aws": {
                "region": "us-east-2",
                "account_id": "123456789012",
                "cluster_name": "test-cluster",
            },
            "runtime": {
                "backend": "vllm",
                "pipeline_parallelism": 2,
                "tensor_parallelism": 1,
            },
            "deployment": {
                "capacity_type": "ON_DEMAND",
                "gpu_instance_type": "g5.2xlarge",
                "gpu_node_selector": {"pool": "gpu"},
            },
        }

    def api(
        self,
        *,
        groups=None,
        quota=32,
        account="123456789012",
    ):
        if groups is None:
            groups = {
                "gpu-workers": {
                    "status": "ACTIVE",
                    "capacityType": "ON_DEMAND",
                    "instanceTypes": ["g5.2xlarge"],
                    "scalingConfig": {
                        "desiredSize": 0,
                        "maxSize": 2,
                    },
                },
            }

        def call(command, region):
            self.assertEqual(region, "us-east-2")

            service, operation = command[:2]

            if (service, operation) == (
                "sts", "get-caller-identity"
            ):
                return {"Account": account}

            if (service, operation) == (
                "ec2", "describe-instance-types"
            ):
                return {
                    "InstanceTypes": [{
                        "InstanceType": "g5.2xlarge",
                        "VCpuInfo": {
                            "DefaultVCpus": 8
                        },
                        "GpuInfo": {
                            "Gpus": [{"Count": 1}]
                        },
                    }]
                }

            if (service, operation) == (
                "eks", "list-nodegroups"
            ):
                return {
                    "nodegroups": list(groups)
                }

            if (service, operation) == (
                "eks", "describe-nodegroup"
            ):
                return {
                    "nodegroup": groups[command[-1]]
                }

            if (service, operation) == (
                "service-quotas",
                "list-service-quotas",
            ):
                return {
                    "Quotas": [{
                        "QuotaName":
                            "Running On-Demand G and VT instances",
                        "QuotaCode": "TEST-CODE",
                        "Value": quota,
                    }]
                }

            raise AssertionError(command)

        return call

    def test_unique_gpu_group_selected(self):
        result = enrich(self.config, self.api())

        self.assertEqual(
            result["deployment"]["gpu_nodegroup"],
            "gpu-workers",
        )
        self.assertEqual(
            result["deployment"]["gpu_node_selector"],
            {
                "eks.amazonaws.com/nodegroup":
                    "gpu-workers"
            },
        )
        self.assertEqual(
            result["aws_discovery"][
                "required_vcpus_for_two_nodes"
            ],
            16,
        )

    def test_insufficient_quota_detected(self):
        result = enrich(
            self.config,
            self.api(quota=8),
        )

        self.assertEqual(
            result["aws_discovery"]["quota_status"],
            "BELOW_REQUIRED_TOTAL",
        )

    def test_multiple_groups_not_guessed(self):
        groups = {
            "gpu-a": {
                "status": "ACTIVE",
                "capacityType": "ON_DEMAND",
                "instanceTypes": ["g5.2xlarge"],
            },
            "gpu-b": {
                "status": "ACTIVE",
                "capacityType": "ON_DEMAND",
                "instanceTypes": ["g5.2xlarge"],
            },
        }

        result = enrich(
            self.config,
            self.api(groups=groups),
        )

        self.assertIsNone(
            result["deployment"]["gpu_nodegroup"]
        )
        self.assertIsNone(
            result["deployment"]["gpu_node_selector"]
        )

    def test_explicit_group_selection(self):
        groups = {
            "gpu-a": {
                "status": "ACTIVE",
                "capacityType": "ON_DEMAND",
                "instanceTypes": ["g5.2xlarge"],
            },
            "gpu-b": {
                "status": "ACTIVE",
                "capacityType": "ON_DEMAND",
                "instanceTypes": ["g5.2xlarge"],
            },
        }

        result = enrich(
            self.config,
            self.api(groups=groups),
            requested_nodegroup="gpu-b",
        )

        self.assertEqual(
            result["deployment"]["gpu_nodegroup"],
            "gpu-b",
        )

    def test_wrong_account_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "account does not match"
        ):
            enrich(
                self.config,
                self.api(account="999999999999"),
            )

    def test_existing_config_not_mutated(self):
        original = deepcopy(self.config)

        enrich(self.config, self.api())

        self.assertEqual(self.config, original)


if __name__ == "__main__":
    unittest.main()
