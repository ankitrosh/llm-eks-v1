"""Offline tests for EKS CPU-only group discovery."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_5_aws_cpu import discover_cpu


def group(instance, **changes):
    value = {
        "status": "ACTIVE",
        "capacityType": "ON_DEMAND",
        "instanceTypes": [instance],
        "scalingConfig": {
            "desiredSize": 0,
            "maxSize": 1,
        },
        "labels": {"pool": "cpu"},
    }
    value.update(changes)
    return value


class TestCPUDiscovery(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema_version": 1,
            "aws": {
                "account_id": "123456789012",
                "region": "us-east-2",
                "cluster_name": "test-cluster",
            },
            "deployment": {},
            "aws_discovery": {"quota_value": 8},
        }

    def api(self, groups=None, account="123456789012"):
        if groups is None:
            groups = {
                "cpu-system": group("t3.large"),
                "gpu-a10g": group(
                    "g5.xlarge",
                    labels={"pool": "gpu"},
                ),
            }

        types = {
            "t3.large": (2, 8192, False),
            "m5.large": (2, 8192, False),
            "t3.nano": (2, 512, False),
            "g5.xlarge": (4, 16384, True),
        }

        def call(command, region):
            self.assertEqual(region, "us-east-2")
            service, action = command[:2]

            if (service, action) == (
                "sts", "get-caller-identity"
            ):
                return {"Account": account}

            if (service, action) == (
                "eks", "list-nodegroups"
            ):
                return {"nodegroups": list(groups)}

            if (service, action) == (
                "eks", "describe-nodegroup"
            ):
                return {
                    "nodegroup": groups[command[-1]]
                }

            if (service, action) == (
                "ec2", "describe-instance-types"
            ):
                result = []
                for name in command[3:]:
                    cpu, ram, gpu = types[name]
                    item = {
                        "InstanceType": name,
                        "VCpuInfo": {"DefaultVCpus": cpu},
                        "MemoryInfo": {"SizeInMiB": ram},
                    }
                    if gpu:
                        item["GpuInfo"] = {
                            "Gpus": [{"Count": 1}]
                        }
                    result.append(item)
                return {"InstanceTypes": result}

            raise AssertionError(command)

        return call

    def test_unique_cpu_group(self):
        result = discover_cpu(
            self.config, self.api()
        )
        self.assertEqual(
            result["deployment"]["cpu_nodegroup"],
            "cpu-system",
        )
        self.assertEqual(
            result["deployment"]["cpu_node_selector"],
            {
                "eks.amazonaws.com/nodegroup":
                    "cpu-system"
            },
        )

    def test_gpu_group_excluded(self):
        result = discover_cpu(
            self.config, self.api()
        )
        self.assertEqual(
            result["aws_discovery"][
                "cpu_nodegroup_candidates"
            ],
            ["cpu-system"],
        )

    def test_multiple_candidates_not_guessed(self):
        groups = {
            "cpu-a": group("t3.large"),
            "cpu-b": group("m5.large"),
        }
        result = discover_cpu(
            self.config, self.api(groups)
        )
        self.assertIsNone(
            result["deployment"]["cpu_nodegroup"]
        )

    def test_explicit_cpu_selection(self):
        groups = {
            "cpu-a": group("t3.large"),
            "cpu-b": group("m5.large"),
        }
        result = discover_cpu(
            self.config,
            self.api(groups),
            requested="cpu-b",
        )
        self.assertEqual(
            result["deployment"]["cpu_nodegroup"],
            "cpu-b",
        )

    def test_ineligible_explicit_group_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "not an eligible"
        ):
            discover_cpu(
                self.config, self.api(),
                requested="gpu-a10g",
            )

    def test_underpowered_group_excluded(self):
        groups = {
            "tiny": group("t3.nano"),
        }
        result = discover_cpu(
            self.config, self.api(groups)
        )
        self.assertIsNone(
            result["deployment"]["cpu_nodegroup"]
        )

    def test_inactive_group_excluded(self):
        groups = {
            "inactive": group(
                "t3.large", status="CREATING"
            )
        }
        result = discover_cpu(
            self.config, self.api(groups)
        )
        self.assertIsNone(
            result["deployment"]["cpu_nodegroup"]
        )

    def test_wrong_account_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "account mismatch"
        ):
            discover_cpu(
                self.config,
                self.api(account="999999999999"),
            )

    def test_no_mutation(self):
        original = deepcopy(self.config)
        discover_cpu(self.config, self.api())
        self.assertEqual(self.config, original)


if __name__ == "__main__":
    unittest.main()
