"""Offline AWS discovery tests; no real AWS calls."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v3_5_aws_configure.py"

spec = importlib.util.spec_from_file_location(
    "v35_aws_configure", SCRIPT
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestAWSDiscovery(unittest.TestCase):

    def setUp(self):
        self.template = {
            "schema_version": 1,
            "aws": {
                "region": "auto",
                "account_id": "auto",
                "cluster_name": None,
            },
            "model": {"pvc_rank0": None},
        }

    def api(
        self, clusters,
        status="ACTIVE",
        account="123456789012",
    ):
        def call(command, region):
            self.assertEqual(region, "us-west-2")

            if command[:2] == [
                "sts", "get-caller-identity"
            ]:
                return {"Account": account}

            if command[:2] == [
                "eks", "list-clusters"
            ]:
                return {"clusters": clusters}

            if command[:2] == [
                "eks", "describe-cluster"
            ]:
                return {
                    "cluster": {"status": status}
                }

            raise AssertionError(
                f"Unexpected AWS command: {command}"
            )

        return call

    def test_unique_cluster_selected(self):
        out, count = module.discover(
            self.template,
            "us-west-2",
            None,
            self.api(["alpha"]),
        )

        self.assertEqual(count, 1)
        self.assertEqual(
            out["aws"]["cluster_name"], "alpha"
        )
        self.assertEqual(
            out["aws"]["account_id"], "123456789012"
        )
        self.assertEqual(
            out["aws"]["region"], "us-west-2"
        )

        # Original configuration remains unchanged.
        self.assertEqual(
            self.template["aws"]["account_id"], "auto"
        )

    def test_multiple_clusters_not_guessed(self):
        out, count = module.discover(
            self.template,
            "us-west-2",
            None,
            self.api(["beta", "alpha"]),
        )

        self.assertEqual(count, 2)
        self.assertIsNone(
            out["aws"]["cluster_name"]
        )

    def test_explicit_cluster_selected(self):
        out, _ = module.discover(
            self.template,
            "us-west-2",
            "beta",
            self.api(["beta", "alpha"]),
        )

        self.assertEqual(
            out["aws"]["cluster_name"], "beta"
        )

    def test_unknown_cluster_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "not available"
        ):
            module.discover(
                self.template,
                "us-west-2",
                "missing",
                self.api(["alpha"]),
            )

    def test_inactive_cluster_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "not ACTIVE"
        ):
            module.discover(
                self.template,
                "us-west-2",
                None,
                self.api(
                    ["alpha"], status="CREATING"
                ),
            )

    def test_incorrect_account_rejected(self):
        self.template["aws"]["account_id"] = (
            "999999999999"
        )

        with self.assertRaisesRegex(
            ValueError, "differs"
        ):
            module.discover(
                self.template,
                "us-west-2",
                None,
                self.api(["alpha"]),
            )


if __name__ == "__main__":
    unittest.main()
