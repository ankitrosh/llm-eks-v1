import sys
import unittest

from llm_runtime.v3_plugin_registry import (
    PluginError,
    collect_attention_plugins,
    inspect_loaded_plugin,
    load_plugin,
    validate_reference,
)


class PluginRegistryTests(unittest.TestCase):

    def test_reference_syntax(self):
        self.assertEqual(
            validate_reference(
                "experiments.my_attention:MyAttention"
            ),
            "experiments.my_attention:MyAttention"
        )

    def test_invalid_reference_rejected(self):
        with self.assertRaises(PluginError):
            validate_reference(
                "some-module;import os:Bad"
            )

    def test_static_discovery_does_not_import(self):
        name = "nonexistent_example_plugin_xyz"

        plan = {
            "stages": [{
                "rank": 1,
                "attention": [{
                    "start_layer": 2,
                    "end_layer": 4,
                    "implementation": "custom",
                    "plugin": name + ".attention:Custom",
                }],
            }]
        }

        result = collect_attention_plugins(plan)

        self.assertEqual(len(result), 1)
        self.assertNotIn(name, sys.modules)
        self.assertEqual(
            result[0]["status"],
            "DECLARED_NOT_LOADED"
        )

    def test_demo_plugin_contract(self):
        result = inspect_loaded_plugin(
            "experiments.my_attention:MyAttention",
            "attention",
        )

        self.assertEqual(
            result["status"],
            "INTERFACE_CHECK_PASSED"
        )

    def test_demo_plugin_wrapper(self):
        extension = load_plugin(
            "experiments.my_attention:MyAttention",
            "attention",
        )

        def native(values):
            return [2 * value for value in values]

        module = extension.create(
            global_layer_idx=2,
            native_module=native,
            model_config={},
            options={},
        )

        self.assertEqual(
            module([1.0, 2.0, 3.0]),
            [2.0, 4.0, 6.0],
        )

    def test_missing_plugin_rejected(self):
        with self.assertRaises(PluginError):
            load_plugin(
                "missing_v3_package_xyz:Unknown",
                "attention",
            )


if __name__ == "__main__":
    unittest.main()
