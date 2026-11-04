import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_runtime.runtime_discovery import inspect_runtime


def fake_torch(cuda: bool):
    return SimpleNamespace(
        version=SimpleNamespace(cuda="13.0"),
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                scaled_dot_product_attention=lambda *a, **k: None
            )
        ),
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_capability=lambda i: (8, 6),
            get_device_name=lambda i: "NVIDIA A10G",
        ),
    )


class RuntimeDiscoveryTests(unittest.TestCase):

    def test_no_torch_installed(self):
        with patch(
            "llm_runtime.runtime_discovery._package_version",
            return_value=None,
        ):
            report = inspect_runtime()

        self.assertFalse(report.torch_import_ok)
        self.assertEqual(report.backend_candidates, ())
        self.assertIsNone(report.compute_capability)

    def test_cpu_environment(self):
        packages = {
            "torch": "2.14.1",
            "transformers": "4.56.2",
            "flash-attn": "2.8.0",
        }

        with patch(
            "llm_runtime.runtime_discovery._package_version",
            side_effect=packages.get,
        ), patch(
            "llm_runtime.runtime_discovery.importlib.import_module",
            return_value=fake_torch(False),
        ):
            report = inspect_runtime()

        self.assertEqual(
            report.device_type, "cpu_or_unavailable"
        )
        self.assertEqual(
            report.backend_candidates, ("sdpa",)
        )
        self.assertFalse(report.flash_attn_importable)

    def test_cuda_with_importable_flash_attention(self):
        packages = {
            "torch": "2.14.1",
            "flash-attn": "2.8.0",
        }

        def importer(name):
            if name == "torch":
                return fake_torch(True)

            if name == "flash_attn.flash_attn_interface":
                return SimpleNamespace(
                    flash_attn_func=lambda: None
                )

            raise ImportError(name)

        with patch(
            "llm_runtime.runtime_discovery._package_version",
            side_effect=packages.get,
        ), patch(
            "llm_runtime.runtime_discovery.importlib.import_module",
            side_effect=importer,
        ):
            report = inspect_runtime()

        self.assertEqual(
            report.compute_capability, (8, 6)
        )
        self.assertEqual(report.gpu_name, "NVIDIA A10G")
        self.assertEqual(
            report.backend_candidates,
            ("sdpa", "flash_attention_2"),
        )

    def test_installed_but_binary_import_fails(self):
        packages = {
            "torch": "2.14.1",
            "flash-attn": "2.8.0",
        }

        def importer(name):
            if name == "torch":
                return fake_torch(True)
            raise ImportError("incompatible extension")

        with patch(
            "llm_runtime.runtime_discovery._package_version",
            side_effect=packages.get,
        ), patch(
            "llm_runtime.runtime_discovery.importlib.import_module",
            side_effect=importer,
        ):
            report = inspect_runtime()

        self.assertEqual(
            report.backend_candidates, ("sdpa",)
        )
        self.assertFalse(report.flash_attn_importable)
        self.assertTrue(
            any(
                "FlashAttention import failed" in note
                for note in report.notes
            )
        )


if __name__ == "__main__":
    unittest.main()
