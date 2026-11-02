import json
import unittest
from types import SimpleNamespace

from llm_runtime.model_metadata import from_transformers_config


class ModelMetadataTests(unittest.TestCase):

    def test_qwen3_config_object(self):
        config = SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=40,
            hidden_size=5120,
            num_attention_heads=40,
            num_key_value_heads=8,
            head_dim=128,
            torch_dtype="bfloat16",
        )

        meta = from_transformers_config(
            "Qwen/Qwen3-14B", config
        )

        self.assertEqual(
            (meta.num_layers, meta.head_dim), (40, 128)
        )
        self.assertEqual(meta.num_key_value_heads, 8)
        self.assertEqual(meta.dtype, "bfloat16")
        self.assertIsNone(meta.checkpoint_quantization)

        json.dumps(meta.to_dict())

    def test_other_family_config_dict(self):
        config = dict(
            model_type="llama",
            num_hidden_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
        )

        meta = from_transformers_config(
            "example/llama", config
        )

        self.assertEqual(meta.head_dim, 128)
        self.assertEqual(meta.num_key_value_heads, 32)

    def test_quantized_checkpoint_detection(self):
        config = dict(
            model_type="qwen3",
            num_hidden_layers=64,
            hidden_size=5120,
            num_attention_heads=64,
            head_dim=128,
            quantization_config={
                "quant_method": "gptq"
            },
        )

        meta = from_transformers_config(
            "example/32b", config
        )

        self.assertEqual(
            meta.checkpoint_quantization, "gptq"
        )
        self.assertEqual(meta.num_layers, 64)

    def test_missing_layers_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "num_hidden_layers"
        ):
            from_transformers_config(
                "invalid", dict(model_type="custom")
            )

    def test_ambiguous_head_dim_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "head_dim"
        ):
            from_transformers_config(
                "invalid",
                dict(
                    model_type="custom",
                    num_hidden_layers=2,
                    hidden_size=100,
                    num_attention_heads=3,
                ),
            )


if __name__ == "__main__":
    unittest.main()
