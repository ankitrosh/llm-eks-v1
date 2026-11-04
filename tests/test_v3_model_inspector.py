import unittest

from llm_runtime.v3_model_inspector import (
    InspectionError,
    inspect_config,
)


QWEN = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 40,
    "hidden_size": 5120,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "quantization_config": {
        "quant_method": "gptq",
        "bits": 8,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
    },
}

GPT2 = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "n_layer": 6,
    "n_embd": 768,
    "n_head": 12,
}


class InspectorTests(unittest.TestCase):

    def test_qwen3_known_checkpoint(self):
        result = inspect_config(
            "JunHowie/Qwen3-14B-GPTQ-Int8",
            QWEN,
            expected_layers=40,
        )
        self.assertEqual(
            result["checkpoint_status"],
            "PREVIOUSLY_TESTED_CHECKPOINT_FAMILY",
        )

    def test_gpt2_adapter_planned(self):
        result = inspect_config(
            "distilbert/distilgpt2",
            GPT2,
            expected_layers=6,
        )
        self.assertEqual(
            result["adapter_status"],
            "PLANNED_V3_2",
        )

    def test_layer_mismatch_rejected(self):
        with self.assertRaises(InspectionError):
            inspect_config(
                "distilbert/distilgpt2",
                GPT2,
                expected_layers=40,
            )

    def test_unsupported_architecture(self):
        result = inspect_config(
            "example/custom-llama",
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 12,
            },
        )
        self.assertEqual(
            result["checkpoint_status"],
            "ADAPTER_REQUIRED",
        )

    def test_wrong_architecture_variant(self):
        config = {
            **GPT2,
            "architectures": [
                "GPT2ForSequenceClassification"
            ],
        }
        result = inspect_config(
            "example/gpt2-classifier", config
        )
        self.assertEqual(
            result["checkpoint_status"],
            "ARCHITECTURE_VARIANT_UNSUPPORTED",
        )

    def test_other_quantization_not_verified(self):
        config = {
            **QWEN,
            "quantization_config": {
                **QWEN["quantization_config"],
                "bits": 4,
            },
        }
        result = inspect_config(
            "JunHowie/Qwen3-14B-GPTQ-Int8",
            config,
        )
        self.assertEqual(
            result["quantization"]["layout_status"],
            "NOT_VERIFIED",
        )


if __name__ == "__main__":
    unittest.main()
