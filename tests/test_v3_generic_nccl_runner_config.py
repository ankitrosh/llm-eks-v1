import json
import tempfile
import unittest
from pathlib import Path

from scripts.v3_generic_nccl_runner import (
    load_config,
    prompt_count,
)


class GenericRunnerConfigTests(unittest.TestCase):

    def setUp(self):
        self.data = {
            "provider": {
                "factory_path": "example.module:Factory",
                "kwargs": {},
            },
            "pipeline": {
                "world_size": 2,
                "max_active_requests": 4,
                "max_pending": 4,
            },
            "generation": {
                "prompts": ["hello", "world"],
                "max_new_tokens": 8,
            },
            "tokenizer_path": "/models/tokenizer",
        }

    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data))
            return load_config(path)

    def test_text_prompts(self):
        self.assertEqual(
            prompt_count(self.load(self.data)),
            2,
        )

    def test_pretokenized_prompts(self):
        self.data["generation"] = {
            "token_ids": [[1], [1, 2, 3]]
        }
        del self.data["tokenizer_path"]

        self.assertEqual(
            prompt_count(self.load(self.data)),
            2,
        )

    def test_missing_input(self):
        del self.data["generation"]["prompts"]

        with self.assertRaises(ValueError):
            self.load(self.data)

    def test_double_input(self):
        self.data["generation"]["token_ids"] = [[1, 2]]

        with self.assertRaises(ValueError):
            self.load(self.data)

    def test_malformed_ids(self):
        del self.data["generation"]["prompts"]

        self.data["generation"]["token_ids"] = [
            [0, True]
        ]

        with self.assertRaises(ValueError):
            self.load(self.data)

    def test_missing_tokenizer(self):
        del self.data["tokenizer_path"]

        with self.assertRaises(ValueError):
            self.load(self.data)

    def test_invalid_selector(self):
        self.data["token_selector"] = "not_a_path"

        with self.assertRaises(ValueError):
            self.load(self.data)

    def test_bad_warmup_length(self):
        self.data["generation"]["warmup_new_tokens"] = 0

        with self.assertRaises(ValueError):
            self.load(self.data)


if __name__ == "__main__":
    unittest.main()
