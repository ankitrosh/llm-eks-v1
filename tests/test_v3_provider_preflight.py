import copy
import unittest

from scripts.v3_provider_preflight import validate


class ProviderPreflightTests(unittest.TestCase):

    def setUp(self):
        self.config = {
            "provider": {
                "factory_path": "example.module:Factory",
                "kwargs": {"split_at": 20},
            },
            "pipeline": {
                "world_size": 2,
                "max_active_requests": 4,
            },
            "generation": {
                "prompts": ["Hello"],
                "max_new_tokens": 8,
            },
        }

    def test_valid_config(self):
        path, kwargs = validate(self.config)
        self.assertEqual(path, "example.module:Factory")
        self.assertEqual(kwargs["split_at"], 20)

    def test_invalid_factory_path(self):
        data = copy.deepcopy(self.config)
        data["provider"]["factory_path"] = "invalid"

        with self.assertRaises(ValueError):
            validate(data)

    def test_invalid_world_size(self):
        data = copy.deepcopy(self.config)
        data["pipeline"]["world_size"] = 3

        with self.assertRaises(ValueError):
            validate(data)

    def test_invalid_concurrency(self):
        data = copy.deepcopy(self.config)
        data["pipeline"]["max_active_requests"] = 0

        with self.assertRaises(ValueError):
            validate(data)

    def test_invalid_prompt(self):
        data = copy.deepcopy(self.config)
        data["generation"]["prompts"] = [""]

        with self.assertRaises(ValueError):
            validate(data)

    def test_invalid_token_count(self):
        data = copy.deepcopy(self.config)
        data["generation"]["max_new_tokens"] = -1

        with self.assertRaises(ValueError):
            validate(data)


if __name__ == "__main__":
    unittest.main()
