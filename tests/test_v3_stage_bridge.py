import json
import tempfile
import unittest
from pathlib import Path

from llm_runtime.v3_stage_bridge import (
    load_qwen3_stage_v3,
)


class FakeQwen3Stage:
    def __init__(self, model_dir, rank, split=20):
        self.rank = rank
        self.start = 0 if rank == 0 else split
        self.stop = split if rank == 0 else 40
        self.model_dir = model_dir

    def load(self):
        return self

    def make_cache(self):
        return {"rank": self.rank}

    def forward_local(self, data, start_position, cache):
        return data


class StageBridgeTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.config = {
            "model_type": "qwen3",
            "num_hidden_layers": 40,
            "tie_word_embeddings": False,
        }

        self.write_config()

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self):
        (self.root / "config.json").write_text(
            json.dumps(self.config)
        )

    def build(self, rank, split):
        return load_qwen3_stage_v3(
            self.root,
            rank,
            split,
            stage_factory=FakeQwen3Stage,
        )

    def test_equal_partition(self):
        first = self.build(0, 20)
        last = self.build(1, 20)

        self.assertEqual((first.start, first.stop), (0, 20))
        self.assertEqual((last.start, last.stop), (20, 40))

    def test_unequal_partition(self):
        first = self.build(0, 23)
        last = self.build(1, 23)

        self.assertEqual((first.start, first.stop), (0, 23))
        self.assertEqual((last.start, last.stop), (23, 40))

    def test_returns_legacy_compatible_stage(self):
        stage = self.build(0, 20)

        self.assertTrue(callable(stage.make_cache))
        self.assertTrue(callable(stage.forward_local))

        self.assertEqual(
            stage.forward_local("input", 0, stage.make_cache()),
            "input",
        )

    def test_wrong_architecture_rejected(self):
        self.config["model_type"] = "gpt2"
        self.write_config()

        with self.assertRaises(ValueError):
            self.build(0, 20)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            self.build(0, 40)

    def test_invalid_rank_rejected(self):
        with self.assertRaises(ValueError):
            self.build(2, 20)

    def test_tied_embeddings_rejected(self):
        self.config["tie_word_embeddings"] = True
        self.write_config()

        with self.assertRaises(ValueError):
            self.build(0, 20)


if __name__ == "__main__":
    unittest.main()
