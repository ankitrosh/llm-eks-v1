import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llm_runtime.v3_plan_loader import (
    PlanLoadError,
    configure_serving_from_plan,
    digest,
    load_runtime_plan,
)
from llm_runtime.v3_runtime_plan import (
    compile_runtime_plan,
)


CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 40,
    "tie_word_embeddings": False,
}


class PlanLoaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.config_path = self.root / "config.json"
        self.plan_path = self.root / "plan.json"

        self.config_path.write_text(json.dumps(CONFIG))

        self.plan = compile_runtime_plan({
            "model": {
                "id": "Qwen/Qwen3-14B",
                "num_layers": 40,
            },
            "pipeline": {
                "num_ranks": 2,
                "partition": "manual",
                "stages": [
                    {
                        "rank": 0,
                        "start_layer": 0,
                        "end_layer": 23,
                    },
                    {
                        "rank": 1,
                        "start_layer": 23,
                        "end_layer": 40,
                    },
                ],
            },
        }, CONFIG)

        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        self.plan_path.write_text(
            json.dumps(self.plan)
        )

    def resign(self):
        unsigned = {
            k: v for k, v in self.plan.items()
            if k != "plan_sha256"
        }
        self.plan["plan_sha256"] = digest(unsigned)
        self.save()

    def test_rank_zero(self):
        plan = load_runtime_plan(
            self.plan_path, self.root,
            rank=0, world_size=2,
        )
        self.assertEqual(plan["pipeline"]["split"], 23)

    def test_rank_one(self):
        plan = load_runtime_plan(
            self.plan_path, self.root,
            rank=1, world_size=2,
        )
        self.assertEqual(
            plan["pipeline"]["stages"][1]["start_layer"],
            23,
        )

    def test_three_rank_runtime_rejected(self):
        with self.assertRaises(PlanLoadError):
            load_runtime_plan(
                self.plan_path, self.root,
                rank=0, world_size=3,
            )

    def test_plan_tampering_rejected(self):
        self.plan["pipeline"]["split"] = 20
        self.save()

        with self.assertRaises(PlanLoadError):
            load_runtime_plan(
                self.plan_path, self.root, rank=0
            )

    def test_checkpoint_mismatch_rejected(self):
        altered = copy.deepcopy(CONFIG)
        altered["num_hidden_layers"] = 41

        self.config_path.write_text(
            json.dumps(altered)
        )

        with self.assertRaises(PlanLoadError):
            load_runtime_plan(
                self.plan_path, self.root, rank=0
            )

    def test_unknown_env_rejected(self):
        self.plan["runtime_environment"][
            "UNEXPECTED_SETTING"
        ] = "unexpected"

        self.resign()

        with self.assertRaises(PlanLoadError):
            load_runtime_plan(
                self.plan_path, self.root, rank=0
            )

    def test_apply_config(self):
        args = SimpleNamespace(
            model_dir=str(self.root),
            rank=0,
            split=20,
        )

        env = {
            "LLM_RUNTIME_PLAN": str(self.plan_path),
            "WORLD_SIZE": "2",
        }

        result = configure_serving_from_plan(
            args, env
        )

        self.assertEqual(args.split, 23)
        self.assertEqual(
            env["LLM_STAGE_IMPL"], "v3_adapter"
        )
        self.assertIsNotNone(result)

    def test_legacy_path_unchanged(self):
        args = SimpleNamespace(
            model_dir=str(self.root),
            rank=0,
            split=20,
        )

        result = configure_serving_from_plan(
            args, {}
        )

        self.assertIsNone(result)
        self.assertEqual(args.split, 20)


if __name__ == "__main__":
    unittest.main()
