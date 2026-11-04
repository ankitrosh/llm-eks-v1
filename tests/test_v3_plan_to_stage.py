import json
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

from llm_runtime.v3_runtime_plan import (
    compile_runtime_plan,
)
from llm_runtime.v3_plan_loader import (
    configure_serving_from_plan,
)
from llm_runtime.v3_stage_bridge import (
    load_qwen3_stage_v3,
)


CHECKPOINT = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 40,
    "tie_word_embeddings": False,
}


class FakeStage:
    def __init__(self, model_dir, rank, split=20):
        self.rank = rank
        self.start = 0 if rank == 0 else split
        self.stop = split if rank == 0 else 40

    def load(self):
        return self

    def make_cache(self):
        return {}

    def forward_local(self, data, start_position, cache):
        return data


class PlanToStageTests(unittest.TestCase):

    def test_complete_two_rank_configuration(self):
        manifest = {
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
            "attention": {
                "default": {
                    "implementation": "model_native",
                    "backend": "sdpa",
                },
            },
        }

        plan = compile_runtime_plan(
            manifest, CHECKPOINT
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / "config.json").write_text(
                json.dumps(CHECKPOINT)
            )

            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))

            observed = []

            for rank in (0, 1):
                args = SimpleNamespace(
                    model_dir=str(root),
                    rank=rank,
                    split=20,
                )

                env = {
                    "WORLD_SIZE": "2",
                    "LLM_RUNTIME_PLAN": str(plan_path),
                }

                loaded = configure_serving_from_plan(
                    args, env
                )

                self.assertEqual(args.split, 23)
                self.assertEqual(
                    env["LLM_STAGE_IMPL"],
                    "v3_adapter",
                )
                self.assertEqual(
                    env["LLM_ATTENTION_BACKEND"],
                    "sdpa",
                )

                stage = load_qwen3_stage_v3(
                    root,
                    rank,
                    args.split,
                    stage_factory=FakeStage,
                )

                observed.append(
                    (stage.rank, stage.start, stage.stop)
                )

                self.assertEqual(
                    loaded["plan_sha256"],
                    plan["plan_sha256"],
                )

            self.assertEqual(
                observed,
                [(0, 0, 23), (1, 23, 40)],
            )


if __name__ == "__main__":
    unittest.main()
