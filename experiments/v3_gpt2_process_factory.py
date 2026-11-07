"""Real GPT-2 stage executors for the V3 CPU process transport."""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import GPT2Config
from transformers.cache_utils import DynamicCache

from llm_runtime.v3_attention_plan import resolve_attention_plan
from llm_runtime.v3_attention_install import install_attention_extensions
from llm_runtime.v3_extension_contracts import StageInputs, StageSpec
from llm_runtime.v3_pipeline_scheduler import StageResult
from llm_runtime.v3_plugin_registry import load_plugin


class GPT2StageExecutor:
    def __init__(self, adapter, stage, config, rank, selected_layer, device):
        self.adapter = adapter
        self.stage = stage
        self.config = config
        self.rank = rank
        self.selected_layer = selected_layer
        self.device = torch.device(device)

    def __call__(self, task, cache):
        if cache is None:
            assert task.step == 0
            cache = DynamicCache(config=self.config)
        else:
            assert isinstance(cache, DynamicCache)
            assert task.step > 0

        payload = task.payload

        if self.rank == 0:
            if isinstance(payload, int):
                payload = torch.tensor(
                    [[payload]], dtype=torch.long,
                    device=self.device,
                )
            assert payload.ndim == 2
            payload = payload.to(self.device)
        else:
            assert isinstance(payload, torch.Tensor)
            assert payload.ndim == 3
            assert payload.device == self.device

        wrapper = self.stage.blocks[
            str(self.selected_layer)
        ].attn

        before = wrapper.calls

        with torch.inference_mode():
            result = self.adapter.forward_stage(
                self.stage,
                StageInputs(
                    payload=payload,
                    positions=task.position,
                    cache_state=cache,
                    request_id=str(task.request_id),
                ),
            )

        assert wrapper.calls == before + 1
        assert result.cache_state is cache

        output = (
            result.logits
            if result.logits is not None
            else result.hidden_states
        )

        assert isinstance(output, torch.Tensor)

        return StageResult(
            payload=output.contiguous(),
            cache_state=cache,
        )


@dataclass(frozen=True)
class GPT2ProcessFactory:
    checkpoint_dir: str
    manifest_path: str
    device: str = "cpu"
    dtype: str = "float32"

    def __call__(self, rank):
        # Executed inside the owning child process.
        torch.set_num_threads(1)

        manifest = yaml.safe_load(
            Path(self.manifest_path).read_text()
        )

        plan = resolve_attention_plan(manifest)
        config = GPT2Config.from_pretrained(
            self.checkpoint_dir
        )

        assert len(plan["stages"]) == 2
        assert config.n_layer == manifest["model"]["num_layers"]

        specs = [
            StageSpec(
                rank=s["rank"],
                start_layer=s["start_layer"],
                end_layer=s["end_layer"],
                first_stage=(i == 0),
                final_stage=(i == len(plan["stages"]) - 1),
            )
            for i, s in enumerate(plan["stages"])
        ]

        adapter = load_plugin(
            "experiments.v3_gpt2_extension:GPT2Adapter",
            "model",
        )

        adapter.validate_partition(specs)

        stage = adapter.create_stage(
            specs[rank],
            checkpoint=self.checkpoint_dir,
            device=self.device,
            dtype=self.dtype,
        )

        stage.eval()

        installed = install_attention_extensions(
            adapter,
            stage,
            plan,
            config.to_dict(),
        )

        expected = [1] if rank == 0 else [2]
        assert installed == expected, (rank, installed)

        return GPT2StageExecutor(
            adapter,
            stage,
            config,
            rank,
            installed[0],
            self.device,
        )
