"""Two-node NCCL validation of an external V3 model adapter."""

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

from llm_runtime.v3_attention_plan import resolve_attention_plan
from llm_runtime.v3_attention_install import install_attention_extensions
from llm_runtime.v3_extension_contracts import StageInputs, StageSpec
from llm_runtime.v3_plugin_registry import load_plugin


def main():
    rank = int(os.environ["JOB_COMPLETION_INDEX"])
    assert rank in (0, 1)
    assert os.environ["WORLD_SIZE"] == "2"

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    assert dist.is_nccl_available()

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=300),
        device_id=device,
    )

    print(
        f"PASS: rank {rank} initialized NCCL on "
        f"{torch.cuda.get_device_name(0)}",
        flush=True,
    )

    manifest = yaml.safe_load(
        Path(
            "/workspace/v3_gpt2_extension_nccl.yaml"
        ).read_text()
    )

    assert manifest["runtime"]["device"] == "cuda"
    assert (
        manifest["runtime"]["distributed_backend"]
        == "nccl"
    )
    assert manifest["scheduler"]["mode"] == "sequential"

    plan = resolve_attention_plan(manifest)

    specs = [
        StageSpec(
            rank=s["rank"],
            start_layer=s["start_layer"],
            end_layer=s["end_layer"],
            first_stage=(i == 0),
            final_stage=(i == 1),
        )
        for i, s in enumerate(plan["stages"])
    ]

    assert len(specs) == 2
    assert specs[0].end_layer == specs[1].start_layer

    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=64,
        n_layer=4,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
    )

    # Both ranks independently reconstruct the same fixed
    # reference checkpoint. No model download is required.
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as checkpoint:
        GPT2LMHeadModel(config).eval().save_pretrained(
            checkpoint,
            safe_serialization=True,
        )

        adapter = load_plugin(
            "experiments.v3_gpt2_extension:GPT2Adapter",
            "model",
        )

        adapter.validate_partition(specs)

        stage = adapter.create_stage(
            specs[rank],
            checkpoint=checkpoint,
            device="cuda:0",
            dtype="float32",
        )

        installed = install_attention_extensions(
            adapter,
            stage,
            plan,
            config.to_dict(),
        )

        expected_layers = [1] if rank == 0 else [2]
        assert installed == expected_layers, installed

        print(
            f"PASS: rank {rank} loaded layers "
            f"[{specs[rank].start_layer}, "
            f"{specs[rank].end_layer}), "
            f"custom attention {installed}",
            flush=True,
        )

        cache = DynamicCache(config=config)

        reference = None

        if rank == 0:
            reference = GPT2LMHeadModel.from_pretrained(
                checkpoint,
                attn_implementation="eager",
            ).to(device).eval()

        prompt = torch.tensor(
            [[4, 12, 15, 31, 8, 26]],
            dtype=torch.long,
            device=device,
        )

        history = None
        next_token = None

        # One prefill, one prompt continuation, then
        # three generated tokens.
        for step in range(5):
            width = 5 if step == 0 else 1
            position = 0 if step == 0 else 4 + step

            if rank == 0:
                if step == 0:
                    current = prompt[:, :5]
                elif step == 1:
                    current = prompt[:, 5:6]
                else:
                    current = next_token

                history = (
                    current
                    if history is None
                    else torch.cat(
                        (history, current), dim=1
                    )
                )

                with torch.inference_mode():
                    output = adapter.forward_stage(
                        stage,
                        StageInputs(
                            payload=current,
                            positions=position,
                            cache_state=cache,
                        ),
                    )

                hidden = output.hidden_states.contiguous()
                assert hidden.shape == (1, width, 64)

                dist.send(hidden, dst=1)

                logits = torch.empty(
                    (1, width, 128),
                    device=device,
                    dtype=torch.float32,
                )

                dist.recv(logits, src=1)

                with torch.inference_mode():
                    expected = reference(
                        history,
                        use_cache=False,
                    ).logits

                torch.testing.assert_close(
                    logits,
                    expected if step == 0
                    else expected[:, -1:, :],
                    rtol=1e-4,
                    atol=1e-4,
                )

                next_token = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                print(
                    f"PASS: NCCL step {step}, "
                    f"position={position}, "
                    f"logit parity verified",
                    flush=True,
                )

            else:
                hidden = torch.empty(
                    (1, width, 64),
                    device=device,
                    dtype=torch.float32,
                )

                dist.recv(hidden, src=0)

                with torch.inference_mode():
                    result = adapter.forward_stage(
                        stage,
                        StageInputs(
                            payload=hidden,
                            positions=position,
                            cache_state=cache,
                        ),
                    )

                logits = result.logits.contiguous()
                assert logits.shape == (1, width, 128)

                dist.send(logits, dst=0)

        selected_layer = installed[0]
        wrapper = stage.blocks[str(selected_layer)].attn

        assert wrapper.calls == 5

        print(
            f"PASS: rank {rank} custom attention "
            f"executed on all five steps",
            flush=True,
        )

        dist.barrier(device_ids=[0])

        if rank == 0:
            print(
                "PASS: Distributed logits match "
                "unsplit reference",
                flush=True,
            )
            print(
                "PASS: Distributed cached generation",
                flush=True,
            )

        print(
            f"PASS: rank {rank} distributed "
            "extension validation",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
