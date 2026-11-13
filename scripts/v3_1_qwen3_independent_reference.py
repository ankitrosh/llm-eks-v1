"""Independent GPTQ Triton vs V3.1 Marlin full-model validation."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from llm_runtime.providers.v3_1_configured_stage_factory import (
    create_v31_qwen3_stage_factory,
)

PROMPTS = (
    [4, 7, 9],
    [21, 44, 15, 81, 7],
    [7, 3, 12, 19, 11, 5, 6],
)
DECODE_TOKENS = (11, 17, 23)
ATOL = 0.03
RTOL = 0.03


def reference_logits(checkpoint):
    from gptqmodel import BACKEND, GPTQModel

    backend = getattr(BACKEND, "TRITON", None)

    if backend is None:
        raise RuntimeError(
            "Pinned GPTQModel lacks TRITON. "
            f"Available: {list(BACKEND.__members__)}"
        )

    print(
        f"Loading independent reference: {backend}",
        flush=True,
    )

    wrapper = GPTQModel.load(
        str(checkpoint),
        device="cuda:0",
        backend=backend,
        dtype=torch.float16,
        trust_remote_code=False,
    )

    model = getattr(wrapper, "model", wrapper)

    if not callable(model):
        raise TypeError("Reference model is not callable")

    model.eval()

    # Ensure the independent loader actually selected
    # TritonV2 quantized projections, not Marlin or dense.
    from gptqmodel.nn_modules.qlinear.tritonv2 import (
        TritonV2QuantLinear,
    )

    triton_count = sum(
        isinstance(module, TritonV2QuantLinear)
        for module in model.modules()
    )

    expected_count = 7 * 28

    assert triton_count == expected_count, (
        "Independent reference must use TritonV2 "
        f"for all projections: {triton_count}/"
        f"{expected_count}"
    )

    print(
        f"PASS: Independent TritonV2 "
        f"projections={triton_count}",
        flush=True,
    )

    results = {}

    with torch.inference_mode():
        for i, prompt in enumerate(PROMPTS):
            ids = torch.tensor(
                [prompt],
                dtype=torch.long,
                device="cuda:0",
            )

            output = model(input_ids=ids, use_cache=True)

            if output.past_key_values is None:
                raise AssertionError(
                    "Reference did not return KV cache"
                )

            results[(i, 0)] = (
                output.logits[:, -1:, :].float().cpu()
            )

            cache = output.past_key_values

            for step, token in enumerate(
                DECODE_TOKENS, 1
            ):
                next_id = torch.tensor(
                    [[token]],
                    dtype=torch.long,
                    device="cuda:0",
                )

                output = model(
                    input_ids=next_id,
                    past_key_values=cache,
                    use_cache=True,
                )

                cache = output.past_key_values

                results[(i, step)] = (
                    output.logits[:, -1:, :]
                    .float().cpu()
                )

            print(
                f"Reference prompt {i}: "
                f"{len(prompt)} prefill tokens, "
                "3 cached steps",
                flush=True,
            )

    del model, wrapper
    gc.collect()
    torch.cuda.empty_cache()

    return results


def stage_logits(checkpoint, layers):
    left = layers // 2

    config = {
        "model": {
            "checkpoint_dir": str(checkpoint),
            "quantization": {
                "provider": "gptq",
                "options": {"bits": 8},
            },
        },
        "pipeline": {
            "num_ranks": 2,
            "partition": "manual",
            "layer_counts": [left, layers - left],
            "memory": {
                "gpu_memory_gib": [20.0, 20.0],
                "reserve_gib": 1.0,
                "max_active_requests": 1,
                "max_total_tokens_per_request": 128,
            },
        },
        "communication": {
            "backend": "gloo_staged",
            "codec": "identity",
        },
    }

    # Exercise the real public configuration path.
    # No execution_backend setting is supplied.
    factory = create_v31_qwen3_stage_factory(
        config,
        device="cuda:0",
        dtype="float16",
    )

    stages = [factory(i) for i in range(2)]

    projections = sum(
        stage.model.quantized_linear_count
        for stage in stages
    )

    assert projections == 7 * layers, (
        projections, 7 * layers
    )

    results = {}

    with torch.inference_mode():
        for i, prompt in enumerate(PROMPTS):
            caches = [None, None]

            inputs = [
                (
                    0,
                    0,
                    torch.tensor(
                        [prompt], dtype=torch.long
                    ),
                )
            ]

            inputs += [
                (
                    step,
                    len(prompt) + step - 1,
                    token,
                )
                for step, token in enumerate(
                    DECODE_TOKENS, 1
                )
            ]

            for step, position, payload in inputs:
                for rank, stage in enumerate(stages):
                    result = stage(
                        SimpleNamespace(
                            step=step,
                            position=position,
                            payload=payload,
                        ),
                        caches[rank],
                    )

                    caches[rank] = result.cache_state
                    payload = result.payload

                assert payload.ndim == 3
                assert payload.shape[1] == 1

                results[(i, step)] = (
                    payload.detach().float().cpu()
                )

            print(
                f"Marlin prompt {i}: "
                f"{len(prompt)} prefill tokens, "
                "3 cached steps",
                flush=True,
            )

    del stages, factory
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )
    args = parser.parse_args()

    checkpoint = Path(
        args.checkpoint_dir
    ).resolve()

    assert checkpoint.is_dir(), checkpoint
    assert torch.cuda.is_available(), "CUDA required"

    torch.cuda.set_device(0)
    torch.set_num_threads(1)

    assert torch.cuda.get_device_capability(0) == (
        8, 6
    ), "Expected NVIDIA A10G"

    config = Qwen3Config.from_pretrained(
        str(checkpoint)
    )

    assert config.tie_word_embeddings
    assert config.num_hidden_layers == 28

    print(
        "========== INDEPENDENT GPTQ REFERENCE ==========",
        flush=True,
    )
    expected = reference_logits(checkpoint)

    print(
        "========== V3.1 MARLIN EXECUTION ==========",
        flush=True,
    )
    actual = stage_logits(
        checkpoint, config.num_hidden_layers
    )

    assert set(expected) == set(actual)

    max_abs = 0.0

    for key in sorted(expected):
        ref = expected[key]
        got = actual[key]

        assert ref.shape == got.shape
        assert torch.isfinite(ref).all()
        assert torch.isfinite(got).all()

        diff = (got - ref).abs()
        error = diff.max().item()
        max_abs = max(max_abs, error)

        scaled_relative = (
            diff / ref.abs().clamp_min(0.1)
        ).max().item()

        top_match = torch.equal(
            ref.argmax(-1),
            got.argmax(-1),
        )

        print(
            json.dumps({
                "prompt": key[0],
                "step": key[1],
                "max_abs": round(error, 6),
                "max_scaled_rel": round(
                    scaled_relative, 6
                ),
                "top1_agrees": top_match,
            }),
            flush=True,
        )

        if not top_match:
            raise AssertionError(
                f"Top-token mismatch: {key}"
            )

        torch.testing.assert_close(
            got,
            ref,
            rtol=RTOL,
            atol=ATOL,
        )

    print(
        "PASS: Independent GPTQ Triton versus "
        "V3.1 Marlin full-model reference; "
        f"max_abs={max_abs:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
