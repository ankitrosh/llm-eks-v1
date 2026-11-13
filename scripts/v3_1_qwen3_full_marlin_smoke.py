"""Full Qwen3 GPTQ INT8 Marlin smoke on one A10G."""

import argparse
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from llm_runtime.v3_1_quantization_api import QuantizationContext
from llm_runtime.providers.v3_1_gptq_marlin_provider import (
    GPTQMarlinProvider,
)
from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)


def invoke(stages, payload, caches, step, position):
    for rank, stage in enumerate(stages):
        result = stage(
            SimpleNamespace(
                step=step,
                position=position,
                payload=payload,
            ),
            caches[rank],
        )
        payload = result.payload
        caches[rank] = result.cache_state

    assert payload.ndim == 3 and payload.shape[0] == 1
    assert torch.isfinite(payload).all(), "Non-finite logits"
    return payload.detach().float().cpu()


def exercise(stages):
    ids = torch.tensor([[4, 7, 9]], dtype=torch.long)
    caches = [None] * len(stages)

    torch.cuda.synchronize()
    start = time.perf_counter()
    prefill = invoke(stages, ids, caches, 0, 0)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000

    token = 11
    start = time.perf_counter()
    cached = invoke(
        stages, token, caches, 1, ids.shape[1]
    )
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) * 1000

    # Compare cached decode to a fresh, full prefill.
    complete_ids = torch.tensor(
        [[4, 7, 9, token]], dtype=torch.long
    )
    full = invoke(
        stages, complete_ids, [None] * len(stages),
        0, 0,
    )

    torch.testing.assert_close(
        cached, full, rtol=0.03, atol=0.03
    )

    return prefill, cached, prefill_ms, decode_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"

    device = "cuda:0"
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    config = Qwen3Config.from_pretrained(
        args.checkpoint_dir
    )
    layers = config.num_hidden_layers

    assert config.tie_word_embeddings, (
        "Expected tied embeddings checkpoint"
    )

    print(
        "GPU:", torch.cuda.get_device_name(0),
        "Layers:", layers,
        flush=True,
    )

    context = QuantizationContext(
        checkpoint_dir=str(
            Path(args.checkpoint_dir).resolve()
        ),
        device=device,
        activation_dtype="float16",
        options={
            "bits": 8,
            "execution_backend": "marlin",
        },
    )

    provider = GPTQMarlinProvider(context)
    profile = provider.inspect_checkpoint(context)
    provider.validate_runtime(context, profile)

    def load(boundaries):
        factory = Qwen3NRankStageOnlyFactory(
            checkpoint_dir=args.checkpoint_dir,
            layer_boundaries=boundaries,
            device=device,
            dtype="float16",
            quantization_provider=provider,
        )

        stages = [
            factory(rank)
            for rank in range(len(boundaries) - 1)
        ]

        count = sum(
            s.model.quantized_linear_count
            for s in stages
        )

        assert count == 7 * layers, (
            count, 7 * layers
        )
        return stages

    # V3.1 intentionally supports 2..N pipeline ranks.
    # Compare two and three stages on the same GPU.
    torch.cuda.reset_peak_memory_stats()

    print(
        "========== TWO STAGES ON ONE GPU ==========",
        flush=True,
    )

    stages = load((0, layers // 2, layers))

    two_prefill, two_cached, prefill_ms, decode_ms = (
        exercise(stages)
    )

    print(f"Two-stage prefill: {prefill_ms:.2f} ms")
    print(f"Two-stage decode: {decode_ms:.2f} ms")
    print(
        "Two-stage peak GPU allocation:",
        round(
            torch.cuda.max_memory_allocated() / 2**20,
            2,
        ),
        "MiB",
        flush=True,
    )

    del stages
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "========== THREE STAGES ON ONE GPU ==========",
        flush=True,
    )

    boundaries = (
        0,
        layers // 3,
        (2 * layers) // 3,
        layers,
    )

    stages = load(boundaries)

    three_prefill, three_cached, prefill_ms, decode_ms = (
        exercise(stages)
    )

    torch.testing.assert_close(
        three_prefill,
        two_prefill,
        rtol=0.01,
        atol=0.01,
    )

    torch.testing.assert_close(
        three_cached,
        two_cached,
        rtol=0.01,
        atol=0.01,
    )

    print(f"Three-stage prefill: {prefill_ms:.2f} ms")
    print(f"Three-stage decode: {decode_ms:.2f} ms")
    print(
        "Three-stage peak GPU allocation:",
        round(
            torch.cuda.max_memory_allocated() / 2**20,
            2,
        ),
        "MiB",
        flush=True,
    )

    print(
        "PASS: Real Qwen3 GPTQ INT8 Marlin full-model "
        "prefill, cached decode and 2-to-3-stage parity",
        flush=True,
    )


if __name__ == "__main__":
    main()
