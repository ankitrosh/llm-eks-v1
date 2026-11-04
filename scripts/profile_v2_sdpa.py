"""GPU SDPA mask comparison using the pinned Transformers wrapper."""

import math
from types import SimpleNamespace

import torch
from torch.nn.attention.bias import causal_lower_right
from torch.profiler import ProfilerActivity, profile
from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward,
)


def benchmark(cached, new):
    device = "cuda"
    dtype = torch.bfloat16
    total = cached + new

    # Representative Qwen3-14B GQA dimensions.
    heads, kv_heads, head_dim = 40, 8, 128

    q = torch.randn(
        1, heads, new, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        1, kv_heads, total, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        1, kv_heads, total, head_dim, device=device, dtype=dtype
    )

    module = SimpleNamespace(
        num_key_value_groups=heads // kv_heads,
        is_causal=True,
    )

    if cached == 0:
        masks = {"normal_prefill": None}
    else:
        positions = torch.arange(cached, total, device=device)
        key_positions = torch.arange(total, device=device)

        visible = key_positions[None, :] <= positions[:, None]

        explicit = torch.zeros(
            (1, 1, new, total), dtype=dtype, device=device
        )
        explicit.masked_fill_(
            ~visible[None, None, :, :],
            torch.finfo(dtype).min,
        )

        masks = {
            "explicit": explicit,
            "lower_right": causal_lower_right(new, total),
        }

    outputs = {}

    for name, mask in masks.items():
        def run():
            result, _ = sdpa_attention_forward(
                module, q, k, v, mask,
                dropout=0.0,
                scaling=1 / math.sqrt(head_dim),
            )
            return result

        for _ in range(10):
            run()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        elapsed = []

        for _ in range(30):
            start.record()
            output = run()
            stop.record()
            torch.cuda.synchronize()
            elapsed.append(start.elapsed_time(stop))

        peak_extra = max(
            0,
            torch.cuda.max_memory_allocated() - baseline_memory,
        )

        outputs[name] = output.detach().clone()

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ]
        ) as prof:
            run()
            torch.cuda.synchronize()

        kernels = sorted({
            event.key
            for event in prof.key_averages()
            if (
                "scaled_dot_product" in event.key.lower()
                or "flash" in event.key.lower()
                or "efficient_attention" in event.key.lower()
            )
        })

        median_ms = sorted(elapsed)[len(elapsed) // 2]

        print(
            f"cached={cached} new={new} mode={name} "
            f"median_ms={median_ms:.4f} "
            f"peak_extra_mib={peak_extra / 2**20:.2f}",
            flush=True,
        )
        print("KERNELS:", kernels, flush=True)

    if cached:
        expected = outputs["explicit"].float()
        actual = outputs["lower_right"].float()

        max_diff = (expected - actual).abs().max().item()
        rmse = (
            (expected - actual).square().mean().sqrt().item()
        )

        print(
            f"CORRECTNESS cached={cached} new={new} "
            f"max_diff={max_diff:.6f} rmse={rmse:.6f}",
            flush=True,
        )

        torch.testing.assert_close(
            actual, expected, rtol=0.03, atol=0.03
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU required")

    torch.manual_seed(42)
    print("Torch:", torch.__version__, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    for cached, new in [
        (0, 512),
        (64, 8),
        (512, 32),
        (1792, 256),
    ]:
        benchmark(cached, new)

    print("PASS: GPU SDPA profiling and correctness", flush=True)
