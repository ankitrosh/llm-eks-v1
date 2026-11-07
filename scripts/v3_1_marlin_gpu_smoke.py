"""Single-GPU GPTQ INT8 Marlin correctness smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from llm_runtime.v3_1_quantization_api import QuantizationContext
from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    SelectiveTensorReader,
)
from llm_runtime.providers.v3_1_gptq_marlin_provider import (
    GPTQMarlinProvider,
)


PROJECTION = "model.layers.0.self_attn.k_proj"


def independently_dequantize(weights):
    """Decode legacy GPTQ v1 weights without GPTQModel kernels."""

    qweight = weights["qweight"].to(torch.int64)
    qzeros = weights["qzeros"].to(torch.int64)
    scales = weights["scales"].float()
    g_idx = weights["g_idx"].long()

    input_shifts = (
        torch.arange(4, dtype=torch.int64)
        .view(1, 4, 1) * 8
    )

    codes = (
        (qweight.unsqueeze(1) >> input_shifts) & 255
    ).reshape(-1, qweight.shape[1])

    output_shifts = (
        torch.arange(4, dtype=torch.int64)
        .view(1, 1, 4) * 8
    )

    stored_zeros = (
        (qzeros.unsqueeze(-1) >> output_shifts) & 255
    ).reshape(qzeros.shape[0], -1)

    # Legacy GPTQ v1 stores zero_point - 1.
    zeros = (stored_zeros + 1) & 255

    decoded = (
        (codes.float() - zeros[g_idx].float())
        * scales[g_idx]
    )

    return decoded.T.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"

    device = "cuda:0"
    torch.cuda.set_device(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    context = QuantizationContext(
        checkpoint_dir=str(
            Path(args.checkpoint_dir).resolve()
        ),
        device=device,
        activation_dtype="float16",
    )

    provider = GPTQMarlinProvider(context)
    profile = provider.inspect_checkpoint(context)

    provider.validate_runtime(context, profile)

    reader = SelectiveTensorReader(
        args.checkpoint_dir
    )

    weights = {
        field: reader.get(f"{PROJECTION}.{field}")
        for field in (
            "qweight",
            "qzeros",
            "scales",
            "g_idx",
        )
    }

    in_features = weights["g_idx"].numel()
    out_features = weights["qweight"].shape[1]

    print(
        "Projection:", PROJECTION,
        "Shape:", in_features, "x", out_features,
    )

    # Independent reconstruction before Marlin post_init
    # repacks the GPU module's internal tensors.
    reference_weight = independently_dequantize(
        weights
    )

    torch.cuda.reset_peak_memory_stats()

    linear = provider.build_linear(
        module_name=PROJECTION,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        reader=reader,
        device=device,
        dtype=torch.float16,
    )

    print("Module:", type(linear).__name__)

    for token_count in (1, 16, 64):
        torch.manual_seed(1234)

        x = torch.randn(
            token_count,
            in_features,
            device=device,
            dtype=torch.float16,
        )

        with torch.inference_mode():
            expected = F.linear(
                x.float().cpu(),
                reference_weight,
            ).to(device)

            # Trigger compilation/initialization, then
            # compare actual packed-kernel output.
            for _ in range(3):
                actual = linear(x)

            torch.cuda.synchronize()

        actual_float = actual.float()

        assert actual_float.shape == expected.shape
        assert bool(torch.isfinite(actual_float).all())

        difference = actual_float - expected

        relative_rmse = (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected).clamp_min(1e-8)
        ).item()

        cosine = F.cosine_similarity(
            actual_float.flatten().unsqueeze(0),
            expected.flatten().unsqueeze(0),
        ).item()

        print(
            f"Tokens={token_count} "
            f"relative_RMSE={relative_rmse:.6f} "
            f"cosine={cosine:.8f}",
            flush=True,
        )

        assert relative_rmse < 0.03
        assert cosine > 0.999

    print(
        "Peak GPU memory (MiB):",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**2,
            2,
        ),
    )

    print("PASS: Marlin agrees with independent GPTQ decoder")


if __name__ == "__main__":
    main()
