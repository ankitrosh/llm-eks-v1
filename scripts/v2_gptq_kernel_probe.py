"""Test GPTQModel TritonV2 INT8 arithmetic on one CUDA GPU."""

import torch

from gptqmodel.nn_modules.qlinear.tritonv2 import (
    TRITON_AVAILABLE,
    TritonV2QuantLinear,
)

def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    assert TRITON_AVAILABLE, "GPTQ Triton backend unavailable"

    torch.manual_seed(42)

    device = "cuda"
    features = 256

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Triton backend available:", TRITON_AVAILABLE)

    layer = TritonV2QuantLinear(
        bits=8,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=features,
        out_features=features,
        bias=False,
        pack_dtype=torch.int32,
    ).to(device).eval()

    # Create synthetic 8-bit weights and pack four values
    # into each int32, matching GPTQ's packed layout.
    values = torch.randint(
        96, 160, (features, features),
        device=device,
        dtype=torch.int32,
    )

    packed = (
        values[0::4]
        | (values[1::4] << 8)
        | (values[2::4] << 16)
        | (values[3::4] << 24)
    )

    with torch.no_grad():
        layer.qweight.copy_(packed)
        layer.qzeros.fill_(0x7F7F7F7F)
        layer.scales.fill_(0.02)

        assert layer.g_idx.device.type == "cuda"

        # Independent PyTorch reference calculation.
        # GPTQ stores quantized weights as [in, out].
        group = layer.g_idx.long()

        weights = (
            (values.float() - 127.0)
            * layer.scales[group].float()
        ).to(torch.bfloat16)

        for batch in (1, 8, 32):
            x = torch.randn(
                batch, features,
                device=device,
                dtype=torch.bfloat16,
            ) * 0.5

            actual = layer(x)
            expected = x @ weights

            torch.cuda.synchronize()

            torch.testing.assert_close(
                actual, expected,
                rtol=0.03,
                atol=0.03,
            )

            difference = (
                actual.float() - expected.float()
            ).abs().max().item()

            print(
                f"PASS: batch={batch} "
                f"shape={tuple(actual.shape)} "
                f"max_abs_error={difference:.6f}"
            )

    print(
        "Peak allocated GPU memory:",
        round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "MiB",
    )
    print("PASS: GPTQ INT8 Triton GPU correctness")

if __name__ == "__main__":
    main()
