import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F

from llm_runtime.providers.v3_1_gptq_int8_torch import (
    GPTQInt8TorchLinear,
)


def make_packed(fmt):
    k, n, groups = 12, 8, 3

    codes = (
        torch.arange(k * n).reshape(k, n) * 7 + 17
    ) % 256

    zeros = (
        torch.arange(groups * n).reshape(groups, n)
        * 13 + 1
    ) % 256

    scales = (
        torch.arange(
            groups * n, dtype=torch.float32
        ).reshape(groups, n) / 100 + 0.02
    )

    g_idx = torch.tensor(
        [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        dtype=torch.int32,
    )

    def pack_rows(values):
        pieces = values.reshape(-1, 4, n)

        return sum(
            pieces[:, j, :].to(torch.int64)
            << (j * 8)
            for j in range(4)
        ).to(torch.int32)

    def pack_cols(values):
        pieces = values.reshape(groups, -1, 4)

        return sum(
            pieces[:, :, j].to(torch.int64)
            << (j * 8)
            for j in range(4)
        ).to(torch.int32)

    stored_zeros = (
        (zeros - 1) % 256
        if fmt == "gptq"
        else zeros
    )

    packed = {
        "qweight": pack_rows(codes),
        "qzeros": pack_cols(stored_zeros),
        "scales": scales,
        "g_idx": g_idx,
        "bias": torch.ones(n),
        "in_features": k,
        "out_features": n,
        "checkpoint_format": fmt,
    }

    expected_weight = (
        (codes.float() - zeros[g_idx].float())
        * scales[g_idx]
    ).T.contiguous()

    return packed, expected_weight


@pytest.mark.parametrize("fmt", ["gptq", "gptq_v2"])
def test_int8_packed_dequantization(fmt):
    packed, expected = make_packed(fmt)

    layer = GPTQInt8TorchLinear(**packed)

    torch.testing.assert_close(
        layer.dequantize_weight(),
        expected,
    )


@pytest.mark.parametrize("fmt", ["gptq", "gptq_v2"])
def test_int8_linear_forward(fmt):
    packed, expected = make_packed(fmt)

    layer = GPTQInt8TorchLinear(**packed)

    hidden = torch.randn(2, 3, 12)

    torch.testing.assert_close(
        layer(hidden),
        F.linear(hidden, expected, torch.ones(8)),
    )


def test_bad_packed_weight_shape_rejected():
    packed, _ = make_packed("gptq_v2")
    packed["qweight"] = packed["qweight"][:1]

    with pytest.raises(ValueError, match="qweight"):
        GPTQInt8TorchLinear(**packed)


def test_invalid_group_indices_rejected():
    packed, _ = make_packed("gptq_v2")
    packed["g_idx"][0] = 99

    with pytest.raises(ValueError, match="group"):
        GPTQInt8TorchLinear(**packed)


def test_gpu_execution_not_silently_enabled():
    packed, _ = make_packed("gptq_v2")
    layer = GPTQInt8TorchLinear(**packed)

    with pytest.raises(
        RuntimeError,
        match="CPU float32",
    ):
        layer(torch.ones(1, 12, dtype=torch.float16))
