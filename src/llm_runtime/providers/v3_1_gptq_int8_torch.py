"""Slow CPU reference for INT32-packed GPTQ INT8 linear layers.

Supports the conventional gptq and gptq_v2 packing formats.
Used to validate correctness before choosing optimized GPU kernels.

Does NOT perform GPTQ quantization or calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    StageCheckpointError,
)

MAX_REFERENCE_ELEMENTS = 2_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpack_input_codes(words, in_features):
    # Each INT32 contains four 8-bit input-row codes.
    shifts = torch.tensor(
        [0, 8, 16, 24], device=words.device
    )
    values = (
        words.to(torch.int64).unsqueeze(1)
        >> shifts[None, :, None]
    ) & 255
    return values.reshape(-1, words.shape[1])[:in_features]


def unpack_output_zeros(words, out_features):
    # Each INT32 contains four 8-bit output-column zeros.
    shifts = torch.tensor(
        [0, 8, 16, 24], device=words.device
    )
    values = (
        words.to(torch.int64).unsqueeze(-1)
        >> shifts
    ) & 255
    return values.reshape(words.shape[0], -1)[:, :out_features]


class GPTQInt8TorchLinear(nn.Module):
    def __init__(
        self, *,
        qweight,
        qzeros,
        scales,
        g_idx,
        bias,
        in_features,
        out_features,
        checkpoint_format,
    ):
        super().__init__()

        require(
            checkpoint_format in ("gptq", "gptq_v2"),
            "Unsupported GPTQ format",
        )
        require(
            type(in_features) is int and in_features > 0
            and type(out_features) is int and out_features > 0,
            "Invalid linear dimensions",
        )
        require(
            in_features * out_features <= MAX_REFERENCE_ELEMENTS,
            "Layer too large for CPU reference execution",
        )
        require(
            qweight.dtype == torch.int32
            and qweight.ndim == 2
            and tuple(qweight.shape) == (
                math.ceil(in_features / 4), out_features
            ),
            "Invalid qweight format or dimensions",
        )
        require(
            scales.ndim == 2 and scales.is_floating_point(),
            "Invalid scales tensor",
        )

        groups = scales.shape[0]

        require(
            groups > 0 and scales.shape[1] == out_features,
            "Invalid scales dimensions",
        )
        require(
            qzeros.dtype == torch.int32
            and tuple(qzeros.shape) == (
                groups, math.ceil(out_features / 4)
            ),
            "Invalid qzeros dimensions",
        )
        require(
            g_idx.dtype in (torch.int32, torch.int64)
            and tuple(g_idx.shape) == (in_features,)
            and bool(
                ((g_idx >= 0) & (g_idx < groups)).all()
            ),
            "Invalid group indices",
        )
        require(
            bool(torch.isfinite(scales.float()).all())
            and bool((scales > 0).all()),
            "Invalid quantization scales",
        )
        require(
            bias is None or (
                tuple(bias.shape) == (out_features,)
                and bias.is_floating_point()
            ),
            "Invalid bias",
        )

        self.in_features = in_features
        self.out_features = out_features
        self.checkpoint_format = checkpoint_format

        self.register_buffer(
            "qweight", qweight.detach().contiguous().clone()
        )
        self.register_buffer(
            "qzeros", qzeros.detach().contiguous().clone()
        )
        self.register_buffer(
            "scales", scales.detach().float().contiguous().clone()
        )
        self.register_buffer(
            "g_idx", g_idx.detach().long().contiguous().clone()
        )
        self.register_buffer(
            "bias",
            None if bias is None
            else bias.detach().float().contiguous().clone(),
        )

    def dequantize_weight(self):
        codes = unpack_input_codes(
            self.qweight, self.in_features
        )
        zeros = unpack_output_zeros(
            self.qzeros, self.out_features
        )

        if self.checkpoint_format == "gptq":
            # Legacy GPTQ stores zero point minus one.
            zeros = (zeros + 1) & 255

        groups = self.g_idx

        weight = (
            codes.float() - zeros[groups].float()
        ) * self.scales[groups]

        return weight.T.contiguous()

    def forward(self, hidden):
        if (
            hidden.device.type != "cpu"
            or hidden.dtype != torch.float32
        ):
            raise RuntimeError(
                "GPTQ reference requires CPU float32 activations"
            )

        return F.linear(
            hidden,
            self.dequantize_weight(),
            self.bias,
        )


class GPTQInt8TorchReferenceProvider:
    """Compatible with the V3.1 build_linear adapter contract."""

    def __init__(self, context):
        self.context = context

        config_path = (
            Path(context.checkpoint_dir) / "config.json"
        )
        config = json.loads(config_path.read_text())
        quant = config.get("quantization_config") or {}

        require(
            quant.get("quant_method") == "gptq",
            "Not a GPTQ checkpoint",
        )
        require(
            type(quant.get("bits")) is int
            and quant["bits"] == 8,
            "Only GPTQ INT8 is supported",
        )

        self.checkpoint_format = quant.get(
            "format", "gptq"
        )
        self.group_size = quant.get("group_size", 128)

        require(
            self.checkpoint_format in ("gptq", "gptq_v2"),
            "Unsupported GPTQ packing format",
        )
        require(
            type(self.group_size) is int
            and (
                self.group_size == -1
                or self.group_size > 0
            ),
            "Invalid GPTQ group size",
        )

    def inspect_checkpoint(self, context):
        from llm_runtime.v3_1_gptq_int8_inspection import (
            inspect_qwen3_gptq_int8,
        )
        return inspect_qwen3_gptq_int8(
            context.checkpoint_dir
        )

    def validate_runtime(self, context, profile):
        require(
            context.device == "cpu"
            and context.activation_dtype == "float32",
            "CPU reference requires float32 CPU execution",
        )
        require(
            profile.bits == 8
            and profile.format_id == self.checkpoint_format,
            "GPTQ checkpoint format mismatch",
        )

    def build_linear(
        self, *,
        module_name,
        in_features,
        out_features,
        bias,
        reader,
        device,
        dtype,
    ):
        require(
            torch.device(device).type == "cpu"
            and dtype == torch.float32,
            "CPU reference requires float32 CPU execution",
        )

        qweight = reader.get(module_name + ".qweight")
        qzeros = reader.get(module_name + ".qzeros")
        scales = reader.get(module_name + ".scales")

        try:
            g_idx = reader.get(module_name + ".g_idx")
        except StageCheckpointError as exc:
            if not str(exc).startswith(
                "Missing checkpoint tensor:"
            ):
                raise

            size = (
                in_features
                if self.group_size == -1
                else self.group_size
            )
            g_idx = (
                torch.arange(in_features, dtype=torch.int32)
                // size
            )

        bias_value = (
            reader.get(module_name + ".bias")
            if bias else None
        )

        return GPTQInt8TorchLinear(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bias=bias_value,
            in_features=in_features,
            out_features=out_features,
            checkpoint_format=self.checkpoint_format,
        )


def create_provider(context):
    return GPTQInt8TorchReferenceProvider(context)
