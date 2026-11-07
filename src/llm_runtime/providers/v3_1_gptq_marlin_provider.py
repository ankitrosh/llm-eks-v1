"""GPTQ INT8 Marlin execution provider for V3.1.

Reuses the GPTQModel 4.0.0 Marlin kernel ABI validated in V2.
Initial target: symmetric GPTQ v1, INT8, group_size=128,
desc_act=False, FP16/BF16, NVIDIA A10G (SM 8.6).

Offline inspection works without CUDA. Execution requires
the Marlin-enabled image and compiled CUDA extension.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from llm_runtime.v3_1_checkpoint_profile import (
    _checkpoint_tensor_sizes,
)
from llm_runtime.v3_1_gptq_int8_inspection import (
    inspect_qwen3_gptq_int8,
)
from llm_runtime.v3_1_quantization_api import (
    QuantizationProfile,
    QuantizationProviderError,
)


def require(condition, message):
    if not condition:
        raise QuantizationProviderError(message)


class GPTQMarlinProvider:
    def __init__(self, context):
        self.context = context
        self.root = Path(context.checkpoint_dir).resolve()

        config = json.loads(
            (self.root / "config.json").read_text()
        )

        self.quant = config.get("quantization_config") or {}
        self.profile = inspect_qwen3_gptq_int8(self.root)

        require(
            self.profile.format_id == "gptq",
            "V3.1 Marlin adapter currently requires "
            "legacy GPTQ v1 checkpoint packing",
        )
        require(
            self.profile.bits == 8
            and self.profile.group_size == 128,
            "Marlin adapter requires INT8 group_size=128",
        )
        require(
            self.quant.get("sym") is True,
            "Marlin adapter requires sym=true",
        )
        require(
            self.quant.get("desc_act") is False,
            "Marlin adapter requires desc_act=false",
        )
        require(
            self.quant.get("pack_dtype", "int32")
            in ("int32", "torch.int32"),
            "Marlin adapter requires INT32 packing",
        )
        require(
            not self.quant.get("dynamic"),
            "Mixed/dynamic GPTQ requires another provider",
        )

        self.tensor_names = frozenset(
            _checkpoint_tensor_sizes(self.root)
        )

    def inspect_checkpoint(self, context):
        require(
            Path(context.checkpoint_dir).resolve() == self.root,
            "Quantization checkpoint path changed",
        )
        return self.profile

    def should_quantize(self, module_name):
        packed = module_name + ".qweight"
        dense = module_name + ".weight"

        has_packed = packed in self.tensor_names
        has_dense = dense in self.tensor_names

        require(
            has_packed != has_dense,
            f"Ambiguous or missing projection: {module_name}",
        )

        return has_packed

    def validate_runtime(self, context, profile):
        require(
            isinstance(profile, QuantizationProfile)
            and profile == self.profile,
            "Quantization profile mismatch",
        )

        device = torch.device(context.device)

        require(
            device.type == "cuda",
            "Marlin execution requires CUDA",
        )
        require(
            context.activation_dtype
            in ("float16", "bfloat16"),
            "Marlin requires FP16 or BF16 activations",
        )
        require(
            torch.cuda.is_available(),
            "CUDA device is unavailable",
        )
        require(
            torch.cuda.get_device_capability(device) == (8, 6),
            "This Marlin build targets SM 8.6 (A10G)",
        )

        try:
            installed_version = version("gptqmodel")
        except PackageNotFoundError as exc:
            raise QuantizationProviderError(
                "GPTQModel is not installed"
            ) from exc

        require(
            installed_version == "4.0.0",
            "Expected GPTQModel 4.0.0",
        )

        try:
            import gptqmodel_marlin_kernels  # noqa: F401
            from gptqmodel.nn_modules.qlinear.marlin import (
                MarlinQuantLinear,
            )
        except ImportError as exc:
            raise QuantizationProviderError(
                "Marlin extension or module unavailable"
            ) from exc

        self.linear_class = MarlinQuantLinear

    def build_linear(
        self,
        *,
        module_name,
        in_features,
        out_features,
        bias,
        reader,
        device,
        dtype,
    ):
        require(
            hasattr(self, "linear_class"),
            "Marlin runtime was not validated",
        )
        require(
            torch.device(device).type == "cuda"
            and dtype in (torch.float16, torch.bfloat16),
            "Marlin requires CUDA FP16/BF16",
        )
        require(
            in_features % 32 == 0
            and out_features % 32 == 0,
            "Marlin projection dimensions must be "
            "divisible by 32",
        )

        names = ("qweight", "qzeros", "scales", "g_idx")

        weights = {
            name: reader.get(f"{module_name}.{name}")
            for name in names
        }

        require(
            weights["qweight"].dtype == torch.int32
            and weights["qzeros"].dtype == torch.int32
            and weights["g_idx"].dtype == torch.int32,
            "Invalid packed GPTQ integer dtype",
        )
        require(
            weights["scales"].is_floating_point(),
            "Invalid GPTQ scales dtype",
        )
        require(
            tuple(weights["qweight"].shape)
            == (in_features // 4, out_features),
            "Invalid GPTQ qweight shape",
        )
        require(
            tuple(weights["qzeros"].shape)
            == (
                (in_features + 127) // 128,
                out_features // 4,
            ),
            "Invalid GPTQ qzeros shape",
        )
        require(
            tuple(weights["scales"].shape)
            == (
                (in_features + 127) // 128,
                out_features,
            ),
            "Invalid GPTQ scales shape",
        )
        require(
            tuple(weights["g_idx"].shape)
            == (in_features,),
            "Invalid GPTQ g_idx shape",
        )
        require(
            torch.equal(
                weights["g_idx"],
                torch.arange(
                    in_features,
                    dtype=torch.int32,
                ) // 128,
            ),
            "Unexpected activation-order group indices",
        )
        require(
            bool(torch.isfinite(
                weights["scales"].float()
            ).all())
            and bool((weights["scales"] > 0).all()),
            "Invalid GPTQ scales",
        )

        if bias:
            weights["bias"] = reader.get(
                module_name + ".bias"
            )

        linear = self.linear_class(
            bits=8,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=torch.int32,
        )

        # Preserve exact packed tensor dtypes and layout.
        # Matches the proven V2 Marlin loading sequence.
        linear.load_state_dict(
            weights,
            strict=True,
            assign=True,
        )

        linear = linear.to(device).eval()
        linear.post_init()

        return linear


def create_provider(context):
    return GPTQMarlinProvider(context)
