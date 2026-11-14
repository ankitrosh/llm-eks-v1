import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from gptqmodel.nn_modules.qlinear.tritonv2 import (
    TritonV2QuantLinear,
)
from gptqmodel.nn_modules.qlinear.marlin import (
    MarlinQuantLinear,
)
from gptqmodel.utils.model import (
    convert_gptq_v1_to_v2_format_module,
)

ROOT = Path("/models/qwen3-14b-gptq")
INDEX = json.loads(
    (ROOT / "model.safetensors.index.json").read_text()
)["weight_map"]

PREFIXES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
]

SIZES = [1, 128, 512, 2048]
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

torch.manual_seed(42)
torch.set_num_threads(1)


def read_projection(prefix):
    names = {
        name: name[len(prefix) + 1:]
        for name in INDEX
        if name.startswith(prefix + ".")
    }

    required = {"qweight", "qzeros", "scales", "g_idx"}
    assert set(names.values()) == required, (
        prefix, set(names.values())
    )

    tensors = {}

    for shard in sorted({INDEX[n] for n in names}):
        with safe_open(
            str(ROOT / shard),
            framework="pt",
            device="cpu",
        ) as reader:
            for full_name, short_name in names.items():
                if INDEX[full_name] == shard:
                    tensors[short_name] = reader.get_tensor(
                        full_name
                    )

    return tensors


def make_linear(cls, weights):
    in_features = weights["g_idx"].numel()
    out_features = weights["qweight"].shape[1]

    assert weights["qweight"].shape[0] * 4 == in_features

    layer = cls(
        bits=8,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
    )

    layer.load_state_dict(
        {k: v.clone() for k, v in weights.items()},
        strict=True,
        assign=True,
    )

    layer = layer.to(DEVICE).eval()

    if isinstance(layer, TritonV2QuantLinear):
        convert_gptq_v1_to_v2_format_module(
            module=layer,
            bits=8,
            pack_dtype=torch.int32,
        )
    else:
        layer.post_init()

    return layer


@torch.inference_mode()
def benchmark(module, x, warmups=8, iterations=30):
    for _ in range(warmups):
        module(x)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    samples = []

    for _ in range(iterations):
        start.record()
        module(x)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    return statistics.median(samples)


def compare(a, b):
    a = a.float().flatten()
    b = b.float().flatten()

    error = a - b

    relative_rmse = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(a).clamp_min(1e-8)
    ).item()

    cosine = F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()

    return relative_rmse, cosine


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, flush=True)

    results = []

    for prefix in PREFIXES:
        print(f"\n========== {prefix} ==========", flush=True)

        weights = read_projection(prefix)

        triton = make_linear(
            TritonV2QuantLinear, weights
        )

        marlin = make_linear(
            MarlinQuantLinear, weights
        )

        print(
            "Shape:",
            weights["g_idx"].numel(),
            "x",
            weights["qweight"].shape[1],
            flush=True,
        )

        for m in SIZES:
            x = torch.randn(
                m,
                weights["g_idx"].numel(),
                device=DEVICE,
                dtype=DTYPE,
            )

            with torch.inference_mode():
                y_triton = triton(x)
                y_marlin = marlin(x)

            torch.cuda.synchronize()

            assert torch.isfinite(y_triton).all()
            assert torch.isfinite(y_marlin).all()

            rmse, cosine = compare(y_triton, y_marlin)

            print(
                f"M={m} correctness: "
                f"relative_RMSE={rmse:.6f} "
                f"cosine={cosine:.6f}",
                flush=True,
            )

            if rmse > 0.05 or cosine < 0.995:
                raise AssertionError(
                    f"Marlin numerical mismatch: {prefix}, M={m}"
                )

            t_triton = benchmark(triton, x)
            t_marlin = benchmark(marlin, x)

            speedup = t_triton / t_marlin

            print(
                f"M={m:4d} "
                f"Triton={t_triton:.3f}ms "
                f"Marlin={t_marlin:.3f}ms "
                f"Speedup={speedup:.2f}x",
                flush=True,
            )

            results.append({
                "projection": prefix,
                "M": m,
                "triton_ms": t_triton,
                "marlin_ms": t_marlin,
                "speedup": speedup,
                "relative_rmse": rmse,
                "cosine": cosine,
            })

            del x, y_triton, y_marlin

        del triton, marlin, weights
        torch.cuda.empty_cache()

    print("\n========== FINAL RESULTS ==========")
    print(json.dumps(results, indent=2), flush=True)
    print("\nPASS: Marlin correctness and kernel comparison")


if __name__ == "__main__":
    main()
