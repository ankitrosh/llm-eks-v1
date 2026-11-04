"""Validate Qwen3-32B GPTQ layer construction without GPU or weights."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from qwen3_stage_loader import Qwen3Stage


PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class LoaderHarness:
    # Exercise the actual implementation from our stage loader.
    _install_gptq_linears = Qwen3Stage._install_gptq_linears

    def __init__(self, weight_map):
        self.index = weight_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    args = parser.parse_args()

    root = Path(args.metadata_dir)

    cfg = AutoConfig.from_pretrained(str(root))
    index = json.loads(
        (root / "model.safetensors.index.json").read_text()
    )["weight_map"]

    assert cfg.model_type == "qwen3"
    assert cfg.num_hidden_layers == 64

    harness = LoaderHarness(index)

    from gptqmodel.nn_modules.qlinear.tritonv2 import (
        TritonV2QuantLinear,
    )

    checked = 0

    for layer_id in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{layer_id}."

        with torch.device("meta"):
            layer = Qwen3DecoderLayer(cfg, layer_id)
            harness._install_gptq_linears(
                layer,
                global_layer=layer_id,
            )

        expected = {
            name[len(prefix):]
            for name in index
            if name.startswith(prefix)
        }

        actual = set(layer.state_dict())

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        assert not missing, (
            f"Layer {layer_id}: missing keys {missing}"
        )
        assert not extra, (
            f"Layer {layer_id}: extra keys {extra}"
        )

        for name in PROJECTIONS:
            module = layer.get_submodule(name)

            assert isinstance(module, TritonV2QuantLinear)
            assert module.bits == 8
            assert module.group_size == 128
            assert module.qweight.dtype == torch.int32
            assert module.qzeros.dtype == torch.int32
            assert module.g_idx.dtype == torch.int32
            assert module.qweight.device.type == "meta"

            checked += 1

        if layer_id % 16 == 15:
            print(
                f"PASS: layers 0-{layer_id}, "
                f"{checked} quantized projections",
                flush=True,
            )

    assert checked == 448

    print("PASS: All 64 layers constructed")
    print("PASS: All 448 GPTQ projections verified")
    print("PASS: State-dictionary keys match checkpoint index")
    print("PASS: No GPU or checkpoint weights required")


if __name__ == "__main__":
    main()
