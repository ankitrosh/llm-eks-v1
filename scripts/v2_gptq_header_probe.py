"""Read safetensors headers remotely; validate Qwen3 GPTQ tensor layouts."""

import json
import struct
import urllib.request
from pathlib import Path

ROOT = Path("tests/fixtures/qwen3-32b-gptq")
REPO = "JunHowie/Qwen3-32B-GPTQ-Int8"
LAYERS = (0, 31, 32, 63)

cfg = json.loads((ROOT / "config.json").read_text())
index = json.loads(
    (ROOT / "model.safetensors.index.json").read_text()
)["weight_map"]

hidden = cfg["hidden_size"]
head_dim = cfg["head_dim"]
q_size = cfg["num_attention_heads"] * head_dim
kv_size = cfg["num_key_value_heads"] * head_dim
intermediate = cfg["intermediate_size"]

projections = {
    "self_attn.q_proj": (hidden, q_size),
    "self_attn.k_proj": (hidden, kv_size),
    "self_attn.v_proj": (hidden, kv_size),
    "self_attn.o_proj": (q_size, hidden),
    "mlp.gate_proj": (hidden, intermediate),
    "mlp.up_proj": (hidden, intermediate),
    "mlp.down_proj": (intermediate, hidden),
}


def read_range(shard, start, end):
    url = (
        f"https://huggingface.co/{REPO}/resolve/main/{shard}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "llm-eks-gptq-header-probe",
        },
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise RuntimeError(
                f"{shard}: expected HTTP 206, "
                f"got {response.status}; refusing full download"
            )

        expected_range = f"bytes {start}-{end}/"
        actual_range = response.headers.get("Content-Range", "")

        if not actual_range.startswith(expected_range):
            raise RuntimeError(
                f"{shard}: unexpected Content-Range: {actual_range}"
            )

        payload = response.read(end - start + 1)

    if len(payload) != end - start + 1:
        raise RuntimeError(f"{shard}: incomplete range response")

    return payload


def read_header(shard):
    first = read_range(shard, 0, 7)
    length = struct.unpack("<Q", first)[0]

    if not 0 < length <= 8 * 1024 * 1024:
        raise RuntimeError(
            f"{shard}: invalid or excessive header length {length}"
        )

    payload = read_range(shard, 8, length + 7)
    return json.loads(payload), length


names = [
    name
    for name in index
    if any(name.startswith(f"model.layers.{i}.") for i in LAYERS)
]

shards = sorted({index[name] for name in names})

print("========== REMOTE SAFETENSORS HEADERS ==========")
print("Required shards:", len(shards), flush=True)

headers = {}

for shard in shards:
    headers[shard], length = read_header(shard)
    print(
        f"PASS: {shard}, header={length:,} bytes",
        flush=True,
    )

print("\n========== VALIDATE REAL TENSOR SHAPES ==========")

validated = 0

for layer_id in LAYERS:
    for projection, (in_features, out_features) in projections.items():
        prefix = f"model.layers.{layer_id}.{projection}"

        assert in_features % 128 == 0
        assert in_features % 4 == 0
        assert out_features % 4 == 0

        groups = in_features // 128

        expected = {
            "qweight": ([in_features // 4, out_features], "I32"),
            "qzeros": ([groups, out_features // 4], "I32"),
            "scales": ([groups, out_features], None),
            "g_idx": ([in_features], "I32"),
        }

        for field, (shape, dtype) in expected.items():
            name = f"{prefix}.{field}"
            shard = index[name]
            info = headers[shard][name]

            if info["shape"] != shape:
                raise AssertionError(
                    f"{name}: actual shape {info['shape']}, "
                    f"expected {shape}"
                )

            if dtype is not None:
                assert info["dtype"] == dtype, (
                    name, info["dtype"], dtype
                )
            else:
                assert info["dtype"] in {"F16", "BF16"}, (
                    name, info["dtype"]
                )

            validated += 1

    print(
        f"PASS: Layer {layer_id} — "
        "all 7 quantized projections",
        flush=True,
    )

assert validated == len(LAYERS) * 7 * 4

print("\n========== RESULTS ==========")
print("Tensor metadata entries validated:", validated)
print("Checkpoint weights downloaded: 0")
print("PASS: GPTQ packed tensor layouts match expected shapes")
