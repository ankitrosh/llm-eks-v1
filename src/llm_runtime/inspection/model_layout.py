"""Build checkpoint-backed logical model metadata. No GPU needed."""

import hashlib
import json
from pathlib import Path


QWEN3_WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_layout(config, model_id, weight_index=None):
    architectures = config.get("architectures", [])

    if architectures != ["Qwen3ForCausalLM"]:
        raise ValueError(
            "V3.3 layout inspector currently supports only "
            f"Qwen3ForCausalLM; got {architectures!r}"
        )

    if config.get("use_sliding_window", False):
        raise ValueError(
            "Sliding-window Qwen3 requires a layout adapter"
        )

    n_layers = int(config["num_hidden_layers"])
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config["num_key_value_heads"])
    hidden = int(config["hidden_size"])
    head_dim = int(
        config.get("head_dim") or (hidden // n_heads)
    )

    if min(
        n_layers, n_heads, n_kv_heads, hidden, head_dim
    ) <= 0:
        raise ValueError("Invalid model dimensions")

    if n_heads % n_kv_heads:
        raise ValueError(
            "Attention heads not divisible by KV heads"
        )

    weight_map = None

    if weight_index is not None:
        weight_map = weight_index.get("weight_map")

        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Invalid safetensors weight index")

        required = [
            "model.embed_tokens.weight",
            "model.norm.weight",
        ]

        if not config.get("tie_word_embeddings", False):
            required.append("lm_head.weight")

        for i in range(n_layers):
            required.extend(
                f"model.layers.{i}.{suffix}"
                for suffix in QWEN3_WEIGHT_SUFFIXES
            )

        missing = [
            key for key in required
            if key not in weight_map
        ]

        if missing:
            raise ValueError(
                f"Checkpoint missing {len(missing)} "
                f"expected weights: {missing[:5]}"
            )

    layers = []

    for i in range(n_layers):
        base = f"model.layers.{i}"

        entry = {
            "index": i,
            "module_path": base,
            "attention": {
                "module_path": f"{base}.self_attn",
                "kind": "causal_grouped_query_attention",
                "q_heads": n_heads,
                "kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "projections": {
                    name: f"{base}.self_attn.{name}_proj"
                    for name in ("q", "k", "v", "o")
                },
                "q_norm": f"{base}.self_attn.q_norm",
                "k_norm": f"{base}.self_attn.k_norm",
            },
            "mlp": {
                "module_path": f"{base}.mlp",
                "kind": "dense_swiglu",
                "projections": {
                    name: f"{base}.mlp.{name}_proj"
                    for name in ("gate", "up", "down")
                },
            },
            "norms": [
                f"{base}.input_layernorm",
                f"{base}.post_attention_layernorm",
            ],
        }

        if weight_map is not None:
            keys = sorted(
                key for key in weight_map
                if key.startswith(base + ".")
            )

            entry["checkpoint"] = {
                "weight_keys": keys,
                "shards": sorted({
                    weight_map[key] for key in keys
                }),
            }

        layers.append(entry)

    layout = {
        "schema_version": 1,
        "kind": "logical_checkpoint_layout",
        "model_id": model_id,
        "architecture": "Qwen3ForCausalLM",
        "verification": (
            "checkpoint_weight_index"
            if weight_map is not None
            else "config_inferred_unverified"
        ),
        "note": (
            "Logical checkpoint paths, not a traced "
            "forward graph or fused vLLM module layout."
        ),
        "embedding": "model.embed_tokens",
        "transformer_container": "model.layers",
        "num_layers": n_layers,
        "layers": layers,
        "final_norm": "model.norm",
        "output_head": "lm_head",
        "tied_embeddings": bool(
            config.get("tie_word_embeddings", False)
        ),
    }

    if weight_map is not None:
        layout["checkpoint"] = {
            "tensors": len(weight_map),
            "shards": sorted(set(weight_map.values())),
            "total_size_bytes": weight_index.get(
                "metadata", {}
            ).get("total_size"),
        }

    capabilities = {
        "schema_version": 1,
        "kind": "model_capabilities",
        "model_id": model_id,
        "architecture": "Qwen3ForCausalLM",
        "model_type": config.get("model_type"),
        "num_layers": n_layers,
        "hidden_size": hidden,
        "intermediate_size": config.get(
            "intermediate_size"
        ),
        "vocab_size": config.get("vocab_size"),
        "q_heads": n_heads,
        "kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "attention_kind": "causal_grouped_query_attention",
        "max_position_embeddings": config.get(
            "max_position_embeddings"
        ),
        "checkpoint_dtype": config.get(
            "dtype", config.get("torch_dtype")
        ),
        "quantization_config": config.get(
            "quantization_config"
        ),
        "tied_embeddings": bool(
            config.get("tie_word_embeddings", False)
        ),
    }

    return layout, capabilities
