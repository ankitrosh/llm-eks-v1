"""Qwen3 full-decoder blockwise decode correctness reference.

Supports full-attention Qwen3 eager models.
Prefill/cache initialization happens separately.
No fused CUDA kernels or sliding attention.
"""

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
)

from llm_runtime.v3_2_block_attention import (
    block_decode_attention,
)


def _validate_model(model, cache, request_ids, tokens):
    ids = tuple(request_ids)
    tokens = tuple(tokens)

    if (
        not ids
        or len(ids) != len(tokens)
        or any(type(rid) is not int or rid < 1 for rid in ids)
        or len(ids) != len(set(ids))
        or any(type(token) is not int or token < 0
               for token in tokens)
    ):
        raise ValueError("Invalid request IDs or tokens")

    config = model.config

    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Only Qwen3 models are supported")

    if model.training:
        raise ValueError("Block decode requires eval mode")

    if any(
        kind != "full_attention"
        for kind in config.layer_types
    ):
        raise ValueError(
            "Sliding/hybrid attention is not supported"
        )

    if (
        len(model.model.layers) != cache.layers
        or config.num_key_value_heads != cache.kv_heads
        or config.head_dim != cache.head_dim
    ):
        raise ValueError(
            "Model/cache layer or head geometry mismatch"
        )

    if cache._pending is not None:
        raise RuntimeError("KV reservation already pending")

    positions = []

    for rid in ids:
        length = cache._committed.get(rid)

        if length is None or length < 1:
            raise ValueError(
                f"Request {rid} has no committed KV"
            )

        if length >= cache.manager.max_context_tokens:
            raise ValueError("Maximum context length reached")

        positions.append(length)

    return ids, tokens, tuple(positions)


@torch.inference_mode()
def qwen3_block_decode_step(
    model, cache, request_ids, tokens,
):
    """One batched full-model decode without DynamicCache.

    Reads historical K/V from blocks and commits newly
    generated K/V after the complete model forward.

    Returns float32 logits [B, vocab_size].
    """
    ids, tokens, lengths = _validate_model(
        model, cache, request_ids, tokens
    )

    device = next(model.parameters()).device

    if device != cache.device:
        raise ValueError(
            "Model and KV cache must share a device"
        )

    input_ids = torch.tensor(
        [[value] for value in tokens],
        dtype=torch.long,
        device=device,
    )

    position_ids = torch.tensor(
        [[length] for length in lengths],
        dtype=torch.long,
        device=device,
    )

    hidden = model.model.embed_tokens(input_ids)
    rope = model.model.rotary_emb(
        hidden, position_ids
    )

    new_kv = []

    for layer_index, layer in enumerate(
        model.model.layers
    ):
        residual = hidden
        x = layer.input_layernorm(hidden)
        attn = layer.self_attn

        batch, seq_len, _ = x.shape

        if seq_len != 1:
            raise RuntimeError(
                "Only single-token decode is supported"
            )

        shape = (batch, seq_len, -1, attn.head_dim)

        q = attn.q_norm(
            attn.q_proj(x).view(*shape)
        ).transpose(1, 2)

        k = attn.k_norm(
            attn.k_proj(x).view(*shape)
        ).transpose(1, 2)

        v = attn.v_proj(x).view(
            *shape
        ).transpose(1, 2)

        q, k = apply_rotary_pos_emb(
            q, k, *rope
        )

        context = block_decode_attention(
            cache,
            layer=layer_index,
            request_ids=ids,
            query=q[:, :, 0, :],
            new_keys=k[:, :, 0, :],
            new_values=v[:, :, 0, :],
            scale=attn.scaling,
        )

        context = context.reshape(
            batch, 1, -1
        ).contiguous()

        hidden = residual + attn.o_proj(context)

        residual = hidden
        hidden = residual + layer.mlp(
            layer.post_attention_layernorm(hidden)
        )

        new_kv.append((k, v))

    logits = model.lm_head(
        model.model.norm(hidden)
    )[:, -1, :].float()

    receipt = cache.prepare_batch({
        rid: length + 1
        for rid, length in zip(ids, lengths)
    })

    try:
        for layer_index, (keys, values) in enumerate(new_kv):
            for index, rid in enumerate(ids):
                cache.stage_write(
                    rid,
                    layer_index,
                    keys[index:index + 1].contiguous(),
                    values[index:index + 1].contiguous(),
                )

        cache.commit_batch(receipt)

    except Exception:
        if cache._pending is not None:
            cache.rollback_batch(receipt)
        raise

    return logits
