"""Batched Qwen3 decode for one pipeline stage.

Uses stage-local physical KV blocks and global Qwen3 layer IDs.
Supports full-attention Qwen3, one token per request.
"""

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
)

from llm_runtime.v3_2_block_attention import (
    block_decode_attention,
)


def stage_block_decode(stage, cache, request_ids, payload):
    """Execute one decode microbatch on a single pipeline stage.

    Rank 0 payload: tuple of token IDs.
    Other ranks: hidden states [B, 1, hidden_size].

    Returns hidden states or final logits.
    """
    ids = tuple(request_ids)

    if (
        not ids
        or any(type(rid) is not int or rid < 1 for rid in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Invalid request IDs")

    rank = stage.rank
    partial = stage.model
    config = stage.config
    owned = tuple(partial.owned_layers)

    if (
        not owned
        or cache.layers != len(owned)
        or cache.kv_heads != config.num_key_value_heads
        or cache.head_dim != config.head_dim
    ):
        raise ValueError("Stage/cache geometry mismatch")

    if any(x != "full_attention" for x in config.layer_types):
        raise ValueError("Unsupported attention type")

    if cache._pending is not None:
        raise RuntimeError("Stage KV transaction pending")

    lengths = []

    for rid in ids:
        length = cache._committed.get(rid)

        if length is None or length < 1:
            raise ValueError("Request lacks stage-local KV")

        if length >= cache.manager.max_context_tokens:
            raise ValueError("Stage context capacity exceeded")

        lengths.append(length)

    device = stage.device

    if rank == 0:
        tokens = tuple(payload)

        if (
            len(tokens) != len(ids)
            or any(
                type(token) is not int or token < 0
                for token in tokens
            )
        ):
            raise ValueError("Invalid decode tokens")

        input_ids = torch.tensor(
            [[token] for token in tokens],
            dtype=torch.long,
            device=device,
        )

        hidden = partial.model.embed_tokens(input_ids)
    else:
        hidden = payload

        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.shape != (
                len(ids), 1, config.hidden_size
            )
            or hidden.device != device
        ):
            raise ValueError("Invalid incoming hidden states")

    if hidden.dtype != cache.dtype:
        raise ValueError("Hidden/KV dtype mismatch")

    positions = torch.tensor(
        [[length] for length in lengths],
        dtype=torch.long,
        device=device,
    )

    new_kv = []

    with torch.inference_mode():
        rope = partial.model.rotary_emb(
            hidden, positions
        )

        for local_index, global_index in enumerate(owned):
            layer = partial.model.layers[global_index]
            attn = layer.self_attn

            residual = hidden
            x = layer.input_layernorm(hidden)

            batch = len(ids)
            shape = (batch, 1, -1, attn.head_dim)

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
                layer=local_index,
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

            hidden = hidden + layer.mlp(
                layer.post_attention_layernorm(hidden)
            )

            new_kv.append((k, v))

        # Compute the stage output before committing KV.
        if rank == stage.layout.world_size - 1:
            output = partial.lm_head(
                partial.model.norm(hidden)
            )[:, -1, :].float()
        else:
            output = hidden.contiguous()

        receipt = cache.prepare_batch({
            rid: length + 1
            for rid, length in zip(ids, lengths)
        })

        try:
            for local_index, (keys, values) in enumerate(new_kv):
                for batch_index, rid in enumerate(ids):
                    cache.stage_write(
                        rid,
                        local_index,
                        keys[
                            batch_index:batch_index + 1
                        ].contiguous(),
                        values[
                            batch_index:batch_index + 1
                        ].contiguous(),
                    )

            cache.commit_batch(receipt)

        except Exception:
            if cache._pending is not None:
                cache.rollback_batch(receipt)
            raise

    return output


def validate_stage_alignment(caches, request_ids):
    """Check identical request KV positions across ranks."""
    ids = tuple(request_ids)

    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid batch membership")

    expected = None

    for cache in caches:
        positions = tuple(
            cache._committed.get(rid)
            for rid in ids
        )

        if any(
            type(value) is not int or value < 1
            for value in positions
        ):
            raise ValueError("Missing stage KV position")

        if expected is None:
            expected = positions
        elif positions != expected:
            raise RuntimeError(
                "Pipeline stages disagree on KV positions"
            )

    return expected
