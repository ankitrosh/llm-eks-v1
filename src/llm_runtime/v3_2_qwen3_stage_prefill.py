"""V3.2 batched/chunked Qwen3 stage prefill.

Supports:
- Multiple requests with different chunk lengths
- Initial and incremental prefill
- Stage-local block-backed KV ownership
- Batched projections and MLP
- Per-request causal attention reference

Full-attention Qwen3 only. CPU correctness reference;
not an optimized paged-attention kernel.
"""

from dataclasses import dataclass
import os

import torch

from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
)


@dataclass(frozen=True)
class PrefillHidden:
    """Padded hidden states forwarded between stages."""
    hidden: torch.Tensor
    lengths: tuple[int, ...]


@torch.inference_mode()
def stage_block_prefill(
    stage,
    cache,
    request_ids,
    positions,
    payload,
):
    """Execute one batched prefill microbatch.

    Rank 0 payload:
        tuple of token sequences, one per request.

    Other ranks:
        PrefillHidden([B, max_chunk, hidden_size], lengths)

    positions:
        Number of previously committed KV tokens per request.

    Returns PrefillHidden for intermediate stages,
    or [B, vocab_size] logits at the final stage.
    """
    ids = tuple(request_ids)
    positions = tuple(positions)

    if (
        not ids
        or len(ids) != len(positions)
        or len(set(ids)) != len(ids)
        or any(
            type(rid) is not int or rid < 1
            for rid in ids
        )
        or any(
            type(pos) is not int or pos < 0
            for pos in positions
        )
    ):
        raise ValueError("Invalid prefill batch metadata")

    config = stage.config
    owned = tuple(stage.model.owned_layers)

    if any(
        kind != "full_attention"
        for kind in config.layer_types
    ):
        raise ValueError("Full-attention Qwen3 required")

    if (
        not owned
        or cache.layers != len(owned)
        or cache.kv_heads != config.num_key_value_heads
        or cache.head_dim != config.head_dim
    ):
        raise ValueError("Stage/cache geometry mismatch")

    if cache._pending is not None:
        raise RuntimeError("KV transaction already pending")

    # Validate ownership and KV position before computation.
    for rid, position in zip(ids, positions):
        committed = cache._committed.get(rid, 0)

        if committed != position:
            raise RuntimeError(
                "Stage-local KV position mismatch"
            )

    batch = len(ids)
    device = stage.device

    if stage.rank == 0:
        if not isinstance(payload, (tuple, list)):
            raise ValueError("Expected prefill token sequences")

        if len(payload) != batch:
            raise ValueError("Prefill batch size mismatch")

        sequences = tuple(tuple(x) for x in payload)

        if any(
            not sequence
            or any(
                type(token) is not int or token < 0
                for token in sequence
            )
            for sequence in sequences
        ):
            raise ValueError("Invalid prefill tokens")

        lengths = tuple(
            len(sequence) for sequence in sequences
        )

        width = max(lengths)

        padded = torch.zeros(
            (batch, width),
            dtype=torch.long,
            device=device,
        )

        for index, sequence in enumerate(sequences):
            padded[index, :len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )

        hidden = stage.model.model.embed_tokens(padded)

    else:
        if not isinstance(payload, PrefillHidden):
            raise ValueError("Expected PrefillHidden")

        hidden = payload.hidden
        lengths = tuple(payload.lengths)

        if (
            len(lengths) != batch
            or any(
                type(length) is not int or length < 1
                for length in lengths
            )
            or not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape != (
                batch,
                max(lengths),
                config.hidden_size,
            )
            or hidden.device != device
        ):
            raise ValueError(
                "Invalid forwarded prefill hidden states"
            )

    if hidden.dtype != cache.dtype:
        raise ValueError("Prefill hidden/KV dtype mismatch")

    for position, length in zip(positions, lengths):
        if (
            position + length
            > cache.manager.max_context_tokens
        ):
            raise ValueError(
                "Prefill exceeds KV context capacity"
            )

    width = hidden.shape[1]

    # Padded query positions are irrelevant, but use valid
    # nonnegative position IDs for rotary embedding.
    position_ids = torch.zeros(
        (batch, width),
        dtype=torch.long,
        device=device,
    )

    for index, (position, length) in enumerate(
        zip(positions, lengths)
    ):
        position_ids[index, :length] = torch.arange(
            position,
            position + length,
            device=device,
            dtype=torch.long,
        )

    rope = stage.model.model.rotary_emb(
        hidden, position_ids
    )

    # Existing KV is read only during chunked prefill.
    # Decode continues to use direct block attention.
    histories = {
        rid: cache.read_request(rid)
        for rid, position in zip(ids, positions)
        if position > 0
    }

    pending_layers = []

    for local_index, global_index in enumerate(owned):
        layer = stage.model.model.layers[global_index]
        attn = layer.self_attn

        residual = hidden
        x = layer.input_layernorm(hidden)

        shape = (
            batch, width, -1, attn.head_dim
        )

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

        query_heads = q.shape[1]
        kv_heads = k.shape[1]

        if query_heads % kv_heads:
            raise ValueError(
                "Invalid grouped-query attention geometry"
            )

        groups = query_heads // kv_heads

        context = torch.zeros(
            (batch, width, query_heads * attn.head_dim),
            dtype=hidden.dtype,
            device=device,
        )

        for index, (rid, old_len, take) in enumerate(
            zip(ids, positions, lengths)
        ):
            current_k = k[
                index:index + 1, :, :take, :
            ]
            current_v = v[
                index:index + 1, :, :take, :
            ]

            if old_len:
                previous_k, previous_v = histories[
                    rid
                ][local_index]

                full_k = torch.cat(
                    (previous_k, current_k), dim=2
                )
                full_v = torch.cat(
                    (previous_v, current_v), dim=2
                )
            else:
                full_k = current_k
                full_v = current_v

            repeated_k = full_k.repeat_interleave(
                groups, dim=1
            )
            repeated_v = full_v.repeat_interleave(
                groups, dim=1
            )

            query = q[
                index:index + 1, :, :take, :
            ]

            use_fp32_attention = (
                os.getenv("D_PREFILL_FP32_DIAG", "0") == "1"
            )

            if use_fp32_attention:
                scores = (
                    query.float()
                    @ repeated_k.float().transpose(-2, -1)
                ) * attn.scaling
            else:
                scores = (
                    query @ repeated_k.transpose(-2, -1)
                ) * attn.scaling

            # Query i may see all committed history plus
            # the first i+1 tokens of its current chunk.
            key_positions = torch.arange(
                old_len + take,
                device=device,
            )

            query_positions = (
                old_len
                + torch.arange(
                    take, device=device
                )
            )

            invalid = (
                key_positions.unsqueeze(0)
                > query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                invalid[None, None],
                torch.finfo(scores.dtype).min,
            )

            if use_fp32_attention:
                probabilities = torch.softmax(
                    scores, dim=-1,
                    dtype=torch.float32,
                )
                attended = (
                    probabilities @ repeated_v.float()
                ).to(query.dtype)
            else:
                probabilities = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)

                attended = (
                    probabilities @ repeated_v
                )

            context[index, :take] = (
                attended.transpose(1, 2)
                .reshape(take, -1)
            )

        hidden = residual + attn.o_proj(context)

        hidden = hidden + layer.mlp(
            layer.post_attention_layernorm(hidden)
        )

        pending_layers.append((k, v))

    if stage.rank == stage.layout.world_size - 1:
        last_hidden = torch.stack(
            [
                hidden[index, length - 1]
                for index, length in enumerate(lengths)
            ]
        )

        output = stage.model.lm_head(
            stage.model.model.norm(last_hidden)
        ).float()

    else:
        output = PrefillHidden(
            hidden.contiguous(),
            lengths,
        )

    # Commit after all model layers succeed.
    receipt = cache.prepare_batch({
        rid: old_len + length
        for rid, old_len, length in zip(
            ids, positions, lengths
        )
    })

    try:
        for local_index, (keys, values) in enumerate(
            pending_layers
        ):
            for index, (rid, length) in enumerate(
                zip(ids, lengths)
            ):
                cache.stage_write(
                    rid,
                    local_index,
                    keys[
                        index:index + 1,
                        :,
                        :length,
                        :,
                    ].contiguous(),
                    values[
                        index:index + 1,
                        :,
                        :length,
                        :,
                    ].contiguous(),
                )

        cache.commit_batch(receipt)

    except Exception:
        if cache._pending is not None:
            cache.rollback_batch(receipt)
        raise

    return output
