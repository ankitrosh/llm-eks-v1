"""Dynamic K/V batching for full-attention Qwen3 models.

Correctness-first implementation. Repacking copies KV tensors.
Sliding-window, hybrid and static caches are not supported.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer


@dataclass(frozen=True)
class MergedCache:
    cache: DynamicCache
    attention_mask: torch.Tensor
    lengths: tuple[int, ...]


def _check_caches(caches):
    if not caches:
        raise ValueError("No caches supplied")

    n_layers = len(caches[0].layers)
    if n_layers < 1:
        raise ValueError("Empty cache")

    lengths = []

    for cache in caches:
        if not isinstance(cache, DynamicCache):
            raise TypeError("DynamicCache required")
        if len(cache.layers) != n_layers:
            raise ValueError("Layer count mismatch")

        length = cache.layers[0].get_seq_length()
        if length < 1:
            raise ValueError("Uninitialized cache")

        for layer in cache.layers:
            if not isinstance(layer, DynamicLayer):
                raise ValueError(
                    "Only full-attention DynamicLayer supported"
                )
            if layer.get_seq_length() != length:
                raise ValueError("Layer KV length mismatch")
            if layer.keys.shape[0] != 1:
                raise ValueError(
                    "Expected individual batch size 1"
                )

        lengths.append(length)

    return tuple(lengths), n_layers


def merge_caches(caches, config):
    caches = tuple(caches)
    lengths, n_layers = _check_caches(caches)
    maximum = max(lengths)

    merged = DynamicCache(config=config)

    for layer_idx in range(n_layers):
        keys = []
        values = []
        reference = caches[0].layers[layer_idx].keys

        for cache, length in zip(caches, lengths):
            layer = cache.layers[layer_idx]

            if (
                layer.keys.dtype != reference.dtype
                or layer.keys.device != reference.device
                or layer.keys.shape[1:] != (
                    reference.shape[1],
                    length,
                    reference.shape[3],
                )
            ):
                raise ValueError("Incompatible KV layout")

            padding = maximum - length

            keys.append(
                F.pad(layer.keys, (0, 0, padding, 0))
            )
            values.append(
                F.pad(layer.values, (0, 0, padding, 0))
            )

        merged.update(
            torch.cat(keys, dim=0),
            torch.cat(values, dim=0),
            layer_idx,
        )

    mask = torch.zeros(
        (len(caches), maximum),
        dtype=torch.long,
        device=caches[0].layers[0].keys.device,
    )

    for index, length in enumerate(lengths):
        mask[index, -length:] = 1

    return MergedCache(merged, mask, lengths)


def split_caches(cache, lengths, config):
    lengths = tuple(lengths)
    batch_size = cache.layers[0].keys.shape[0]
    available = cache.layers[0].get_seq_length()

    if (
        len(lengths) != batch_size
        or any(
            type(length) is not int
            or length < 1
            or length > available
            for length in lengths
        )
    ):
        raise ValueError("Invalid output cache lengths")

    output = []

    for index, length in enumerate(lengths):
        individual = DynamicCache(config=config)

        for layer_idx, layer in enumerate(cache.layers):
            individual.update(
                layer.keys[
                    index:index + 1, :, -length:, :
                ].contiguous(),
                layer.values[
                    index:index + 1, :, -length:, :
                ].contiguous(),
                layer_idx,
            )

        output.append(individual)

    return tuple(output)


@torch.inference_mode()
def batched_decode_step(model, caches, tokens):
    """One model forward with heterogeneous per-request KV."""
    caches = tuple(caches)
    tokens = tuple(tokens)

    if (
        len(caches) != len(tokens)
        or any(type(token) is not int or token < 0
               for token in tokens)
    ):
        raise ValueError("Invalid batched decode inputs")

    merged = merge_caches(caches, model.config)
    count = len(tokens)

    input_ids = torch.tensor(
        [[token] for token in tokens],
        dtype=torch.long,
        device=merged.attention_mask.device,
    )

    attention_mask = torch.cat(
        (
            merged.attention_mask,
            torch.ones(
                (count, 1),
                dtype=torch.long,
                device=input_ids.device,
            ),
        ),
        dim=1,
    )

    # Logical positions differ per request.
    position_ids = torch.tensor(
        [[length] for length in merged.lengths],
        dtype=torch.long,
        device=input_ids.device,
    )

    # Physical packed cache position is shared.
    cache_position = torch.tensor(
        [merged.attention_mask.shape[1]],
        dtype=torch.long,
        device=input_ids.device,
    )

    result = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=merged.cache,
        use_cache=True,
    )

    updated = split_caches(
        result.past_key_values,
        tuple(length + 1 for length in merged.lengths),
        model.config,
    )

    return result.logits[:, -1, :], updated
