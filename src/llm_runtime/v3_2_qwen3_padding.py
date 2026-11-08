"""Variable-length Qwen3 input batching helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PaddedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


def prepare_prefill(prompts, pad_token_id=0, device="cpu"):
    """Left-pad prompts, preserving logical positions."""
    if not prompts:
        raise ValueError("At least one prompt required")

    prompts = [tuple(p) for p in prompts]

    if any(
        not p or any(type(t) is not int or t < 0 for t in p)
        for p in prompts
    ):
        raise ValueError("Invalid prompt tokens")

    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Invalid pad token")

    length = max(map(len, prompts))
    ids = []
    masks = []

    for prompt in prompts:
        padding = length - len(prompt)

        ids.append(
            [pad_token_id] * padding + list(prompt)
        )
        masks.append(
            [0] * padding + [1] * len(prompt)
        )

    input_ids = torch.tensor(
        ids, dtype=torch.long, device=device
    )
    mask = torch.tensor(
        masks, dtype=torch.long, device=device
    )

    positions = (mask.cumsum(-1) - 1).clamp(min=0)

    return PaddedBatch(input_ids, mask, positions)


def prepare_decode(previous_mask, tokens):
    """Append one token per request to an existing batch."""
    if previous_mask.ndim != 2:
        raise ValueError("Expected 2D attention mask")

    tokens = tuple(tokens)

    if (
        len(tokens) != previous_mask.shape[0]
        or any(type(t) is not int or t < 0 for t in tokens)
    ):
        raise ValueError("Invalid decode tokens")

    device = previous_mask.device

    ids = torch.tensor(
        [[t] for t in tokens],
        dtype=torch.long,
        device=device,
    )

    extra = torch.ones(
        (len(tokens), 1),
        dtype=previous_mask.dtype,
        device=device,
    )

    mask = torch.cat(
        (previous_mask, extra), dim=-1
    )

    positions = mask.long().sum(-1, keepdim=True) - 1

    return PaddedBatch(ids, mask, positions)
