"""Compare explicit causal masking to lower-right SDPA bias.

Requires PyTorch and Transformers 4.56.2.
Runs entirely on CPU; no GPU or model weights needed.
"""

import math
from types import SimpleNamespace

import torch
from torch.nn.attention.bias import causal_lower_right
from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward,
)


def run_case(cached_tokens, new_tokens):
    torch.manual_seed(42)

    total = cached_tokens + new_tokens
    heads = 8
    kv_heads = 2
    dim = 64

    q = torch.randn(1, heads, new_tokens, dim)
    k = torch.randn(1, kv_heads, total, dim)
    v = torch.randn(1, kv_heads, total, dim)

    positions = torch.arange(
        cached_tokens, total
    )
    key_positions = torch.arange(total)

    visible = (
        key_positions[None, :] <= positions[:, None]
    )

    explicit = torch.zeros(
        (1, 1, new_tokens, total),
        dtype=q.dtype,
    )
    explicit.masked_fill_(
        ~visible[None, None, :, :],
        torch.finfo(q.dtype).min,
    )

    lower_right = causal_lower_right(
        new_tokens, total
    )

    module = SimpleNamespace(
        num_key_value_groups=heads // kv_heads,
        is_causal=True,
    )

    def attention(mask):
        result, _ = sdpa_attention_forward(
            module, q, k, v, mask,
            dropout=0.0,
            scaling=1 / math.sqrt(dim),
        )
        return result

    expected = attention(explicit)
    actual = attention(lower_right)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-4,
        atol=2e-5,
    )

    print(
        f"PASS: cached={cached_tokens}, "
        f"new={new_tokens}, "
        f"max_diff={(actual - expected).abs().max().item():.7f}"
    )


if __name__ == "__main__":
    print("Torch:", torch.__version__)
    for cached, new in [
        (8, 2),
        (64, 8),
        (256, 16),
        (512, 32),
    ]:
        run_case(cached, new)

    print("PASS: Lower-right mask CPU correctness")
