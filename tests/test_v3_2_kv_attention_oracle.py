"""V3.2 KV integrity and attention differential test.

Does not load model weights or require NCCL.
Run on CPU first, then CUDA.
"""

import argparse
import math

import torch

from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_block_attention import (
    block_decode_attention,
)
from llm_runtime.v3_2_block_attention_fast import (
    block_decode_attention_fast,
)


BLOCK_SIZE = 16
KV_HEADS = 8
QUERY_HEADS = 40
HEAD_DIM = 128
DTYPE = torch.bfloat16


def make_tensor(shape, generator, device):
    return torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=DTYPE)


def attention_oracle(q, keys, values):
    """Independent FP32 full-context attention calculation."""

    group = QUERY_HEADS // KV_HEADS

    k = keys.repeat_interleave(group, dim=0).float()
    v = values.repeat_interleave(group, dim=0).float()

    scores = torch.einsum(
        "hd,htd->ht", q.float(), k.float()
    ) / math.sqrt(HEAD_DIM)

    probabilities = torch.softmax(
        scores, dim=-1
    )

    return torch.einsum(
        "ht,htd->hd", probabilities, v.float()
    ).to(DTYPE)


@torch.inference_mode()
def run_case(device, lengths, case_number):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261111 + case_number)

    ids = tuple(
        101 + i * 103 for i in range(len(lengths))
    )

    capacity = sum(
        (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        for length in lengths
    ) + 8

    cache = BlockKVCache(
        layers=1,
        capacity_blocks=capacity,
        block_size=BLOCK_SIZE,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_context_tokens=256,
        dtype=DTYPE,
        device=device,
    )

    originals = {}

    # 1. Write each request through the actual KV allocator.
    for rid, length in zip(ids, lengths):
        k = make_tensor(
            (1, KV_HEADS, length, HEAD_DIM),
            generator, device,
        )

        v = make_tensor(
            (1, KV_HEADS, length, HEAD_DIM),
            generator, device,
        )

        receipt = cache.prepare_batch({rid: length})

        cache.stage_write(rid, 0, k, v)
        cache.commit_batch(receipt)

        originals[rid] = (k, v)

    # 2. Verify exact KV roundtrip.
    for rid, length in zip(ids, lengths):
        original_k, original_v = originals[rid]
        cached_k, cached_v = cache.read_request(rid)[0]

        assert cached_k.shape == original_k.shape
        assert cached_v.shape == original_v.shape

        assert torch.equal(cached_k, original_k), (
            f"KV KEY CORRUPTION rid={rid} length={length}"
        )

        assert torch.equal(cached_v, original_v), (
            f"KV VALUE CORRUPTION rid={rid} length={length}"
        )

        expected_blocks = (
            length + BLOCK_SIZE - 1
        ) // BLOCK_SIZE

        assert (
            len(cache.manager.block_table(rid))
            == expected_blocks
        )

    print(
        f"KV_EXACT_PASS device={device} "
        f"lengths={lengths}",
        flush=True,
    )

    # 3. Deliberately reorder the requests during attention.
    ordered_ids = tuple(reversed(ids))

    q = make_tensor(
        (len(ids), QUERY_HEADS, HEAD_DIM),
        generator, device,
    )

    new_k = make_tensor(
        (len(ids), KV_HEADS, HEAD_DIM),
        generator, device,
    )

    new_v = make_tensor(
        (len(ids), KV_HEADS, HEAD_DIM),
        generator, device,
    )

    # 4. Compare reference block attention against an
    # independent full-context FP32 computation.
    reference = block_decode_attention(
        cache,
        layer=0,
        request_ids=ordered_ids,
        query=q,
        new_keys=new_k,
        new_values=new_v,
    )

    expected_rows = []

    for i, rid in enumerate(ordered_ids):
        original_k, original_v = originals[rid]

        keys = torch.cat(
            (original_k[0], new_k[i, :, None, :]),
            dim=1,
        )

        values = torch.cat(
            (original_v[0], new_v[i, :, None, :]),
            dim=1,
        )

        expected_rows.append(
            attention_oracle(
                q[i],
                keys,
                values,
            )
        )

    expected = torch.stack(expected_rows)

    def compare(label, actual):
        difference = (
            actual.float() - expected.float()
        )

        max_abs = difference.abs().max().item()

        rel_l2 = (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(
                expected.float()
            ).clamp_min(1e-12)
        ).item()

        print(
            f"ATTENTION_RESULT "
            f"device={device} "
            f"lengths={lengths} "
            f"path={label} "
            f"relative_l2={rel_l2:.8f} "
            f"max_abs={max_abs:.6f}",
            flush=True,
        )

        # Diagnostic guardrail, not bitwise equality.
        assert rel_l2 < 0.02, (
            f"{label}: excessive relative error"
        )

        assert max_abs < 0.05, (
            f"{label}: excessive absolute error"
        )

    compare("block_reference", reference)

    # 5. Run CUDA optimized attention where supported.
    if device.type == "cuda":
        fast = block_decode_attention_fast(
            cache,
            layer=0,
            request_ids=ordered_ids,
            query=q,
            new_keys=new_k,
            new_values=new_v,
        )

        compare("cuda_sdpa", fast)

    # 6. Release requests and verify zero leaks.
    for rid in ids:
        assert cache.release_request(rid)

    snapshot = cache.snapshot()

    assert snapshot["requests"] == 0
    assert snapshot["allocated_blocks"] == 0
    assert not snapshot["pending_batch"]

    print(
        f"CACHE_CLEANUP_PASS lengths={lengths}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if args.device == "cuda" else "cpu"
    )

    if device.type == "cuda":
        assert torch.cuda.is_available()

    torch.set_num_threads(2)

    lengths_to_test = (
        15, 16, 17,
        127, 128, 129,
        143, 144, 145,
        159, 160, 161,
        180,
    )

    for number, length in enumerate(lengths_to_test):
        run_case(device, (length,), number)

    # Mixed-length and reordered-request tests.
    run_case(device, (127, 128, 129), 100)
    run_case(device, (143, 160, 180), 101)

    print(
        f"V32_KV_ATTENTION_ORACLE_PASS "
        f"device={device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
