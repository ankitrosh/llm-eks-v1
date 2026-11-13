"""Full 40-layer HF eager vs V3.2D logit comparison."""

import math
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import transformers
from transformers.cache_utils import DynamicCache

from llm_runtime.providers.v3_qwen3_stage_only_factory import (
    Qwen3StageOnlyFactory,
)
from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_qwen3_stage_prefill import (
    PrefillHidden,
    stage_block_prefill,
)
import llm_runtime.v3_2_qwen3_stage_batch as batch_module
from llm_runtime.v3_2_block_attention_fast import (
    block_decode_attention_fast,
)

DEVICE = "cuda:0"
PROMPT = int(os.getenv("HF_PARITY_PROMPT", "128"))
DECODE = int(os.getenv("HF_PARITY_DECODE", "64"))
RID = 910123


def relative_error(actual, reference):
    x = actual.detach().float()
    y = reference.detach().float()
    assert x.shape == y.shape

    diff = x - y
    return (
        (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(y).clamp_min(1e-12)
        ).item(),
        diff.abs().max().item(),
    )


@torch.inference_mode()
def main():
    rank = int(os.environ["RANK"])

    assert int(os.environ["WORLD_SIZE"]) == 2
    assert rank in (0, 1)
    assert PROMPT >= 1 and DECODE >= 1
    assert torch.cuda.is_available()

    torch.cuda.set_device(0)
    torch.set_num_threads(2)

    print(
        f"FULL_PARITY_START rank={rank} "
        f"torch={torch.__version__} "
        f"transformers={transformers.__version__}",
        flush=True,
    )

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=900),
    )

    try:
        stage = Qwen3StageOnlyFactory(
            checkpoint_dir=os.getenv(
                "MODEL_DIR", "/models/qwen3-14b"
            ),
            device=DEVICE,
            dtype="bfloat16",
            split_at=20,
        )(rank)

        stage.layout = SimpleNamespace(world_size=2)

        partial = stage.model
        model = partial.model

        assert tuple(partial.owned_layers) == tuple(
            range(rank * 20, rank * 20 + 20)
        )
        assert stage.config._attn_implementation == "eager"
        assert all(
            t == "full_attention"
            for t in stage.config.layer_types
        )
        assert (
            PROMPT + DECODE
            <= stage.config.max_position_embeddings
        )

        print(
            f"FULL_PARITY_LOADED rank={rank} "
            f"layers={partial.owned_layers[0]}.."
            f"{partial.owned_layers[-1]} "
            f"gpu_gib="
            f"{torch.cuda.memory_allocated()/2**30:.3f}",
            flush=True,
        )

        # Independent caches for the two execution paths.
        hf_cache = DynamicCache(config=stage.config)

        cache_limit = PROMPT + DECODE + 16
        block_size = 16

        block_cache = BlockKVCache(
            layers=len(partial.owned_layers),
            capacity_blocks=math.ceil(
                cache_limit / block_size
            ),
            block_size=block_size,
            kv_heads=stage.config.num_key_value_heads,
            head_dim=stage.config.head_dim,
            max_context_tokens=cache_limit,
            dtype=torch.bfloat16,
            device=DEVICE,
        )

        # Use the V3.2D optimized decode backend.
        # Only changes this diagnostic process.
        batch_module.block_decode_attention = (
            block_decode_attention_fast
        )

        def hf_forward(input_data, start):
            if rank == 0:
                ids = torch.tensor(
                    [input_data],
                    device=DEVICE,
                    dtype=torch.long,
                )
                hidden = model.embed_tokens(ids)
            else:
                hidden = input_data

            assert hidden.dtype == torch.bfloat16

            length = hidden.shape[1]
            assert hidden.shape == (
                1, length, stage.config.hidden_size
            )

            positions = torch.arange(
                start,
                start + length,
                device=DEVICE,
                dtype=torch.long,
            )

            position_ids = positions.unsqueeze(0)
            rope = model.rotary_emb(hidden, position_ids)

            # Equivalent additive causal mask for HF eager
            # attention. Explicit construction avoids querying
            # unused cache layers on rank 1.
            key_positions = torch.arange(
                start + length, device=DEVICE
            )

            allowed = (
                key_positions.unsqueeze(0)
                <= positions.unsqueeze(1)
            )

            mask = torch.zeros(
                (1, 1, length, start + length),
                device=DEVICE,
                dtype=hidden.dtype,
            )

            mask.masked_fill_(
                ~allowed[None, None],
                torch.finfo(hidden.dtype).min,
            )

            for idx in partial.owned_layers:
                hidden = model.layers[idx](
                    hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=hf_cache,
                    use_cache=True,
                    cache_position=positions,
                    position_embeddings=rope,
                )

            if rank == 1:
                hidden = partial.lm_head(
                    model.norm(hidden)
                )[:, -1, :].float()

            return hidden

        # Fixed identical tokens, independent of predictions.
        vocab = stage.config.vocab_size

        all_ids = tuple(
            128 + ((7919 * i + 1543) % (vocab - 128))
            for i in range(PROMPT + DECODE)
        )

        forced = all_ids[PROMPT:]

        # ---------------- PREFILL ----------------

        if rank == 0:
            hf_hidden = hf_forward(
                all_ids[:PROMPT], 0
            )

            ours = stage_block_prefill(
                stage,
                block_cache,
                (RID,),
                (0,),
                (all_ids[:PROMPT],),
            )

            assert ours.lengths == (PROMPT,)

            rel, maximum = relative_error(
                ours.hidden, hf_hidden
            )

            print(
                f"FULL_PARITY_STAGE0_PREFILL "
                f"relative_l2={rel:.8f} "
                f"max_abs={maximum:.6f}",
                flush=True,
            )

            dist.send(hf_hidden.contiguous(), dst=1)
            dist.send(ours.hidden.contiguous(), dst=1)

        else:
            ref_input = torch.empty(
                (1, PROMPT, stage.config.hidden_size),
                device=DEVICE,
                dtype=torch.bfloat16,
            )
            ours_input = torch.empty_like(ref_input)

            dist.recv(ref_input, src=0)
            dist.recv(ours_input, src=0)

            logits_hf = hf_forward(ref_input, 0)

            logits_ours = stage_block_prefill(
                stage,
                block_cache,
                (RID,),
                (0,),
                PrefillHidden(
                    ours_input, (PROMPT,)
                ),
            )

        tested = 0
        mismatches = 0
        high_margin_mismatches = 0
        worst_rel = -1.0
        worst_step = -1

        def compare_logits(step, actual, reference):
            nonlocal tested, mismatches
            nonlocal high_margin_mismatches
            nonlocal worst_rel, worst_step

            assert actual.shape == reference.shape == (
                1, stage.config.vocab_size
            )

            rel, maximum = relative_error(
                actual, reference
            )

            top = torch.topk(
                reference[0].float(), k=2
            )

            hf_top1 = int(top.indices[0].item())
            our_top1 = int(
                actual[0].argmax().item()
            )
            margin = float(
                (top.values[0] - top.values[1]).item()
            )

            agree = int(hf_top1 == our_top1)

            tested += 1
            mismatches += 1 - agree

            if not agree and margin > 0.5:
                high_margin_mismatches += 1

            if rel > worst_rel:
                worst_rel = rel
                worst_step = step

            print(
                f"FULL_PARITY_LOGITS step={step} "
                f"relative_l2={rel:.8f} "
                f"max_abs={maximum:.6f} "
                f"top1_agree={agree} "
                f"hf_top1={hf_top1} "
                f"our_top1={our_top1} "
                f"hf_margin={margin:.6f}",
                flush=True,
            )

        if rank == 1:
            compare_logits(
                0, logits_ours, logits_hf
            )

        # ---------------- FORCED DECODE ----------------

        for step, token in enumerate(forced, 1):
            pos = PROMPT + step - 1

            if rank == 0:
                ref_hidden = hf_forward(
                    (token,), pos
                )

                ours_hidden = (
                    batch_module.stage_block_decode(
                        stage,
                        block_cache,
                        (RID,),
                        (token,),
                    )
                )

                assert (
                    ref_hidden.shape
                    == ours_hidden.shape
                    == (1, 1, stage.config.hidden_size)
                )

                if step in (1, 16, 32, 64, DECODE):
                    rel, maximum = relative_error(
                        ours_hidden, ref_hidden
                    )

                    print(
                        f"FULL_PARITY_STAGE0 step={step} "
                        f"relative_l2={rel:.8f} "
                        f"max_abs={maximum:.6f}",
                        flush=True,
                    )

                dist.send(
                    ref_hidden.contiguous(), dst=1
                )
                dist.send(
                    ours_hidden.contiguous(), dst=1
                )

            else:
                ref_in = torch.empty(
                    (1, 1, stage.config.hidden_size),
                    device=DEVICE,
                    dtype=torch.bfloat16,
                )
                ours_in = torch.empty_like(ref_in)

                dist.recv(ref_in, src=0)
                dist.recv(ours_in, src=0)

                ref_logits = hf_forward(
                    ref_in, pos
                )

                ours_logits = (
                    batch_module.stage_block_decode(
                        stage,
                        block_cache,
                        (RID,),
                        ours_in,
                    )
                )

                compare_logits(
                    step, ours_logits, ref_logits
                )

        # ---------------- CLEANUP ----------------

        assert block_cache._committed[RID] == (
            PROMPT + DECODE
        )
        assert block_cache.release_request(RID)

        snapshot = block_cache.snapshot()

        assert snapshot["requests"] == 0
        assert snapshot["allocated_blocks"] == 0
        assert not snapshot["pending_batch"]

        if rank == 1:
            print(
                f"FULL_PARITY_COMPLETE "
                f"measured={tested} "
                f"top1_mismatches={mismatches} "
                f"high_margin_mismatches="
                f"{high_margin_mismatches} "
                f"worst_step={worst_step} "
                f"worst_relative_l2={worst_rel:.8f} "
                f"cache_requests={snapshot['requests']}",
                flush=True,
            )
        else:
            print(
                "FULL_PARITY_RANK0_COMPLETE "
                "cache_requests=0",
                flush=True,
            )

        dist.barrier(device_ids=[0])

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
