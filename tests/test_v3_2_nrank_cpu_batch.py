"""CPU pipeline batched-decode correctness for 2, 3, 4 ranks."""

import pytest
import torch

from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_pipeline_scheduler import StageTask
from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)
from llm_runtime.v3_2_kv_storage import BlockKVCache
from llm_runtime.v3_2_batch_planner import (
    ContinuousBatchPlanner,
)
from llm_runtime.v3_2_qwen3_stage_batch import (
    stage_block_decode,
    validate_stage_alignment,
)


LAYOUTS = {
    2: (0, 3, 6),
    3: (0, 1, 3, 6),
    4: (0, 1, 2, 4, 6),
}

PROMPTS = (
    (2, 5, 8),
    (7, 4, 12, 9, 3),
)


def make_model():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 6,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )

    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def create_stages(model, checkpoint, boundaries):
    model.save_pretrained(
        checkpoint, safe_serialization=True
    )

    factory = Qwen3NRankStageOnlyFactory(
        checkpoint_dir=str(checkpoint),
        layer_boundaries=boundaries,
        device="cpu",
        dtype="float32",
    )

    stages = [
        factory(rank)
        for rank in range(len(boundaries) - 1)
    ]

    caches = []

    for stage in stages:
        caches.append(BlockKVCache(
            layers=len(stage.model.owned_layers),
            capacity_blocks=16,
            block_size=4,
            kv_heads=2,
            head_dim=16,
            max_context_tokens=32,
            dtype=torch.float32,
        ))

    return stages, caches


def prefill_request(stages, caches, rid, prompt):
    payload = torch.tensor(
        [prompt], dtype=torch.long
    )

    with torch.inference_mode():
        for stage, cache in zip(stages, caches):
            result = stage(
                StageTask(rid, 0, 0, payload),
                None,
            )

            payload = result.payload
            dynamic = result.cache_state

            receipt = cache.prepare_batch({
                rid: len(prompt)
            })

            try:
                for local, global_layer in enumerate(
                    stage.model.owned_layers
                ):
                    layer = dynamic.layers[global_layer]

                    cache.stage_write(
                        rid,
                        local,
                        layer.keys,
                        layer.values,
                    )

                cache.commit_batch(receipt)

            except Exception:
                if cache._pending is not None:
                    cache.rollback_batch(receipt)
                raise

    return payload[:, -1, :].float()


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_distributed_batched_decode_parity(tmp_path, ranks):
    model = make_model()
    boundaries = LAYOUTS[ranks]

    stages, caches = create_stages(
        model,
        tmp_path / "checkpoint",
        boundaries,
    )

    planner = ContinuousBatchPlanner({
        "pipeline": {
            "memory": {
                "max_active_requests": 2,
                "max_total_tokens_per_request": 32,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "policy": "fcfs",
            "max_requests_per_batch": 2,
            "max_batch_tokens": 16,
            "max_queue_requests": 2,
        },
    })

    with torch.inference_mode():
        references = [
            model(
                torch.tensor([prompt]),
                use_cache=True,
            )
            for prompt in PROMPTS
        ]

        for rid, prompt in enumerate(PROMPTS, start=1):
            planner.submit(
                rid, prompt, max_new_tokens=2
            )

        prefill_plan = planner.next_batch()
        assert prefill_plan.phase == "prefill"
        assert len(prefill_plan.items) == 2

        prefill_logits = []

        for item in prefill_plan.items:
            prefill_logits.append(
                prefill_request(
                    stages, caches,
                    item.request_id, item.tokens,
                )
            )

        for actual, reference in zip(
            prefill_logits, references
        ):
            torch.testing.assert_close(
                actual,
                reference.logits[:, -1, :],
                rtol=1e-4,
                atol=1e-4,
            )

        first_tokens = {
            rid: int(logits.argmax())
            for rid, logits in enumerate(
                prefill_logits, start=1
            )
        }

        planner.complete_batch(first_tokens)

        positions = validate_stage_alignment(
            caches, (1, 2)
        )
        assert positions == (3, 5)

        decode_plan = planner.next_batch()
        assert decode_plan.phase == "decode"
        assert len(decode_plan.items) == 2

        ids = tuple(
            item.request_id
            for item in decode_plan.items
        )

        tokens = tuple(
            item.tokens[0]
            for item in decode_plan.items
        )

        payload = tokens

        for stage, cache in zip(stages, caches):
            payload = stage_block_decode(
                stage, cache, ids, payload
            )

        actual = payload

        references = [
            model(
                torch.tensor([[first_tokens[rid]]]),
                past_key_values=reference.past_key_values,
                use_cache=True,
            )
            for rid, reference in enumerate(
                references, start=1
            )
        ]

        expected = torch.cat([
            reference.logits[:, -1, :]
            for reference in references
        ])

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

        assert validate_stage_alignment(
            caches, ids
        ) == (4, 6)

        # Verify KV values for every rank and owned layer.
        for stage, cache in zip(stages, caches):
            for rid, reference in enumerate(
                references, start=1
            ):
                stored = cache.read_request(rid)

                for local, global_layer in enumerate(
                    stage.model.owned_layers
                ):
                    actual_k, actual_v = stored[local]
                    ref = (
                        reference.past_key_values
                        .layers[global_layer]
                    )

                    torch.testing.assert_close(
                        actual_k, ref.keys,
                        rtol=1e-4, atol=1e-4,
                    )
                    torch.testing.assert_close(
                        actual_v, ref.values,
                        rtol=1e-4, atol=1e-4,
                    )

        result = planner.complete_batch({
            rid: int(actual[i].argmax())
            for i, rid in enumerate(ids)
        })

        assert set(result.finished) == {1, 2}

        for rid in planner.drain_releases():
            for cache in caches:
                assert cache.release_request(rid)

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


def test_reject_stage_position_mismatch():
    a = BlockKVCache(
        layers=1,
        capacity_blocks=4,
        block_size=4,
        kv_heads=2,
        head_dim=16,
        max_context_tokens=16,
    )
    b = BlockKVCache(
        layers=1,
        capacity_blocks=4,
        block_size=4,
        kv_heads=2,
        head_dim=16,
        max_context_tokens=16,
    )

    for cache, length in ((a, 3), (b, 5)):
        receipt = cache.prepare_batch({1: length})
        cache.stage_write(
            1, 0,
            torch.zeros(1, 2, length, 16),
            torch.zeros(1, 2, length, 16),
        )
        cache.commit_batch(receipt)

    with pytest.raises(RuntimeError, match="disagree"):
        validate_stage_alignment((a, b), (1,))


# Gate 7B: Real Qwen3 stages connected by ordered CPU transport.

from llm_runtime.v3_2_microbatch_transport import (
    CPUMicrobatchTransport,
)

from llm_runtime.v3_2_policy_transport import (
    PolicyCPUMicrobatchTransport,
)


@pytest.mark.parametrize(
    "transport_cls",
    [CPUMicrobatchTransport, PolicyCPUMicrobatchTransport],
)
@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_threaded_microbatch_pipeline_parity(
    tmp_path, ranks, transport_cls
):
    model = make_model()

    stages, caches = create_stages(
        model,
        tmp_path / "threaded-checkpoint",
        LAYOUTS[ranks],
    )

    prompts = (
        (2, 5, 8),
        (7, 4, 12, 9, 3),
        (3, 9, 11, 17),
        (4, 8, 16, 32, 64, 96),
    )

    with torch.inference_mode():
        references = [
            model(
                torch.tensor([prompt]),
                use_cache=True,
            )
            for prompt in prompts
        ]

        # Gate 7B retains the validated per-request prefill.
        for rid, prompt in enumerate(prompts, start=1):
            actual_prefill = prefill_request(
                stages,
                caches,
                rid,
                prompt,
            )

            torch.testing.assert_close(
                actual_prefill,
                references[rid - 1].logits[:, -1, :],
                rtol=1e-4,
                atol=1e-4,
            )

        tokens = tuple(
            int(reference.logits[0, -1].argmax())
            for reference in references
        )

    stage_functions = []

    for stage, cache in zip(stages, caches):

        def execute(packet, stage=stage, cache=cache):
            positions = tuple(
                cache._committed.get(rid)
                for rid in packet.request_ids
            )

            if positions != packet.positions:
                raise RuntimeError(
                    "Stage-local KV position mismatch"
                )

            with torch.inference_mode():
                return stage_block_decode(
                    stage,
                    cache,
                    packet.request_ids,
                    packet.payload,
                )

        stage_functions.append(execute)

    with transport_cls(
        stage_functions,
        max_pending=2,
    ) as pipeline:

        # Both microbatches are admitted before either
        # result is consumed.
        first = pipeline.submit(
            (1, 2),
            (len(prompts[0]), len(prompts[1])),
            tokens[:2],
        )

        second = pipeline.submit(
            (3, 4),
            (len(prompts[2]), len(prompts[3])),
            tokens[2:],
        )

        first_logits = pipeline.wait(first)
        second_logits = pipeline.wait(second)

        state = pipeline.snapshot()

        assert state["stage_calls"] == (
            2,
        ) * ranks

        assert state["completed_microbatches"] == 2
        assert state["reserved_request_ids"] == ()
        assert not state["failed"]

    with torch.inference_mode():
        references = [
            model(
                torch.tensor([[token]]),
                past_key_values=reference.past_key_values,
                use_cache=True,
            )
            for token, reference in zip(tokens, references)
        ]

        expected = torch.cat([
            reference.logits[:, -1, :]
            for reference in references
        ])

    actual = torch.cat(
        (first_logits, second_logits),
        dim=0,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    expected_positions = tuple(
        len(prompt) + 1
        for prompt in prompts
    )

    assert validate_stage_alignment(
        caches, (1, 2, 3, 4)
    ) == expected_positions

    # Check KV correctness independently for every
    # request and for each rank's owned layers.
    for stage, cache in zip(stages, caches):
        for rid, reference in enumerate(
            references, start=1
        ):
            stored = cache.read_request(rid)

            for local, global_layer in enumerate(
                stage.model.owned_layers
            ):
                actual_k, actual_v = stored[local]

                expected_layer = (
                    reference.past_key_values
                    .layers[global_layer]
                )

                torch.testing.assert_close(
                    actual_k,
                    expected_layer.keys,
                    rtol=1e-4,
                    atol=1e-4,
                )

                torch.testing.assert_close(
                    actual_v,
                    expected_layer.values,
                    rtol=1e-4,
                    atol=1e-4,
                )

    for rid in range(1, 5):
        for cache in caches:
            assert cache.release_request(rid)

    assert all(
        cache.snapshot()["allocated_blocks"] == 0
        for cache in caches
    )


# Gate 7C.1: Planner + prefill + transport + KV lifecycle.

from llm_runtime.v3_2_nrank_cpu_runtime import (
    NRankCPUContinuousRuntime,
)


def greedy_reference(model, prompt, count):
    tokens = list(prompt)
    output = []

    with torch.inference_mode():
        for _ in range(count):
            logits = model(
                torch.tensor([tokens]),
                use_cache=False,
            ).logits[0, -1]

            token = int(logits.argmax())
            output.append(token)
            tokens.append(token)

    return tuple(output)


def runtime_config(active=4):
    return {
        "pipeline": {
            "memory": {
                "max_active_requests": active,
                "max_total_tokens_per_request": 32,
            }
        },
        "scheduler": {
            "mode": "continuous",
            "policy": "fcfs",
            "max_requests_per_batch": active,
            "max_batch_tokens": 32,
            "max_queue_requests": 8,
        },
    }


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_integrated_cpu_runtime(tmp_path, ranks):
    model = make_model()

    stages, caches = create_stages(
        model,
        tmp_path / "integrated-checkpoint",
        LAYOUTS[ranks],
    )

    prompt_a = (2, 5, 8)
    prompt_b = (7, 4, 12, 9, 3)

    with NRankCPUContinuousRuntime(
        stages,
        caches,
        runtime_config(),
        stop_token_ids=(),
    ) as runtime:
        a = runtime.submit(prompt_a, 4)
        b = runtime.submit(prompt_b, 3)

        for _ in range(16):
            if (
                runtime.result(a) is not None
                and runtime.result(b) is not None
            ):
                break

            runtime.advance()
        else:
            raise AssertionError(
                "Runtime did not complete requests"
            )

        assert runtime.result(a) == greedy_reference(
            model, prompt_a, 4
        )
        assert runtime.result(b) == greedy_reference(
            model, prompt_b, 3
        )

        state = runtime.snapshot()

        assert not state["failed"]
        assert state["transport"][
            "completed_microbatches"
        ] >= 2

        assert all(
            cache_state["allocated_blocks"] == 0
            for cache_state in state["kv_blocks"]
        )


def test_integrated_cpu_runtime_late_arrival(tmp_path):
    model = make_model()

    stages, caches = create_stages(
        model,
        tmp_path / "late-checkpoint",
        LAYOUTS[2],
    )

    prompt_a = (2, 5, 8)
    prompt_b = (7, 4, 12, 9)

    with NRankCPUContinuousRuntime(
        stages,
        caches,
        runtime_config(),
        stop_token_ids=(),
    ) as runtime:
        a = runtime.submit(prompt_a, 5)

        runtime.advance()  # A prefill
        runtime.advance()  # A decode

        b = runtime.submit(prompt_b, 3)

        for _ in range(16):
            if (
                runtime.result(a) is not None
                and runtime.result(b) is not None
            ):
                break

            runtime.advance()
        else:
            raise AssertionError(
                "Late-arrival requests did not drain"
            )

        assert runtime.result(a) == greedy_reference(
            model, prompt_a, 5
        )
        assert runtime.result(b) == greedy_reference(
            model, prompt_b, 3
        )

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


# Gate 7C.2B: Multiple independent decode batches in flight.

from threading import Event

from llm_runtime.v3_2_concurrent_nrank_runtime import (
    ConcurrentNRankCPURuntime,
)


def concurrent_config(active=4):
    cfg = runtime_config(active=active)
    cfg["scheduler"]["max_requests_per_batch"] = 1
    return cfg


def drain_concurrent(runtime, request_ids):
    for _ in range(80):
        if all(
            runtime.result(rid) is not None
            for rid in request_ids
        ):
            return

        runtime.collect_ready(wait=True, timeout=10)
        runtime.dispatch_ready()

    raise AssertionError(
        "Concurrent runtime did not drain"
    )


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_concurrent_nrank_runtime_parity(tmp_path, ranks):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "concurrent-checkpoint",
        LAYOUTS[ranks],
    )

    prompts = (
        (2, 5, 8),
        (7, 4, 12, 9, 3),
        (3, 9, 11, 17),
        (4, 8, 16, 32, 64, 96),
    )

    with ConcurrentNRankCPURuntime(
        stages,
        caches,
        concurrent_config(),
        max_pending=2,
    ) as runtime:

        ids = tuple(
            runtime.submit(prompt, 3)
            for prompt in prompts
        )

        first = runtime.dispatch_ready()

        assert first["pending"] == 2
        assert len(first["decode_sequences"]) == 2

        drain_concurrent(runtime, ids)

        for rid, prompt in zip(ids, prompts):
            assert runtime.result(rid) == greedy_reference(
                net, prompt, 3
            )

        state = runtime.snapshot()

        assert not state["failed"]
        assert state["pending_batches"] == ()
        assert state["outstanding_batch_ids"] == ()
        assert state["transport"]["completed_microbatches"] >= 4

        assert all(
            entry["allocated_blocks"] == 0
            for entry in state["kv_blocks"]
        )


def test_runtime_has_actual_stage_overlap(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "overlap-checkpoint",
        LAYOUTS[2],
    )

    with ConcurrentNRankCPURuntime(
        stages,
        caches,
        concurrent_config(active=2),
        max_pending=2,
        pipeline_policy="downstream_first",
    ) as runtime:

        entered_stage1 = Event()
        ran_second_stage0 = Event()

        original0, original1 = runtime.transport._functions

        def stage0(packet):
            if packet.sequence == 2:
                assert entered_stage1.wait(5), (
                    "Stage 1 did not start batch 1"
                )
                ran_second_stage0.set()
            return original0(packet)

        def stage1(packet):
            if packet.sequence == 1:
                entered_stage1.set()
                assert ran_second_stage0.wait(5), (
                    "Stage 0 did not overlap batch 2"
                )
            return original1(packet)

        runtime.transport._functions = (
            stage0, stage1
        )

        a = runtime.submit((2, 5, 8), 2)
        b = runtime.submit((7, 4, 12), 2)

        runtime.dispatch_ready()
        drain_concurrent(runtime, (a, b))

        assert entered_stage1.is_set()
        assert ran_second_stage0.is_set()

        assert runtime.result(a) == greedy_reference(
            net, (2, 5, 8), 2
        )
        assert runtime.result(b) == greedy_reference(
            net, (7, 4, 12), 2
        )


def test_runtime_inflight_cancel(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "cancel-checkpoint",
        LAYOUTS[2],
    )

    with ConcurrentNRankCPURuntime(
        stages,
        caches,
        concurrent_config(active=2),
        max_pending=2,
    ) as runtime:

        a = runtime.submit((2, 5, 8), 4)
        b = runtime.submit((7, 4, 12), 3)

        runtime.dispatch_ready()

        assert runtime.cancel(a)

        for _ in range(20):
            runtime.collect_ready(wait=True, timeout=10)
            runtime.dispatch_ready()

            if runtime.result(b) is not None:
                break

        assert runtime.result(a) is None

        assert runtime.result(b) == greedy_reference(
            net, (7, 4, 12), 3
        )

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


def test_runtime_eos_stops_and_releases(tmp_path):
    net = make_model()
    prompt = (2, 5, 8)

    with torch.inference_mode():
        first_token = int(
            net(torch.tensor([prompt]))
            .logits[0, -1].argmax()
        )

    stages, caches = create_stages(
        net,
        tmp_path / "eos-checkpoint",
        LAYOUTS[2],
    )

    with ConcurrentNRankCPURuntime(
        stages,
        caches,
        concurrent_config(active=1),
        max_pending=2,
        stop_token_ids=(first_token,),
    ) as runtime:

        rid = runtime.submit(prompt, 5)
        runtime.dispatch_ready()

        assert runtime.result(rid) == (first_token,)

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


def test_runtime_late_arrival(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "late-concurrent-checkpoint",
        LAYOUTS[2],
    )

    with ConcurrentNRankCPURuntime(
        stages,
        caches,
        concurrent_config(active=2),
        max_pending=2,
    ) as runtime:

        a = runtime.submit((2, 5, 8), 4)
        runtime.dispatch_ready()

        # B arrives while A still has outstanding decode.
        b = runtime.submit((7, 4, 12, 9), 3)

        drain_concurrent(runtime, (a, b))

        assert runtime.result(a) == greedy_reference(
            net, (2, 5, 8), 4
        )
        assert runtime.result(b) == greedy_reference(
            net, (7, 4, 12, 9), 3
        )

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


# Gate 7D.1: Batched/chunked prefill through N stages.

from llm_runtime.v3_2_qwen3_stage_prefill import (
    PrefillHidden,
    stage_block_prefill,
)


@pytest.mark.parametrize("ranks", [2, 3, 4])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_nrank_batched_chunked_prefill_parity(
    tmp_path, ranks, chunk_size
):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "batched-prefill-checkpoint",
        LAYOUTS[ranks],
    )

    prompts = {
        1: (2, 5, 8),
        2: (7, 4, 12, 9, 3),
        3: (3, 8, 11, 21),
    }

    progress = {rid: 0 for rid in prompts}
    completed_logits = {}

    with torch.inference_mode():
        references = {
            rid: net(
                torch.tensor([prompt]),
                use_cache=True,
            )
            for rid, prompt in prompts.items()
        }

        while any(
            progress[rid] < len(prompt)
            for rid, prompt in prompts.items()
        ):
            ids = tuple(
                rid for rid, prompt in prompts.items()
                if progress[rid] < len(prompt)
            )

            positions = tuple(
                progress[rid] for rid in ids
            )

            chunks = tuple(
                prompts[rid][
                    progress[rid]:
                    progress[rid] + (
                        chunk_size
                        if chunk_size is not None
                        else len(prompts[rid])
                    )
                ]
                for rid in ids
            )

            payload = chunks

            for stage, cache in zip(
                stages, caches
            ):
                payload = stage_block_prefill(
                    stage,
                    cache,
                    ids,
                    positions,
                    payload,
                )

            assert payload.shape == (
                len(ids), net.config.vocab_size
            )

            for index, (rid, chunk) in enumerate(
                zip(ids, chunks)
            ):
                progress[rid] += len(chunk)

                if progress[rid] == len(prompts[rid]):
                    completed_logits[rid] = (
                        payload[index].clone()
                    )

            # Every request was initialized on the first
            # iteration, and all stages must stay aligned.
            assert validate_stage_alignment(
                caches,
                tuple(prompts)
            ) == tuple(
                progress[rid] for rid in prompts
            )

        for rid, reference in references.items():
            torch.testing.assert_close(
                completed_logits[rid],
                reference.logits[0, -1, :],
                rtol=1e-4,
                atol=1e-4,
            )

        # The new prefill path must produce KV that the
        # existing block-backed decoder can consume.
        ids = tuple(prompts)

        decode_tokens = tuple(
            int(references[rid].logits[0, -1].argmax())
            for rid in ids
        )

        decode_payload = decode_tokens

        for stage, cache in zip(stages, caches):
            decode_payload = stage_block_decode(
                stage,
                cache,
                ids,
                decode_payload,
            )

        next_references = {
            rid: net(
                torch.tensor([[token]]),
                past_key_values=(
                    references[rid].past_key_values
                ),
                use_cache=True,
            )
            for rid, token in zip(ids, decode_tokens)
        }

        expected = torch.cat([
            next_references[rid].logits[:, -1, :]
            for rid in ids
        ])

        torch.testing.assert_close(
            decode_payload,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

        assert validate_stage_alignment(
            caches, ids
        ) == tuple(
            len(prompts[rid]) + 1 for rid in ids
        )

        # Check actual stored K/V for each owned layer.
        for stage, cache in zip(stages, caches):
            for rid in ids:
                stored = cache.read_request(rid)

                for local, global_layer in enumerate(
                    stage.model.owned_layers
                ):
                    actual_k, actual_v = stored[local]

                    layer = (
                        next_references[rid]
                        .past_key_values
                        .layers[global_layer]
                    )

                    torch.testing.assert_close(
                        actual_k, layer.keys,
                        rtol=1e-4,
                        atol=1e-4,
                    )

                    torch.testing.assert_close(
                        actual_v, layer.values,
                        rtol=1e-4,
                        atol=1e-4,
                    )

    for rid in prompts:
        for cache in caches:
            assert cache.release_request(rid)

    assert all(
        cache.snapshot()["allocated_blocks"] == 0
        for cache in caches
    )


def test_nrank_prefill_rejects_position_mismatch(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "prefill-position-checkpoint",
        LAYOUTS[2],
    )

    with pytest.raises(
        RuntimeError,
        match="position mismatch",
    ):
        stage_block_prefill(
            stages[0],
            caches[0],
            (1,),
            (1,),
            ((2, 5),),
        )

    assert caches[0].snapshot()["allocated_blocks"] == 0


def test_nrank_prefill_rejects_invalid_hidden(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "prefill-shape-checkpoint",
        LAYOUTS[2],
    )

    invalid = PrefillHidden(
        torch.zeros(1, 3, net.config.hidden_size),
        (2,),
    )

    with pytest.raises(ValueError):
        stage_block_prefill(
            stages[1],
            caches[1],
            (1,),
            (0,),
            invalid,
        )

    assert caches[1].snapshot()["allocated_blocks"] == 0


# Gate 7D.2A: Qwen3 mixed-phase policy transport.

@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_mixed_transport_real_qwen3_parity(
    tmp_path, ranks
):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "mixed-transport-checkpoint",
        LAYOUTS[ranks],
    )

    prompts = {
        1: (2, 5, 8),
        2: (7, 4, 12, 9, 3),
        3: (3, 9, 11, 17),
        4: (4, 8, 16, 32, 64, 96),
    }

    with torch.inference_mode():
        references = {
            rid: net(
                torch.tensor([prompt]),
                use_cache=True,
            )
            for rid, prompt in prompts.items()
        }

    workers = []

    for stage, cache in zip(stages, caches):

        def execute(packet, stage=stage, cache=cache):
            if packet.phase == "prefill":
                return stage_block_prefill(
                    stage,
                    cache,
                    packet.request_ids,
                    packet.positions,
                    packet.payload,
                )

            return stage_block_decode(
                stage,
                cache,
                packet.request_ids,
                packet.payload,
            )

        workers.append(execute)

    with PolicyCPUMicrobatchTransport(
        workers,
        max_pending=2,
    ) as transport:

        # Two variable-length prefill microbatches.
        a = transport.submit(
            (1, 2),
            (0, 0),
            (prompts[1], prompts[2]),
            phase="prefill",
        )

        b = transport.submit(
            (3, 4),
            (0, 0),
            (prompts[3], prompts[4]),
            phase="prefill",
        )

        prefill_logits = torch.cat(
            (transport.wait(a), transport.wait(b)),
            dim=0,
        )

        expected_prefill = torch.cat([
            references[rid].logits[:, -1, :]
            for rid in prompts
        ])

        torch.testing.assert_close(
            prefill_logits,
            expected_prefill,
            rtol=1e-4,
            atol=1e-4,
        )

        assert validate_stage_alignment(
            caches, (1, 2, 3, 4)
        ) == tuple(
            len(prompt) for prompt in prompts.values()
        )

        tokens = tuple(
            int(prefill_logits[i].argmax())
            for i in range(4)
        )

        # Decode uses the same stage workers and cache pools.
        a = transport.submit(
            (1, 2),
            (len(prompts[1]), len(prompts[2])),
            tokens[:2],
            phase="decode",
        )

        b = transport.submit(
            (3, 4),
            (len(prompts[3]), len(prompts[4])),
            tokens[2:],
            phase="decode",
        )

        actual = torch.cat(
            (transport.wait(a), transport.wait(b)),
            dim=0,
        )

        state = transport.snapshot()
        assert state["stage_calls"] == (4,) * ranks
        assert not state["failed"]

    with torch.inference_mode():
        next_references = {
            rid: net(
                torch.tensor([[tokens[rid - 1]]]),
                past_key_values=(
                    references[rid].past_key_values
                ),
                use_cache=True,
            )
            for rid in prompts
        }

    expected = torch.cat([
        next_references[rid].logits[:, -1, :]
        for rid in prompts
    ])

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )

    assert validate_stage_alignment(
        caches, (1, 2, 3, 4)
    ) == tuple(
        len(prompt) + 1
        for prompt in prompts.values()
    )

    for stage, cache in zip(stages, caches):
        for rid in prompts:
            stored = cache.read_request(rid)

            for local, global_layer in enumerate(
                stage.model.owned_layers
            ):
                actual_k, actual_v = stored[local]
                reference_layer = (
                    next_references[rid]
                    .past_key_values.layers[global_layer]
                )

                torch.testing.assert_close(
                    actual_k,
                    reference_layer.keys,
                    rtol=1e-4,
                    atol=1e-4,
                )
                torch.testing.assert_close(
                    actual_v,
                    reference_layer.values,
                    rtol=1e-4,
                    atol=1e-4,
                )

    for rid in prompts:
        for cache in caches:
            assert cache.release_request(rid)

    assert all(
        cache.snapshot()["allocated_blocks"] == 0
        for cache in caches
    )


# Gate 7D.2B: Integrated mixed-phase continuous runtime.

from llm_runtime.v3_2_mixed_nrank_runtime import (
    MixedNRankCPURuntime,
)


def mixed_config(active=4, batch_tokens=12, batch_size=None):
    cfg = runtime_config(active=active)
    cfg["scheduler"]["max_batch_tokens"] = batch_tokens
    cfg["scheduler"]["max_requests_per_batch"] = (
        min(2, active) if batch_size is None else batch_size
    )
    return cfg


def drain_mixed(runtime, ids, max_iterations=100):
    for _ in range(max_iterations):
        if all(
            runtime.result(rid) is not None
            for rid in ids
        ):
            return

        runtime.dispatch_ready()

        if runtime.snapshot()["pending_batches"]:
            runtime.collect_ready(
                wait=True,
                timeout=15,
            )
        else:
            raise AssertionError(
                "Runtime has unfinished requests but no work"
            )

    raise AssertionError(
        "Mixed-phase runtime failed to drain"
    )


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_integrated_mixed_phase_parity(tmp_path, ranks):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "mixed-runtime-checkpoint",
        LAYOUTS[ranks],
    )

    prompts = (
        (2, 5, 8),
        (7, 4, 12, 9, 3),
        (3, 9, 11, 17),
        (4, 8, 16, 32, 64, 96),
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        mixed_config(),
        max_pending=2,
        stop_token_ids=(),
    ) as runtime:

        ids = tuple(
            runtime.submit(prompt, 3)
            for prompt in prompts
        )

        first = runtime.dispatch_ready()

        # Two independently scheduled prefill microbatches.
        assert len(first["submitted"]) == 2
        assert all(
            phase == "prefill"
            for _, phase, _ in first["submitted"]
        )

        drain_mixed(runtime, ids)

        for rid, prompt in zip(ids, prompts):
            assert runtime.result(rid) == greedy_reference(
                net, prompt, 3
            )

        state = runtime.snapshot()

        assert not state["failed"]
        assert state["pending_batches"] == ()
        assert state["outstanding_batch_ids"] == ()
        assert state["transport"][
            "completed_microbatches"
        ] >= 4

        assert state["transport"]["stage_calls"][0] >= 4

        assert all(
            entry["allocated_blocks"] == 0
            for entry in state["kv_blocks"]
        )


def test_integrated_chunked_prefill(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "chunked-runtime-checkpoint",
        LAYOUTS[2],
    )

    prompts = (
        (2, 5, 8, 11, 13, 17, 19),
        (7, 4, 12, 9),
    )

    cfg = mixed_config(
        active=2,
        batch_tokens=3,
        batch_size=2,
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        cfg,
        max_pending=2,
    ) as runtime:

        ids = tuple(
            runtime.submit(prompt, 3)
            for prompt in prompts
        )

        # These prompts cannot fit in a single prefill.
        assert max(map(len, prompts)) > 3

        drain_mixed(runtime, ids)

        for rid, prompt in zip(ids, prompts):
            assert runtime.result(rid) == greedy_reference(
                net, prompt, 3
            )

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


def test_actual_mixed_phase_stage_overlap(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "mixed-overlap-checkpoint",
        LAYOUTS[2],
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        mixed_config(
            active=2,
            batch_size=1,
        ),
        max_pending=2,
        pipeline_policy="downstream_first",
    ) as runtime:

        stage1_decode = Event()
        stage0_prefill = Event()

        original0, original1 = (
            runtime.transport._functions
        )

        a = runtime.submit((2, 5, 8), 3)

        # Finish A's initial prefill.
        runtime.dispatch_ready()
        runtime.collect_ready(wait=True, timeout=15)

        assert runtime.result(a) is None

        def guarded_stage0(packet):
            if (
                packet.phase == "prefill"
                and b in packet.request_ids
            ):
                assert stage1_decode.wait(10)
                stage0_prefill.set()

            return original0(packet)

        def guarded_stage1(packet):
            if (
                packet.phase == "decode"
                and a in packet.request_ids
            ):
                stage1_decode.set()
                assert stage0_prefill.wait(10)

            return original1(packet)

        runtime.transport._functions = (
            guarded_stage0,
            guarded_stage1,
        )

        # Dispatch A's decode, leaving it in flight.
        runtime.dispatch_ready()

        # B arrives while A is decoding.
        b = runtime.submit((7, 4, 12, 9), 3)

        dispatched = runtime.dispatch_ready()

        assert any(
            phase == "prefill" and b in ids
            for _, phase, ids in dispatched["submitted"]
        )

        drain_mixed(runtime, (a, b))

        assert stage1_decode.is_set()
        assert stage0_prefill.is_set()

        assert runtime.result(a) == greedy_reference(
            net, (2, 5, 8), 3
        )
        assert runtime.result(b) == greedy_reference(
            net, (7, 4, 12, 9), 3
        )


def test_cancel_during_prefill_defers_kv_release(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "cancel-prefill-checkpoint",
        LAYOUTS[2],
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        mixed_config(
            active=1,
            batch_size=1,
        ),
        max_pending=2,
    ) as runtime:

        entered = Event()
        proceed = Event()

        original0, original1 = (
            runtime.transport._functions
        )

        def delayed_stage1(packet):
            if packet.phase == "prefill":
                entered.set()
                assert proceed.wait(10)

            return original1(packet)

        runtime.transport._functions = (
            original0,
            delayed_stage1,
        )

        rid = runtime.submit((2, 5, 8), 5)

        try:
            runtime.dispatch_ready()
            assert entered.wait(10)

            # Stage 0 has already committed local KV,
            # but Stage 1 has not completed this request.
            assert (
                caches[0].snapshot()["allocated_blocks"]
                > 0
            )

            assert runtime.cancel(rid)

            # Cancellation must not release Stage 0 KV yet.
            assert (
                caches[0].snapshot()["allocated_blocks"]
                > 0
            )

        finally:
            proceed.set()

        runtime.collect_ready(wait=True, timeout=15)

        assert runtime.result(rid) is None
        assert runtime.snapshot()["pending_batches"] == ()

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


def test_mixed_runtime_eos_cleanup(tmp_path):
    net = make_model()
    prompt = (2, 5, 8)

    eos = greedy_reference(net, prompt, 1)[0]

    stages, caches = create_stages(
        net,
        tmp_path / "mixed-eos-checkpoint",
        LAYOUTS[2],
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        mixed_config(active=1),
        max_pending=2,
        stop_token_ids=(eos,),
    ) as runtime:

        rid = runtime.submit(prompt, 5)
        drain_mixed(runtime, (rid,))

        assert runtime.result(rid) == (eos,)

        assert all(
            cache.snapshot()["allocated_blocks"] == 0
            for cache in caches
        )


def test_no_duplicate_inflight_request(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "request-ownership-checkpoint",
        LAYOUTS[2],
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        mixed_config(
            active=2,
            batch_size=1,
        ),
        max_pending=2,
    ) as runtime:

        a = runtime.submit((2, 5, 8), 2)
        b = runtime.submit((7, 4, 12), 2)

        first = runtime.dispatch_ready()
        second = runtime.dispatch_ready()

        assert len(first["submitted"]) == 2
        assert second["submitted"] == ()

        pending = runtime.snapshot()[
            "pending_request_ids"
        ]

        request_ids = tuple(
            rid
            for ids in pending.values()
            for rid in ids
        )

        assert set(request_ids) == {a, b}
        assert len(request_ids) == len(set(request_ids))

        drain_mixed(runtime, (a, b))


def test_stage_failure_poison_requires_restart(tmp_path):
    net = make_model()

    stages, caches = create_stages(
        net,
        tmp_path / "mixed-failure-checkpoint",
        LAYOUTS[2],
    )

    with MixedNRankCPURuntime(
        stages,
        caches,
        mixed_config(active=1),
        max_pending=2,
    ) as runtime:

        original0, _ = runtime.transport._functions

        def failing_stage(packet):
            raise ValueError(
                "Injected stage execution failure"
            )

        runtime.transport._functions = (
            original0,
            failing_stage,
        )

        runtime.submit((2, 5, 8), 3)
        runtime.dispatch_ready()

        with pytest.raises(RuntimeError):
            runtime.collect_ready(
                wait=True,
                timeout=15,
            )

        assert runtime.snapshot()["failed"]

        with pytest.raises(RuntimeError, match="restart"):
            runtime.dispatch_ready()

        # Do not recycle this runtime's potentially
        # inconsistent KV state after a stage failure.
