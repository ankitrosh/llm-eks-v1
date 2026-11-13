"""End-to-end continuous Qwen3 CPU runtime tests."""

from threading import Event, Thread

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from llm_runtime.v3_2_continuous_engine import (
    ContinuousQwen3Engine,
)


def make_model():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=127,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def settings(active=2, batch=2, tokens=16, queue=4):
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
            "max_requests_per_batch": min(active, batch),
            "max_batch_tokens": tokens,
            "max_queue_requests": queue,
        },
    }


def reference(model, prompt, count):
    tokens = list(prompt)
    output = []

    with torch.inference_mode():
        for _ in range(count):
            logits = model(
                torch.tensor([tokens]),
                use_cache=False,
            ).logits[0, -1, :]

            token = int(logits.argmax())
            output.append(token)
            tokens.append(token)

    return tuple(output)


def finish(engine):
    for _ in range(32):
        state = engine.snapshot()
        if not state["waiting"] and not state["running"]:
            return
        engine.advance()
    raise AssertionError("Engine did not drain")


def test_real_batched_prefill_and_decode():
    model = make_model()
    engine = ContinuousQwen3Engine(
        model, settings(), stop_token_ids=()
    )

    a = engine.submit([2, 5, 8], 2)
    b = engine.submit([7, 4, 12, 9, 3], 2)

    finish(engine)

    assert engine.result(a) == reference(
        model, [2, 5, 8], 2
    )
    assert engine.result(b) == reference(
        model, [7, 4, 12, 9, 3], 2
    )

    state = engine.snapshot()
    assert state["model_forwards"] == 2
    assert state["batched_forwards"] == 2
    assert state["cache_request_ids"] == ()


def test_late_arrival_joins_batched_decode():
    model = make_model()
    engine = ContinuousQwen3Engine(
        model, settings(), stop_token_ids=()
    )

    a = engine.submit([2, 5, 8], 5)

    engine.advance()  # A prefill
    engine.advance()  # A decode

    b = engine.submit([7, 4, 12, 9, 3, 6], 3)

    update = engine.advance()
    assert set(update.emitted) == {b}

    update = engine.advance()
    assert set(update.emitted) == {a, b}

    finish(engine)

    assert engine.result(a) == reference(
        model, [2, 5, 8], 5
    )
    assert engine.result(b) == reference(
        model, [7, 4, 12, 9, 3, 6], 3
    )

    assert engine.snapshot()["batched_forwards"] >= 1


def test_chunked_prefill_preserves_correctness():
    model = make_model()
    engine = ContinuousQwen3Engine(
        model, settings(active=1, tokens=3),
        stop_token_ids=(),
    )

    prompt = [2, 5, 8, 11, 13, 17, 19]
    rid = engine.submit(prompt, 3)

    finish(engine)

    assert engine.result(rid) == reference(
        model, prompt, 3
    )
    assert (
        engine.snapshot()["sequential_prefill_forwards"]
        >= 1
    )


def test_waiting_request_can_be_cancelled():
    model = make_model()
    engine = ContinuousQwen3Engine(
        model, settings(active=1, queue=1),
        stop_token_ids=(),
    )

    a = engine.submit([2, 5], 3)
    b = engine.submit([7, 4], 2)

    assert engine.snapshot()["waiting"] == (b,)
    assert engine.cancel(b)

    finish(engine)

    assert engine.result(a) == reference(
        model, [2, 5], 3
    )
    assert engine.result(b) is None
    assert engine.snapshot()["cache_request_ids"] == ()


def test_eos_releases_cache():
    model = make_model()
    engine = ContinuousQwen3Engine(
        model,
        settings(active=1),
        sampler=lambda logits: 7,
        stop_token_ids=(7,),
    )

    rid = engine.submit([2, 5], 8)
    engine.advance()

    assert engine.result(rid) == (7,)
    assert engine.snapshot()["cache_request_ids"] == ()
    assert engine.snapshot()["reserved_kv_tokens"] == 0


def test_inflight_cancellation_defers_cache_release():
    model = make_model()
    engine = ContinuousQwen3Engine(
        model, settings(active=1),
        stop_token_ids=(),
    )

    rid = engine.submit([2, 5, 8], 4)

    entered = Event()
    proceed = Event()
    errors = []

    original = model.forward

    def blocking_forward(*args, **kwargs):
        entered.set()
        if not proceed.wait(timeout=10):
            raise TimeoutError("Test forward was not resumed")
        return original(*args, **kwargs)

    model.forward = blocking_forward

    def worker():
        try:
            engine.advance()
        except Exception as exc:
            errors.append(exc)

    thread = Thread(target=worker)

    try:
        thread.start()
        assert entered.wait(timeout=10)

        assert engine.cancel(rid)
        assert engine.snapshot()["in_flight"] == (rid,)

    finally:
        proceed.set()
        thread.join(timeout=10)
        model.forward = original

    assert not thread.is_alive()
    assert not errors, errors
    assert engine.result(rid) is None
    assert engine.snapshot()["cache_request_ids"] == ()
    assert engine.snapshot()["reserved_kv_tokens"] == 0
