"""End-to-end continuous inference with physical KV blocks."""

from threading import Event, Thread

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import llm_runtime.v3_2_block_engine as block_module
from llm_runtime.v3_2_block_engine import (
    BlockBackedQwen3Engine,
)


def model():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    cfg = Qwen3Config(
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
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def config(active=2, tokens=16):
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
            "max_batch_tokens": tokens,
            "max_queue_requests": 4,
        },
    }


def reference(net, prompt, steps):
    history = list(prompt)
    generated = []

    with torch.inference_mode():
        for _ in range(steps):
            logits = net(
                torch.tensor([history]),
                use_cache=False,
            ).logits[0, -1]

            token = int(logits.argmax())
            generated.append(token)
            history.append(token)

    return tuple(generated)


def drain(engine):
    for _ in range(40):
        state = engine.snapshot()
        if not state["running"] and not state["waiting"]:
            return
        engine.advance()

    raise AssertionError("Engine failed to drain")


def assert_released(engine):
    state = engine.snapshot()
    assert state["cache_request_ids"] == ()
    assert state["kv_blocks"]["allocated_blocks"] == 0
    assert state["reserved_kv_tokens"] == 0


def test_variable_length_block_batches():
    net = model()
    engine = BlockBackedQwen3Engine(
        net, config(), stop_token_ids=(),
    )

    a = engine.submit([2, 5, 8], 3)
    b = engine.submit([7, 4, 12, 9, 3, 6], 3)

    drain(engine)

    assert engine.result(a) == reference(
        net, [2, 5, 8], 3
    )
    assert engine.result(b) == reference(
        net, [7, 4, 12, 9, 3, 6], 3
    )

    assert engine.snapshot()["batched_forwards"] >= 3
    assert_released(engine)


def test_late_arrival_block_decode():
    net = model()
    engine = BlockBackedQwen3Engine(
        net, config(), stop_token_ids=(),
    )

    a = engine.submit([2, 5, 8], 5)
    engine.advance()
    engine.advance()

    b = engine.submit([7, 4, 12, 9, 3, 6], 3)

    engine.advance()  # B prefill
    update = engine.advance()  # A and B decode

    assert set(update.emitted) == {a, b}

    drain(engine)

    assert engine.result(a) == reference(
        net, [2, 5, 8], 5
    )
    assert engine.result(b) == reference(
        net, [7, 4, 12, 9, 3, 6], 3
    )
    assert_released(engine)


def test_chunked_prefill_migrates_on_completion():
    net = model()
    engine = BlockBackedQwen3Engine(
        net, config(active=1, tokens=3),
        stop_token_ids=(),
    )

    prompt = [2, 5, 8, 11, 13, 17, 19]
    rid = engine.submit(prompt, 3)

    engine.advance()
    assert engine.snapshot()["kv_blocks"]["requests"] == 0

    engine.advance()
    assert engine.snapshot()["kv_blocks"]["requests"] == 0

    engine.advance()
    assert engine.snapshot()["kv_blocks"]["requests"] == 1

    drain(engine)

    assert engine.result(rid) == reference(net, prompt, 3)
    assert_released(engine)


def test_decode_never_materializes_dynamic_cache():
    net = model()
    engine = BlockBackedQwen3Engine(
        net, config(active=1), stop_token_ids=(),
    )

    rid = engine.submit([2, 5, 8], 2)
    engine.advance()  # Prefill and migration

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Contiguous DynamicCache execution is forbidden"
        )

    previous_forward = net.forward
    previous_read = engine.block_cache.read_request

    net.forward = forbidden
    engine.block_cache.read_request = forbidden

    try:
        engine.advance()
    finally:
        net.forward = previous_forward
        engine.block_cache.read_request = previous_read

    assert engine.result(rid) is not None
    assert_released(engine)


def test_eos_and_waiting_cancellation():
    net = model()
    engine = BlockBackedQwen3Engine(
        net,
        config(active=1),
        sampler=lambda logits: 7,
        stop_token_ids=(7,),
    )

    a = engine.submit([2, 5], 8)
    b = engine.submit([7, 4], 2)

    assert engine.cancel(b)
    engine.advance()

    assert engine.result(a) == (7,)
    assert engine.result(b) is None
    assert_released(engine)


def test_inflight_block_decode_cancellation(monkeypatch):
    net = model()
    engine = BlockBackedQwen3Engine(
        net, config(active=1), stop_token_ids=(),
    )

    rid = engine.submit([2, 5, 8], 4)
    engine.advance()

    entered = Event()
    proceed = Event()
    errors = []

    original = block_module.qwen3_block_decode_step

    def blocked(*args, **kwargs):
        entered.set()
        if not proceed.wait(timeout=10):
            raise TimeoutError("Decode was not resumed")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        block_module, "qwen3_block_decode_step", blocked
    )

    def run():
        try:
            engine.advance()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run)

    try:
        worker.start()
        assert entered.wait(timeout=10)

        assert engine.cancel(rid)
        assert engine.snapshot()["in_flight"] == (rid,)
    finally:
        proceed.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert not errors, errors
    assert engine.result(rid) is None
    assert_released(engine)
