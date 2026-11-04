"""Persistent two-rank Qwen3-14B NCCL inference server (greedy V1).

Rank 0 owns a FastAPI HTTP API and the request scheduler. Both ranks execute
CUDA and NCCL operations exclusively on their main thread. Rank 1 waits for
rank-0 control broadcasts; no NCCL operations occur in HTTP request threads.

Startup warm-up executes across BOTH ranks before marking the API ready.
The generated token ID is sent from rank 1 to rank 0 every decode step; both
ranks independently stop on the SAME EOS ID to preserve NCCL call ordering.
V1 supports one active request, a bounded queue, no streaming or KV recovery.
"""

import argparse
import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
import logging
import os
import queue
import signal
import statistics
import threading
import time
import uuid
from typing import Any

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from qwen3_stage_loader import Qwen3Stage
from prefix_cache import PrefixCache
from prefix_kv import snapshot_prefix, restore_prefix
from qwen3_nccl_generate import blocking_ms, forward_timed, now_ms

LOG = logging.getLogger("qwen3_serve")

# Control message is [opcode, prompt_len, output_len, eos_token_id].
# eos_token_id=-1 disables early exit (used for deterministic warm-up).
CMD_HEARTBEAT = 0
CMD_GENERATE = 1
CMD_STOP = 2
WORLD_SIZE = 2


_PREFIX_MANAGER = None


def _prefix_enabled() -> bool:
    return os.environ.get(
        "LLM_PREFIX_CACHE_ENABLED", "0"
    ).lower() in ("1", "true", "yes")


def _prefix_manager():
    """One cache per process; inference runs on the main thread."""
    global _PREFIX_MANAGER

    if not _prefix_enabled():
        return None

    if _PREFIX_MANAGER is None:
        _PREFIX_MANAGER = PrefixCache(
            max_bytes=int(os.environ.get(
                "LLM_PREFIX_CACHE_MAX_MIB", "256"
            )) * 1024 * 1024,
            max_entries=int(os.environ.get(
                "LLM_PREFIX_CACHE_MAX_ENTRIES", "4"
            )),
            min_reuse_tokens=int(os.environ.get(
                "LLM_PREFIX_CACHE_MIN_TOKENS", "64"
            )),
        )

    return _PREFIX_MANAGER


def _prefix_model_key(stage, rank):
    """Identify the loaded local model partition."""
    cfg = stage.cfg
    return (
        f"{getattr(cfg, '_name_or_path', 'unknown')}:"
        f"{cfg.num_hidden_layers}:{cfg.hidden_size}:"
        f"{stage.start}:{len(stage.layers)}:"
        f"{stage.dtype}:rank{rank}:"
        f"{os.environ.get('LLM_GPTQ_BACKEND', 'tritonv2')}"
    )


def _prepare_request_cache(stage, rank, prompt, prompt_len,
                           manager, empty_cache):
    """Agree on the reusable prefix across both NCCL ranks."""

    # Both ranks must have the feature enabled or disabled.
    flag = torch.tensor(
        [int(manager is not None)],
        device="cuda:0",
        dtype=torch.int64,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)

    if int(flag.item()) not in (0, WORLD_SIZE):
        raise RuntimeError(
            "LLM_PREFIX_CACHE_ENABLED differs between ranks"
        )

    if manager is None:
        return empty_cache, None, None, 0

    # Rank 1 needs the exact token IDs for cache lookup.
    # This transfers token IDs, not KV tensors.
    if rank == 0:
        ids_gpu = prompt.reshape(-1).contiguous()
    else:
        ids_gpu = torch.empty(
            prompt_len, device="cuda:0", dtype=torch.long
        )

    dist.broadcast(ids_gpu, src=0)
    token_ids = tuple(ids_gpu.cpu().tolist())

    model_key = _prefix_model_key(stage, rank)
    namespace = os.environ.get(
        "LLM_PREFIX_CACHE_NAMESPACE", "default"
    )

    hit = manager.lookup(
        token_ids,
        model_key=model_key,
        namespace=namespace,
    )
    local_reuse = hit.matched_tokens if hit else 0

    # The minimum guarantees both ranks have the entire prefix.
    agreed = torch.tensor(
        [local_reuse], device="cuda:0", dtype=torch.int64
    )
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    reuse = int(agreed.item())

    if reuse == 0:
        return empty_cache, token_ids, model_key, 0

    # If restoring fails on either stage, BOTH stages fall back.
    restored = None
    success = 1

    try:
        if hit is None:
            raise RuntimeError("Local prefix unavailable")
        restored = restore_prefix(
            stage.make_cache, hit.payload, reuse
        )
    except Exception as exc:
        success = 0
        LOG.warning(
            "rank=%s KV restore failed: %s", rank, exc
        )
        manager.clear()

    confirmed = torch.tensor(
        [success], device="cuda:0", dtype=torch.int64
    )
    dist.all_reduce(confirmed, op=dist.ReduceOp.MIN)

    if int(confirmed.item()) == 0:
        return empty_cache, token_ids, model_key, 0

    LOG.info("rank=%s reused %d prefix tokens", rank, reuse)

    return restored, token_ids, model_key, reuse


def _save_prompt_prefix(stage, rank, cache, token_ids,
                        model_key, manager):
    """Store prompt-only KV state, never generated-token state."""

    start = stage.start
    end = start + len(stage.layers)
    namespace = os.environ.get(
        "LLM_PREFIX_CACHE_NAMESPACE", "default"
    )

    try:
        # Estimate snapshot memory before cloning GPU tensors.
        estimated = 0
        for index in range(start, end):
            layer = cache.layers[index]
            if type(layer).__name__ != "DynamicLayer":
                return False
            if cache.get_seq_length(index) != len(token_ids):
                return False

            estimated += (
                layer.keys.numel() * layer.keys.element_size()
                + layer.values.numel() * layer.values.element_size()
            )

        if estimated > manager.max_bytes:
            return False

        # Preserve headroom for decoding and temporary allocations.
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < estimated + 512 * 1024 * 1024:
            LOG.info(
                "rank=%s skipping KV snapshot: memory headroom",
                rank,
            )
            return False

        snapshot = snapshot_prefix(
            cache, start, end, len(token_ids)
        )

        return manager.put(
            token_ids,
            snapshot,
            nbytes=snapshot.nbytes,
            model_key=model_key,
            namespace=namespace,
        )

    except Exception as exc:
        # A cache-storage failure must not corrupt active inference.
        LOG.warning(
            "rank=%s prefix snapshot skipped: %s", rank, exc
        )
        manager.clear()
        return False




class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=64, ge=1, le=4096)


@dataclass
class Job:
    id: str
    prompt: str
    output_len: int
    result: Future


class State:
    """Small state shared between the NCCL/main thread and the HTTP thread."""

    def __init__(self, rank: int, queue_capacity: int, max_new_tokens: int, timeout: float):
        self.rank = rank
        self.queue: queue.Queue[Job] = queue.Queue(maxsize=queue_capacity)
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.max_new_tokens = max_new_tokens
        self.request_timeout = timeout
        self.active = False
        self.completed = 0
        self.failed = 0
        self.last_e2e_ms: float | None = None
        self.last_error: str | None = None
        self.warmup_completed = False
        self.warmup_ms: float | None = None
        self.started_at = time.monotonic()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "rank": self.rank,
                "ready": self.ready.is_set() and not self.stop.is_set(),
                "active": self.active,
                "queue_depth": self.queue.qsize(),
                "completed_requests": self.completed,
                "failed_requests": self.failed,
                "last_e2e_ms": self.last_e2e_ms,
                "last_error": self.last_error,
                "warmup_completed": self.warmup_completed,
                "warmup_ms": self.warmup_ms,
                "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            }


def create_app(state: State) -> FastAPI:
    app = FastAPI(title="Qwen3-14B Pipeline V1", version="0.1.0")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        # Deliberately local health; startup and readiness are separate.
        return {"status": "alive"}

    @app.get("/readyz")
    def ready() -> dict[str, str]:
        if not state.ready.is_set() or state.stop.is_set():
            raise HTTPException(status_code=503, detail="distributed pipeline not ready")
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        # JSON diagnostics in V1, not Prometheus exposition format.
        return state.status()

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> dict[str, Any]:
        if state.rank != 0:
            raise HTTPException(status_code=404, detail="only rank 0 serves inference")
        if not state.ready.is_set() or state.stop.is_set():
            raise HTTPException(status_code=503, detail="distributed pipeline not ready")
        if req.max_new_tokens > state.max_new_tokens:
            raise HTTPException(
                status_code=422,
                detail=f"max_new_tokens cannot exceed {state.max_new_tokens} on this server",
            )

        future: Future = Future()
        job = Job(str(uuid.uuid4()), req.prompt, req.max_new_tokens, future)
        try:
            state.queue.put_nowait(job)
        except queue.Full:
            raise HTTPException(status_code=429, detail="inference queue full") from None

        try:
            # A cancelled client doesn't interrupt an already-running NCCL job;
            # that job safely runs to completion, then its response is discarded.
            return await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=state.request_timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="inference deadline exceeded") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def broadcast_control(rank: int, opcode: int = 0, prompt_len: int = 0,
                      output_len: int = 0, eos_token_id: int = -1):
    if rank == 0:
        control = torch.tensor(
            [opcode, prompt_len, output_len, eos_token_id],
            device="cuda:0", dtype=torch.int64
        )
    else:
        control = torch.empty(4, device="cuda:0", dtype=torch.int64)
    dist.broadcast(control, src=0)
    return tuple(int(x) for x in control.cpu().tolist())


def encode_chat_prompt(tokenizer, message: str) -> list[int]:
    """Qwen3 chat formatting; the same path is used for warm-up and requests."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def eos_reached(token_id: int, eos_token_id: int) -> bool:
    """Identical stopping predicate on both ranks, with -1 disabling EOS."""
    return eos_token_id >= 0 and token_id == eos_token_id


def forward_one_request(stage: Qwen3Stage, rank: int, prompt: torch.Tensor | None,
                        prompt_len: int, output_len: int, tokenizer, eos_token_id: int):
    """Benchmark NCCL prefill/decode protocol, with coordinated EOS stopping."""
    cache = stage.make_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = now_ms()
    last_completion = start
    generated: list[int] = []
    intervals: list[float] = []
    stage_compute_ms: list[float] = []
    stage_comm_ms: list[float] = []
    previous_token = None
    completed_steps = 0
    finish_reason = "length"

    prefix_manager = _prefix_manager()
    prefix_stored = False

    cache, prefix_ids, prefix_model_key, prefix_reused = (
        _prepare_request_cache(
            stage, rank, prompt, prompt_len,
            prefix_manager, cache,
        )
    )

    try:
        for step in range(output_len):
            position = prefix_reused if step == 0 else prompt_len + step - 1
            sequence_len = prompt_len - prefix_reused if step == 0 else 1

            if rank == 0:
                input_ids = prompt[:, prefix_reused:] if step == 0 else previous_token.view(1, 1)
                hidden, gpu_ms, _ = forward_timed(stage, input_ids, position, cache)
                stage_compute_ms.append(float(gpu_ms))
                stage_comm_ms.append(blocking_ms(lambda: dist.send(hidden, dst=1)))
                del hidden

                token = torch.empty(1, device="cuda:0", dtype=torch.int64)
                stage_comm_ms.append(blocking_ms(lambda: dist.recv(token, src=1)))
                previous_token = token
                selected_token_id = int(token.item())
                generated.append(selected_token_id)
            else:
                hidden = torch.empty(
                    (1, sequence_len, stage.cfg.hidden_size),
                    device="cuda:0", dtype=torch.bfloat16
                )
                stage_comm_ms.append(blocking_ms(lambda: dist.recv(hidden, src=0)))
                logits, gpu_ms, _ = forward_timed(stage, hidden, position, cache)
                stage_compute_ms.append(float(gpu_ms))
                del hidden
                token = logits.argmax(dim=-1).to(torch.int64).contiguous()
                stage_comm_ms.append(blocking_ms(lambda: dist.send(token, dst=0)))
                # Rank 1 sees the same selected token that Rank 0 just received.
                # Both independently break; no extra NCCL round trip is needed.
                selected_token_id = int(token.item())
                del token, logits

            # Snapshot immediately after prefill, before any
            # generated token is appended to the active KV cache.
            if step == 0 and prefix_manager is not None:
                prefix_stored = _save_prompt_prefix(
                    stage, rank, cache, prefix_ids,
                    prefix_model_key, prefix_manager,
                )

            completion = now_ms()
            if rank == 0:
                intervals.append(completion - last_completion)
            last_completion = completion
            completed_steps = step + 1
            if eos_reached(selected_token_id, eos_token_id):
                finish_reason = "stop"
                break

        torch.cuda.synchronize()
        total_ms = now_ms() - start
        expected_cache_len = prompt_len + completed_steps - 1
        actual_cache_len = cache.get_seq_length(stage.start)
        if actual_cache_len != expected_cache_len:
            raise RuntimeError(
                f"rank={rank} KV length mismatch: {actual_cache_len} != {expected_cache_len}"
            )

        LOG.info("rank=%s generated %s/%s tokens, reason=%s, compute=%.2fms, "
                 "total=%.2fms, peak=%.1fMiB",
                 rank, completed_steps, output_len, finish_reason,
                 sum(stage_compute_ms), total_ms,
                 torch.cuda.max_memory_allocated() / 2**20)

        if rank == 0:
            return {
                "text": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_ids": generated,
                "prompt_tokens": prompt_len,
                "generated_tokens": len(generated),
                "finish_reason": finish_reason,
                "metrics": {
                    "prefix_cache_enabled": prefix_manager is not None,
                    "prefix_cache_hit": prefix_reused > 0,
                    "prefix_cached_tokens": prefix_reused,
                    "prefix_cache_stored": prefix_stored,
                    "prefix_cache_entries": (
                        prefix_manager.size if prefix_manager else 0
                    ),
                    "prefix_cache_used_mib": (
                        round(prefix_manager.used_bytes / 2**20, 2)
                        if prefix_manager else 0
                    ),
                    "prefill_gpu_ms": round(stage_compute_ms[0], 3),
                    "e2e_ms": round(total_ms, 3),
                    "ttft_ms": round(intervals[0], 3),
                    "decode_interval_median_ms": (
                        round(statistics.median(intervals[1:]), 3)
                        if len(intervals) > 1 else None
                    ),
                    "stage0_compute_gpu_ms": round(sum(stage_compute_ms), 3),
                    "stage0_communication_blocking_ms": round(sum(stage_comm_ms), 3),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                },
            }
        return None
    finally:
        del cache


def run_rank0(state: State, stage: Qwen3Stage, tokenizer, heartbeat_seconds: float,
              max_prompt_tokens: int):
    with torch.inference_mode():
        while not state.stop.is_set():
            try:
                job = state.queue.get(timeout=heartbeat_seconds)
            except queue.Empty:
                broadcast_control(rank=0, opcode=CMD_HEARTBEAT)
                continue

            try:
                if job.result.cancelled():
                    continue
                # Only rank 0 tokenizes, and validation happens BEFORE the
                # generate broadcast so rank 1 remains synchronized on errors.
                ids = encode_chat_prompt(tokenizer, job.prompt)
                if not ids:
                    raise ValueError("prompt produced no tokens")
                if len(ids) > max_prompt_tokens:
                    raise ValueError(
                        f"prompt contains {len(ids)} tokens, limit is {max_prompt_tokens}"
                    )
                if len(ids) + job.output_len - 1 > stage.cfg.max_position_embeddings:
                    raise ValueError("prompt + generation exceeds model context limit")
            except ValueError as exc:
                with state.lock:
                    state.failed += 1
                    state.last_error = str(exc)
                if not job.result.done():
                    job.result.set_exception(exc)
                continue
            finally:
                # Queue slot is freed as soon as rank 0 starts processing.
                state.queue.task_done()

            with state.lock:
                state.active = True
            # Once this control is broadcast, any exception is fatal:
            # rank 1 would otherwise wait for a mismatched NCCL operation.
            broadcast_control(0, CMD_GENERATE, len(ids), job.output_len,
                              eos_token_id=tokenizer.eos_token_id)
            prompt = torch.tensor([ids], device="cuda:0", dtype=torch.long)
            result = forward_one_request(stage, 0, prompt, len(ids), job.output_len,
                                         tokenizer, tokenizer.eos_token_id)
            result["request_id"] = job.id
            if not job.result.done():
                job.result.set_result(result)
            with state.lock:
                state.active = False
                state.completed += 1
                state.last_e2e_ms = result["metrics"]["e2e_ms"]

        broadcast_control(0, CMD_STOP)


def run_rank1(state: State, stage: Qwen3Stage):
    with torch.inference_mode():
        while not state.stop.is_set():
            opcode, prompt_len, output_len, eos_token_id = broadcast_control(rank=1)
            if opcode == CMD_HEARTBEAT:
                continue
            if opcode == CMD_STOP:
                break
            if opcode != CMD_GENERATE:
                raise RuntimeError(f"Unknown control opcode {opcode}")
            if not (0 < prompt_len and 0 < output_len):
                raise RuntimeError("Invalid generation control message")
            if prompt_len + output_len - 1 > stage.cfg.max_position_embeddings:
                raise RuntimeError("Context exceeded on rank 1")
            forward_one_request(stage, 1, None, prompt_len, output_len, None,
                                eos_token_id)


def distributed_warmup(stage: Qwen3Stage, rank: int, tokenizer,
                       output_tokens: int, max_prompt_tokens: int) -> float:
    """Run one coordinated generation before starting HTTP/readiness.

    The -1 EOS sentinel guarantees a fixed number of warm-up decode steps.
    Both ranks then barrier so neither can start its service prematurely.
    """
    start = time.monotonic()
    with torch.inference_mode():
        if rank == 0:
            ids = encode_chat_prompt(tokenizer, "Explain why GPUs are useful.")
            if not ids or len(ids) > max_prompt_tokens:
                raise ValueError("Warm-up prompt exceeds configured prompt limits")
            if len(ids) + output_tokens - 1 > stage.cfg.max_position_embeddings:
                raise ValueError("Warm-up exceeds model context")
            LOG.info("Starting warm-up: prompt=%s tokens, output=%s tokens",
                     len(ids), output_tokens)
            broadcast_control(0, CMD_GENERATE, len(ids), output_tokens, -1)
            prompt = torch.tensor([ids], device="cuda:0", dtype=torch.long)
            forward_one_request(stage, 0, prompt, len(ids), output_tokens,
                                tokenizer, eos_token_id=-1)
        else:
            opcode, prompt_len, output_len, eos_id = broadcast_control(rank=1)
            if opcode != CMD_GENERATE or eos_id != -1 or output_len != output_tokens:
                raise RuntimeError("Unexpected warm-up control message")
            forward_one_request(stage, 1, None, prompt_len, output_len,
                                None, eos_token_id=-1)
        dist.barrier(device_ids=[0])
    elapsed_ms = (time.monotonic() - start) * 1000
    LOG.info("Rank %s warm-up complete in %.1fms", rank, elapsed_ms)
    return elapsed_ms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank", type=int, choices=(0, 1), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--split", type=int, default=20)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--queue-capacity", type=int, default=8)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--heartbeat-seconds", type=float, default=5.0)
    p.add_argument("--collective-timeout-seconds", type=int, default=60)
    p.add_argument("--warmup-output-tokens", type=int, default=16,
                   help="Decode steps before readiness; 0 disables warm-up")
    args = p.parse_args()
    if min(args.queue_capacity, args.max_prompt_tokens, args.max_new_tokens,
           args.request_timeout, args.heartbeat_seconds,
           args.collective_timeout_seconds) <= 0:
        p.error("All limits and timeouts must be positive")
    if args.warmup_output_tokens < 0:
        p.error("warmup-output-tokens must not be negative")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.cuda.set_device(0)
    state = State(args.rank, args.queue_capacity, args.max_new_tokens, args.request_timeout)

    def request_stop(signum, _frame):
        LOG.warning("Received signal %s; stopping after current operation", signum)
        state.stop.set()
        state.ready.clear()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda:0"),
            timeout=timedelta(seconds=args.collective_timeout_seconds),
        )
        if dist.get_world_size() != WORLD_SIZE or dist.get_rank() != args.rank:
            raise RuntimeError("Unexpected world size / rank")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16")

        LOG.info("Rank %s loading Qwen3 stage, split=%s", args.rank, args.split)
        from os import environ as _v3_environ

        stage_impl = _v3_environ.get(
            "LLM_STAGE_IMPL", "legacy"
        ).strip().lower()

        if stage_impl == "legacy":
            stage = Qwen3Stage(
                args.model_dir, args.rank, args.split
            ).load()

        elif stage_impl == "v3_adapter":
            from llm_runtime.v3_stage_bridge import (
                load_qwen3_stage_v3,
            )

            stage = load_qwen3_stage_v3(
                args.model_dir,
                args.rank,
                args.split,
            )

        else:
            raise ValueError(
                f"Unsupported LLM_STAGE_IMPL: {stage_impl}"
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir) if args.rank == 0 else None
        if tokenizer is not None and not isinstance(tokenizer.eos_token_id, int):
            raise RuntimeError("Expected one integer EOS token ID in Qwen3 tokenizer")
        torch.cuda.synchronize()
        dist.barrier(device_ids=[0])
        LOG.info("Rank %s model loaded and NCCL synchronized", args.rank)

        # Both ranks participate BEFORE HTTP readiness. A restart may spend
        # several seconds on kernel initialization; clients should not see it.
        if args.warmup_output_tokens:
            state.warmup_ms = round(distributed_warmup(
                stage, args.rank, tokenizer, args.warmup_output_tokens,
                args.max_prompt_tokens), 2)
            state.warmup_completed = True
        else:
            LOG.warning("GPU warm-up disabled (WARMUP_OUTPUT_TOKENS=0)")

        server = uvicorn.Server(uvicorn.Config(
            create_app(state), host=args.host, port=args.port,
            log_level="info", access_log=False,
        ))
        thread = threading.Thread(target=server.run, name="http-server", daemon=True)
        thread.start()
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError(f"Rank {args.rank}: HTTP server failed to start")
            time.sleep(0.05)
        state.ready.set()
        LOG.info("Rank %s READY HTTP :%s", args.rank, args.port)

        if args.rank == 0:
            run_rank0(state, stage, tokenizer, args.heartbeat_seconds, args.max_prompt_tokens)
        else:
            run_rank1(state, stage)
        state.ready.clear()
        server.should_exit = True
        thread.join(timeout=5)
        if dist.is_initialized():
            dist.destroy_process_group()
    except BaseException:
        # The only safe V1 response to any distributed error is to fail fast.
        # A Kubernetes restart or a manual two-rank relaunch rebuilds NCCL.
        state.ready.clear()
        state.stop.set()
        LOG.exception("Rank %s fatal error; exiting entire worker", args.rank)
        os._exit(1)


if __name__ == "__main__":
    main()
