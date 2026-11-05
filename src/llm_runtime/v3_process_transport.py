"""CPU process transport for the existing V3 pipeline scheduler.

One stage per process. KV caches are owned by child processes.
Messages are pickled; GPU activations will need NCCL later.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from queue import Empty
from threading import Lock
from time import monotonic
from typing import Any, Callable

from .v3_pipeline_scheduler import (
    FatalStageError,
    StageResult,
    StageTask,
)


class RemoteStageError(RuntimeError):
    """Recoverable request-specific error from a remote stage."""


@dataclass(frozen=True)
class _Command:
    kind: str
    request_id: int = -1
    step: int = -1
    position: int = -1
    payload: Any = None


@dataclass(frozen=True)
class _Reply:
    kind: str
    request_id: int = -1
    step: int = -1
    payload: Any = None
    error: str = ""


@dataclass(frozen=True)
class RemoteCacheHandle:
    rank: int
    request_id: int


def _worker_main(rank, factory, inbox, outbox):
    # Model/stage construction happens inside the child.
    try:
        execute = factory(rank)
        if not callable(execute):
            raise TypeError("Factory must return a callable")
    except Exception as exc:
        outbox.put(_Reply(
            "INIT_ERROR",
            error=f"{type(exc).__name__}: {exc}",
        ))
        return

    outbox.put(_Reply("READY", payload=os.getpid()))

    caches = {}
    next_step = {}
    trace = []
    runs = 0
    peak_caches = 0
    releases = 0

    while True:
        cmd = inbox.get()

        if cmd.kind == "STOP":
            return

        if cmd.kind == "RELEASE":
            caches.pop(cmd.request_id, None)
            next_step.pop(cmd.request_id, None)
            releases += 1
            continue

        if cmd.kind == "STATS":
            outbox.put(_Reply("STATS", payload={
                "rank": rank,
                "pid": os.getpid(),
                "runs": runs,
                "cache_entries": len(caches),
                "peak_caches": peak_caches,
                "releases": releases,
                "trace": list(trace),
            }))
            continue

        if cmd.kind != "RUN":
            outbox.put(_Reply(
                "FATAL", cmd.request_id, cmd.step,
                error=f"Unknown command: {cmd.kind}",
            ))
            continue

        started = monotonic()
        cache_present = cmd.request_id in caches

        try:
            expected = next_step.get(cmd.request_id, 0)

            if cmd.step != expected:
                raise FatalStageError(
                    "Out-of-order request step"
                )

            if (cmd.step > 0) != cache_present:
                raise FatalStageError(
                    "Missing or duplicate request cache"
                )

            task = StageTask(
                cmd.request_id,
                cmd.step,
                cmd.position,
                cmd.payload,
            )

            result = execute(
                task,
                caches.get(cmd.request_id),
            )

            if not isinstance(result, StageResult):
                raise TypeError(
                    "Stage must return StageResult"
                )

            caches[cmd.request_id] = result.cache_state
            next_step[cmd.request_id] = cmd.step + 1

            runs += 1
            peak_caches = max(peak_caches, len(caches))

            trace.append({
                "request_id": cmd.request_id,
                "step": cmd.step,
                "position": cmd.position,
                "cache_present": cache_present,
                "started": started,
                "finished": monotonic(),
            })

            if len(trace) > 256:
                del trace[:-256]

            outbox.put(_Reply(
                "RESULT",
                cmd.request_id,
                cmd.step,
                payload=result.payload,
            ))

        except Exception as exc:
            kind = (
                "FATAL"
                if isinstance(exc, FatalStageError)
                else "ERROR"
            )

            outbox.put(_Reply(
                kind,
                cmd.request_id,
                cmd.step,
                error=f"{type(exc).__name__}: {exc}",
            ))


class _ProcessStageClient:

    def __init__(self, rank, factory, context, timeout_s):
        self.rank = rank
        self.timeout_s = timeout_s
        self._lock = Lock()
        self._send_lock = Lock()

        self._inbox = context.Queue()
        self._outbox = context.Queue()

        self._process = context.Process(
            target=_worker_main,
            args=(
                rank,
                factory,
                self._inbox,
                self._outbox,
            ),
            name=f"v3-cpu-rank-{rank}",
            daemon=True,
        )

        self._process.start()

        ready = self._recv()

        if ready.kind != "READY":
            raise FatalStageError(
                f"Rank {rank} startup: {ready.error}"
            )

    def _send(self, command):
        # Preserve command ordering across scheduler threads.
        with self._send_lock:
            self._inbox.put_nowait(command)

    def _recv(self):
        deadline = monotonic() + self.timeout_s

        while monotonic() < deadline:
            remaining = max(
                0.001,
                deadline - monotonic(),
            )

            try:
                return self._outbox.get(
                    timeout=min(0.1, remaining)
                )
            except Empty:
                if not self._process.is_alive():
                    raise FatalStageError(
                        f"Rank {self.rank} process exited"
                    )

        raise FatalStageError(
            f"Rank {self.rank} response timeout"
        )

    def execute(self, task, cache_handle):
        expected = RemoteCacheHandle(
            self.rank,
            task.request_id,
        )

        if (
            cache_handle is not None
            and cache_handle != expected
        ):
            raise FatalStageError(
                "Wrong request cache handle"
            )

        with self._lock:
            if not self._process.is_alive():
                raise FatalStageError(
                    f"Rank {self.rank} process exited"
                )

            self._send(_Command(
                "RUN",
                task.request_id,
                task.step,
                task.position,
                task.payload,
            ))

            reply = self._recv()

            if (
                reply.request_id,
                reply.step,
            ) != (
                task.request_id,
                task.step,
            ):
                raise FatalStageError(
                    "Response ordering mismatch"
                )

            if reply.kind == "FATAL":
                raise FatalStageError(
                    f"Rank {self.rank}: {reply.error}"
                )

            if reply.kind == "ERROR":
                raise RemoteStageError(
                    f"Rank {self.rank}: {reply.error}"
                )

            if reply.kind != "RESULT":
                raise FatalStageError(
                    f"Unexpected response: {reply.kind}"
                )

            return StageResult(
                reply.payload,
                expected,
            )

    def release(self, request_id):
        # Safe to call from the scheduler's completion hook.
        # The child processes RELEASE after earlier RUNs.
        if self._process.is_alive():
            self._send(_Command(
                "RELEASE",
                request_id=request_id,
            ))

    def stats(self):
        with self._lock:
            if not self._process.is_alive():
                raise FatalStageError(
                    f"Rank {self.rank} process exited"
                )

            self._send(_Command("STATS"))
            reply = self._recv()

            if reply.kind != "STATS":
                raise FatalStageError(
                    "Unexpected stats response"
                )

            return reply.payload

    def close(self):
        if self._process.is_alive():
            self._send(_Command("STOP"))
            self._process.join(self.timeout_s)

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2)

        if self._process.is_alive():
            self._process.kill()
            self._process.join(2)

        self._inbox.close()
        self._outbox.close()


class ProcessStageTransport:
    """CPU worker-process transport for PipelineScheduler.

    factory(rank) constructs a callable inside each child.

    The scheduler still owns request ordering and lifecycle.
    Workers own the actual stage-local cache objects.
    """

    def __init__(
        self,
        factory: Callable,
        *,
        num_stages=2,
        timeout_s=10.0,
    ):
        if num_stages < 1 or timeout_s <= 0:
            raise ValueError(
                "Invalid transport configuration"
            )

        context = mp.get_context("spawn")
        self._clients = []

        try:
            for rank in range(num_stages):
                self._clients.append(
                    _ProcessStageClient(
                        rank,
                        factory,
                        context,
                        timeout_s,
                    )
                )
        except BaseException:
            self.close()
            raise

    def stage_functions(self):
        return [
            client.execute
            for client in self._clients
        ]

    def release_request(self, request_id):
        for client in self._clients:
            client.release(request_id)

    def stats(self):
        return tuple(
            client.stats()
            for client in self._clients
        )

    def close(self):
        for client in reversed(self._clients):
            client.close()

        self._clients.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
