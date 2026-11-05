"""Forward-only inference scheduler with cooperative cancellation.

Each stage has one worker. KV caches are per-request and stage-local.
Running model code is never forcibly interrupted.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from itertools import count
from threading import Condition, Thread
from time import monotonic
from typing import Any, Callable


class SchedulerBusy(RuntimeError):
    pass


class FatalStageError(RuntimeError):
    """Irrecoverable stage or transport failure; restart the pipeline."""


class PipelineUnhealthyError(RuntimeError):
    """Reject all new work after a fatal stage error."""


@dataclass(frozen=True)
class StageTask:
    request_id: int
    step: int
    position: int
    payload: Any
    enqueued_at: float = field(default_factory=monotonic, compare=False)


@dataclass
class StageResult:
    payload: Any
    cache_state: Any = None


@dataclass(frozen=True)
class GenerationResult:
    request_id: int
    tokens: tuple[Any, ...]


@dataclass(frozen=True)
class StageEvent:
    at: float
    stage: int
    request_id: int
    step: int
    event: str


@dataclass
class _Request:
    request_id: int
    prompt_length: int
    max_new_tokens: int
    future: Future
    deadline: float | None
    tokens: list[Any] = field(default_factory=list)
    expired: bool = False
    submitted_at: float = field(default_factory=monotonic)


class PipelineScheduler:
    """Run stage_fn(task, cache) -> StageResult per stage.

    Future.cancel() immediately removes queued work.
    In-flight work finishes before its cache is released.
    timeout_s expires the Future without preempting model execution.
    """

    def __init__(
        self,
        stage_fns: list[Callable[[StageTask, Any], StageResult]],
        sample_fn: Callable[[Any], Any],
        *,
        max_active_requests: int = 4,
        stop_fn: Callable[[Any], bool] | None = None,
        trace_limit: int = 2048,
    ):
        if not stage_fns or not all(callable(f) for f in stage_fns):
            raise ValueError("At least one callable stage required")
        if not callable(sample_fn):
            raise TypeError("sample_fn must be callable")
        if max_active_requests < 1 or trace_limit < 0:
            raise ValueError("Invalid scheduler limits")

        self._stages = list(stage_fns)
        self._sample = sample_fn
        self._stop = stop_fn
        self._capacity = max_active_requests

        self._queues = [deque() for _ in stage_fns]
        self._caches = [{} for _ in stage_fns]
        self._running = [None for _ in stage_fns]
        self._requests = {}
        self._ids = count(1)
        self._accepting = True
        self._fatal_error = None
        self._started_at = monotonic()
        self._stats = {
            "submitted_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "cancelled_requests": 0,
            "timed_out_requests": 0,
            "generated_tokens": 0,
            "first_tokens": 0,
            "total_latency_s": 0.0,
            "total_ttft_s": 0.0,
        }
        self._stage_stats = [
            {"tasks": 0, "queue_wait_s": 0.0, "run_s": 0.0}
            for _ in stage_fns
        ]
        self._cv = Condition()
        self._events = deque(maxlen=trace_limit)

        self._workers = [
            Thread(
                target=self._worker,
                args=(i,),
                name=f"v3-stage-{i}",
                daemon=True,
            )
            for i in range(len(stage_fns))
        ]

        self._watchdog = Thread(
            target=self._deadlines,
            name="v3-deadlines",
            daemon=True,
        )

        for worker in self._workers:
            worker.start()
        self._watchdog.start()

    def submit(
        self,
        prompt,
        *,
        prompt_length,
        max_new_tokens,
        timeout_s=None,
    ):
        if prompt_length < 1 or max_new_tokens < 1:
            raise ValueError("Token counts must be positive")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        with self._cv:
            if self._fatal_error is not None:
                raise PipelineUnhealthyError(self._fatal_error)
            if not self._accepting:
                raise RuntimeError("Scheduler closed")

            if len(self._requests) >= self._capacity:
                raise SchedulerBusy("Request capacity reached")

            rid = next(self._ids)
            future = Future()
            deadline = (
                monotonic() + timeout_s
                if timeout_s is not None
                else None
            )

            self._requests[rid] = _Request(
                rid,
                prompt_length,
                max_new_tokens,
                future,
                deadline,
            )

            future.add_done_callback(
                lambda f, rid=rid: self._on_done(rid, f)
            )

            self._queues[0].append(
                StageTask(rid, 0, 0, prompt)
            )
            self._stats["submitted_requests"] += 1
            self._cv.notify_all()
            return future

    def snapshot(self):
        with self._cv:
            return {
                "active_requests": len(self._requests),
                "queued_by_stage": tuple(
                    map(len, self._queues)
                ),
                "busy_by_stage": tuple(
                    r is not None for r in self._running
                ),
                "cache_entries_by_stage": tuple(
                    map(len, self._caches)
                ),
                "events": tuple(self._events),
                "healthy": self._fatal_error is None,
                "fatal_error": self._fatal_error,
            }

    def metrics(self):
        """Cumulative counters; TTFT = first sampled token."""
        with self._cv:
            m = dict(self._stats)
            m["healthy"] = self._fatal_error is None
            m["fatal_error"] = self._fatal_error
            m["active"] = len(self._requests)

            m["tokens_per_second"] = (
                m["generated_tokens"] /
                max(monotonic() - self._started_at, 1e-9)
            )

            m["avg_latency_ms"] = (
                1000 * m["total_latency_s"] / m["completed_requests"]
                if m["completed_requests"] else None
            )

            m["avg_ttft_ms"] = (
                1000 * m["total_ttft_s"] / m["first_tokens"]
                if m["first_tokens"] else None
            )

            m["stages"] = tuple({
                "tasks": v["tasks"],
                "avg_queue_wait_ms": (
                    1000 * v["queue_wait_s"] / v["tasks"]
                    if v["tasks"] else None
                ),
                "avg_execution_ms": (
                    1000 * v["run_s"] / v["tasks"]
                    if v["tasks"] else None
                ),
            } for v in self._stage_stats)

            return m

    def _event(self, rank, task, kind):
        self._events.append(
            StageEvent(
                monotonic(),
                rank,
                task.request_id,
                task.step,
                kind,
            )
        )

    def _forget(self, rid):
        """Requires scheduler lock and no active stage for rid."""
        assert rid not in self._running

        self._requests.pop(rid, None)

        for queue in self._queues:
            queued = [
                t for t in queue
                if t.request_id != rid
            ]
            queue.clear()
            queue.extend(queued)

        for cache in self._caches:
            cache.pop(rid, None)

        self._cv.notify_all()

    def _on_done(self, rid, future):
        if future.cancelled():
            with self._cv:
                state = self._requests.get(rid)
                if state is not None and not state.expired:
                    self._stats["cancelled_requests"] += 1
                if (
                    rid in self._requests
                    and rid not in self._running
                ):
                    self._forget(rid)

                self._cv.notify_all()

    def _deadlines(self):
        while True:
            expired = []

            with self._cv:
                if not self._accepting and not self._requests:
                    return

                now = monotonic()
                next_deadline = None

                for state in list(self._requests.values()):
                    if (
                        state.deadline is None
                        or state.expired
                        or state.future.done()
                    ):
                        continue

                    if state.deadline <= now:
                        state.expired = True
                        self._stats["timed_out_requests"] += 1
                        expired.append(state.future)

                        if state.request_id not in self._running:
                            self._forget(state.request_id)
                    elif (
                        next_deadline is None
                        or state.deadline < next_deadline
                    ):
                        next_deadline = state.deadline

                if not expired:
                    wait = (
                        0.25
                        if next_deadline is None
                        else min(
                            0.25,
                            max(0.0, next_deadline - now),
                        )
                    )
                    self._cv.wait(wait)

            for future in expired:
                try:
                    future.set_exception(
                        TimeoutError("Request deadline exceeded")
                    )
                except InvalidStateError:
                    # Cancellation/completion won the race.
                    pass

    def _trip_fatal(self, rank, error):
        """Fail requests but preserve caches of running stage code."""
        self._fatal_error = f"stage {rank}: {error}"
        self._accepting = False
        pending = []

        for state in list(self._requests.values()):
            if not state.expired and not state.future.done():
                pending.append(state.future)
                self._stats["failed_requests"] += 1

            if state.request_id not in self._running:
                self._forget(state.request_id)

        self._cv.notify_all()
        return pending

    def _worker(self, rank):
        while True:
            with self._cv:
                while not self._queues[rank]:
                    if (
                        not self._accepting
                        and not self._requests
                    ):
                        return
                    self._cv.wait()

                task = self._queues[rank].popleft()
                state = self._requests.get(task.request_id)

                if (
                    state is None
                    or state.expired
                    or state.future.done() or self._fatal_error is not None
                ):
                    if (
                        state is not None
                        and task.request_id not in self._running
                    ):
                        self._forget(task.request_id)
                    continue

                self._running[rank] = task.request_id
                cache = self._caches[rank].get(task.request_id)
                started = monotonic()
                self._stage_stats[rank]["queue_wait_s"] += max(
                    0.0, started - task.enqueued_at
                )
                self._event(rank, task, "start")

            error = None
            result = None
            sampled = None
            should_stop = False

            try:
                result = self._stages[rank](task, cache)

                if not isinstance(result, StageResult):
                    raise TypeError("Stage must return StageResult")

                if rank == len(self._stages) - 1:
                    sampled = self._sample(result.payload)
                    should_stop = (
                        bool(self._stop(sampled))
                        if self._stop else False
                    )
            except Exception as exc:
                error = exc

            completion = None
            failed_futures = []
            with self._cv:
                self._running[rank] = None
                self._stage_stats[rank]["tasks"] += 1
                self._stage_stats[rank]["run_s"] += (
                    monotonic() - started
                )
                self._event(
                    rank,
                    task,
                    "error" if error else "finish",
                )

                state = self._requests.get(task.request_id)

                if isinstance(error, FatalStageError) and self._fatal_error is None:
                    failed_futures = self._trip_fatal(rank, error)
                    if task.request_id in self._requests:
                        self._forget(task.request_id)
                elif state is None:
                    pass

                elif state.expired or state.future.done() or self._fatal_error is not None:
                    self._forget(task.request_id)

                elif error is not None:
                    completion = (state.future, error, True)
                    self._stats["failed_requests"] += 1
                    self._forget(task.request_id)

                else:
                    self._caches[rank][
                        task.request_id
                    ] = result.cache_state

                    if rank < len(self._stages) - 1:
                        self._queues[rank + 1].append(
                            StageTask(
                                task.request_id,
                                task.step,
                                task.position,
                                result.payload,
                            )
                        )
                    else:
                        state.tokens.append(sampled)
                        self._stats["generated_tokens"] += 1

                        if len(state.tokens) == 1:
                            self._stats["first_tokens"] += 1
                            self._stats["total_ttft_s"] += (
                                monotonic() - state.submitted_at
                            )

                        if (
                            should_stop
                            or len(state.tokens)
                            >= state.max_new_tokens
                        ):
                            completion = (
                                state.future,
                                GenerationResult(
                                    task.request_id,
                                    tuple(state.tokens),
                                ),
                                False,
                            )
                            self._stats["completed_requests"] += 1
                            self._stats["total_latency_s"] += (
                                monotonic() - state.submitted_at
                            )
                            self._forget(task.request_id)
                        else:
                            self._queues[0].append(
                                StageTask(
                                    task.request_id,
                                    task.step + 1,
                                    state.prompt_length + task.step,
                                    sampled,
                                )
                            )

                self._cv.notify_all()

            for future in failed_futures:
                try:
                    future.set_exception(
                        PipelineUnhealthyError(self._fatal_error)
                    )
                except InvalidStateError:
                    pass

            if completion is not None:
                future, value, is_error = completion

                try:
                    if is_error:
                        future.set_exception(value)
                    else:
                        future.set_result(value)
                except InvalidStateError:
                    # Cancelled or timed out concurrently.
                    pass

    def close(self, timeout=10.0):
        """Stop admission and drain outstanding requests."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")

        with self._cv:
            self._accepting = False
            self._cv.notify_all()

        deadline = (
            None if timeout is None
            else monotonic() + timeout
        )

        threads = self._workers + [self._watchdog]

        for worker in threads:
            remaining = (
                None if deadline is None
                else max(0, deadline - monotonic())
            )
            worker.join(remaining)

        if any(worker.is_alive() for worker in threads):
            raise TimeoutError(
                "Workers did not drain; "
                "stage code may be blocked"
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
