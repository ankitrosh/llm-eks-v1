"""CPU-only tests for V3.1 N-rank ordered transport."""

import queue
import socket
import threading
import unittest

from collections import defaultdict
from dataclasses import dataclass

from llm_runtime.v3_1_nrank_transport import (
    NRankCoordinator,
    NRankTransportError,
    open_nrank_control,
    run_nrank_worker,
)


@dataclass
class FakeTensor:
    value: int
    shape: tuple = (1, 1)
    dtype: str = "float32"


class FakeBus:
    def __init__(self):
        self.queues = defaultdict(queue.Queue)

    def backend(self, rank):
        bus = self

        class Backend:
            def describe(self, tensor):
                return {
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                }

            def send(self, tensor, dst):
                bus.queues[(rank, dst)].put(tensor)

            def recv(self, src, spec):
                tensor = bus.queues[
                    (src, rank)
                ].get(timeout=5)

                if self.describe(tensor) != spec:
                    raise NRankTransportError(
                        "Tensor specification mismatch"
                    )

                return tensor

        return Backend()


class NRankTransportTests(unittest.TestCase):

    def run_case(self, world_size):
        bus = FakeBus()
        peers = {}
        workers = []
        stats = {}
        errors = []

        def execute(
            rank, request_id, step,
            position, tensor, cache
        ):
            expected_cache = (
                None if step == 0 else step
            )

            if cache != expected_cache:
                raise AssertionError(
                    f"Rank {rank}: incorrect cache state"
                )

            output = FakeTensor(
                tensor.value + rank * 10
            )

            return output, step + 1

        for rank in range(1, world_size):
            master, client = socket.socketpair()

            master.settimeout(10)
            client.settimeout(10)

            peers[rank] = master

            def worker(r=rank, conn=client):
                try:
                    stats[r] = run_nrank_worker(
                        r,
                        world_size,
                        conn,
                        bus.backend(r),
                        execute,
                    )
                except BaseException as exc:
                    errors.append((r, exc))

            thread = threading.Thread(
                target=worker,
                daemon=True,
            )

            thread.start()
            workers.append(thread)

        link = NRankCoordinator(
            peers,
            bus.backend(0),
            max_pending=5,
            timeout_s=10,
        )

        try:
            for step in range(3):
                tickets = [
                    link.publish(
                        request_id,
                        step,
                        step,
                        FakeTensor(request_id),
                    )
                    for request_id in range(1, 4)
                ]

                for request_id, ticket in enumerate(
                    tickets, 1
                ):
                    result = link.await_result(ticket)

                    expected = (
                        request_id
                        + 10 * sum(range(1, world_size))
                    )

                    self.assertEqual(
                        result.value,
                        expected,
                    )

            for request_id in range(1, 4):
                link.release_request(request_id)

        finally:
            link.close()

        for thread in workers:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        self.assertEqual(errors, [])

        for rank in range(1, world_size):
            self.assertEqual(stats[rank]["runs"], 9)
            self.assertEqual(stats[rank]["releases"], 3)
            self.assertEqual(stats[rank]["peak_caches"], 3)

        metrics = link.stats()

        self.assertEqual(metrics["sent"], 9)
        self.assertEqual(metrics["received"], 9)
        self.assertEqual(metrics["releases"], 3)

    def test_three_rank_transport(self):
        self.run_case(3)

    def test_four_rank_transport(self):
        self.run_case(4)

    def test_invalid_topology(self):
        with self.assertRaises(ValueError):
            NRankCoordinator(
                {2: object()},
                object(),
            )

    def test_rank_handshake(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        result = {}
        errors = []

        def listener():
            try:
                result["peers"] = open_nrank_control(
                    0,
                    3,
                    "127.0.0.1",
                    port,
                    timeout_s=5,
                )
            except BaseException as exc:
                errors.append(exc)

        server = threading.Thread(
            target=listener,
            daemon=True,
        )
        server.start()

        clients = []

        try:
            for rank in (2, 1):
                clients.append(
                    open_nrank_control(
                        rank,
                        3,
                        "127.0.0.1",
                        port,
                        timeout_s=5,
                    )
                )

            server.join(timeout=5)

            self.assertFalse(server.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(
                set(result["peers"]),
                {1, 2},
            )

        finally:
            for conn in clients:
                conn.close()

            for conn in result.get(
                "peers", {}
            ).values():
                conn.close()


if __name__ == "__main__":
    unittest.main()
