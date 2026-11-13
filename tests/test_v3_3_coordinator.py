import json
import threading
import unittest

from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_runtime.coordination.vllm_coordinator import (
    CoordinatorServer,
)


class MockServer(ThreadingHTTPServer):
    daemon_threads = True


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        elif self.path == "/v1/models":
            data = b'{"data":[{"id":"test"}]}'
            self.send_response(200)
            self.send_header(
                "Content-Length", str(len(data))
            )
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_error(404)

    def do_POST(self):
        length = int(
            self.headers["Content-Length"]
        )

        body = json.loads(
            self.rfile.read(length)
        )

        if body.get("stream"):
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream"
            )
            self.end_headers()
            self.close_connection = True

            self.wfile.write(
                b'data: {"text":"ok"}\n\n'
            )
            self.wfile.flush()

            self.wfile.write(
                b"data: [DONE]\n\n"
            )
            self.wfile.flush()

        else:
            data = (
                b'{"choices":[{"text":"ok"}]}'
            )

            self.send_response(200)
            self.send_header(
                "Content-Type", "application/json"
            )
            self.send_header(
                "Content-Length", str(len(data))
            )
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *args):
        pass


class RecoveryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({
            "mode": "RUNNING",
            "phase": "SERVING",
        }).encode()

        self.send_response(200)
        self.send_header(
            "Content-Length", str(len(data))
        )
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestVLLMGateway(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.servers = []

        upstream = MockServer(
            ("127.0.0.1", 0),
            UpstreamHandler,
        )

        recovery = MockServer(
            ("127.0.0.1", 0),
            RecoveryHandler,
        )

        gateway = CoordinatorServer(
            ("127.0.0.1", 0),
            upstream=(
                f"http://127.0.0.1:"
                f"{upstream.server_port}"
            ),
            recovery=(
                f"http://127.0.0.1:"
                f"{recovery.server_port}"
            ),
            require_recovery=True,
        )

        cls.base = (
            f"http://127.0.0.1:"
            f"{gateway.server_port}"
        )

        for server in (
            upstream, recovery, gateway
        ):
            thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
            )
            thread.start()
            cls.servers.append((server, thread))

    @classmethod
    def tearDownClass(cls):
        for server, thread in cls.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_ready(self):
        with urlopen(
            self.base + "/readyz", timeout=5
        ) as response:
            self.assertTrue(
                json.load(response)["ready"]
            )

    def test_models(self):
        with urlopen(
            self.base + "/v1/models", timeout=5
        ) as response:
            self.assertEqual(
                json.load(response)["data"][0]["id"],
                "test",
            )

    def test_proxy(self):
        request = Request(
            self.base + "/v1/chat/completions",
            data=b'{"messages":[]}',
            headers={
                "Content-Type": "application/json"
            },
        )

        with urlopen(
            request, timeout=5
        ) as response:
            self.assertEqual(
                json.load(response)["choices"][0]["text"],
                "ok",
            )

    def test_stream(self):
        request = Request(
            self.base + "/v1/chat/completions",
            data=b'{"messages":[],"stream":true}',
            headers={
                "Content-Type": "application/json"
            },
        )

        with urlopen(
            request, timeout=5
        ) as response:
            self.assertEqual(
                response.headers.get(
                    "Transfer-Encoding"
                ),
                "chunked",
            )

            self.assertIn(
                b"data: [DONE]",
                response.read(),
            )

    def test_z_recovery_fails_closed(self):
        # Intentionally last: stops the mock
        # recovery service to simulate failure.
        self.servers[1][0].shutdown()

        with self.assertRaises(
            HTTPError
        ) as error:
            urlopen(
                self.base + "/readyz", timeout=5
            )

        self.assertEqual(
            error.exception.code, 503
        )


if __name__ == "__main__":
    unittest.main()
