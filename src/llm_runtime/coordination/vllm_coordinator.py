"""V3.3 vLLM gateway. CPU-only; Python standard library."""

import hmac
import json
import os

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CoordinatorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, address, *,
        upstream, recovery,
        require_recovery=True,
        api_key=None,
    ):
        self.upstream = upstream.rstrip("/")
        self.recovery = (
            recovery.rstrip("/") if recovery else None
        )
        self.require_recovery = require_recovery
        self.api_key = api_key

        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(
        self, status, data,
        content_type="application/json",
    ):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorized(self):
        secret = self.server.api_key

        if not secret:
            # Only suitable behind private ClusterIP
            # or another authenticated gateway.
            return True

        expected = "Bearer " + secret

        if hmac.compare_digest(
            self.headers.get("Authorization", ""),
            expected,
        ):
            return True

        self.reply(401, {"error": "unauthorized"})
        return False

    def health(self):
        result = {
            "ready": False,
            "vllm": {"ready": False},
            "recovery": None,
        }

        try:
            with urlopen(
                self.server.upstream + "/health",
                timeout=2,
            ) as response:
                result["vllm"]["ready"] = (
                    response.status == 200
                )

        except (
            HTTPError, URLError, OSError, TimeoutError
        ) as exc:
            result["vllm"]["error"] = str(exc)

        recovery_ready = (
            not self.server.require_recovery
        )

        if self.server.require_recovery:
            result["recovery"] = {"ready": False}

            if self.server.recovery:
                try:
                    with urlopen(
                        self.server.recovery + "/status",
                        timeout=2,
                    ) as response:
                        state = json.load(response)

                    recovery_ready = (
                        state.get("mode") == "RUNNING"
                        and state.get("phase") == "SERVING"
                    )

                    result["recovery"] = {
                        "ready": recovery_ready,
                        "mode": state.get("mode"),
                        "phase": state.get("phase"),
                    }

                except (
                    HTTPError, URLError, OSError,
                    TimeoutError, ValueError,
                ) as exc:
                    result["recovery"]["error"] = (
                        str(exc)
                    )

        result["ready"] = (
            result["vllm"]["ready"]
            and recovery_ready
        )

        return result

    def proxy(self, method, body=None):
        headers = {"Accept-Encoding": "identity"}

        if body is not None:
            headers["Content-Type"] = (
                "application/json"
            )

        if self.headers.get("Accept"):
            headers["Accept"] = self.headers["Accept"]

        if self.headers.get("Authorization"):
            headers["Authorization"] = (
                self.headers["Authorization"]
            )

        request = Request(
            self.server.upstream + self.path,
            data=body,
            headers=headers,
            method=method,
        )

        try:
            response = urlopen(
                request, timeout=300
            )
        except HTTPError as exc:
            response = exc
        except (
            URLError, TimeoutError, OSError
        ) as exc:
            return self.reply(
                502,
                {
                    "error": "upstream unavailable",
                    "detail": str(exc),
                },
            )

        with response:
            content_type = response.headers.get(
                "Content-Type", "application/json"
            )

            if content_type.lower().startswith(
                "text/event-stream"
            ):
                # Forward SSE without buffering the
                # entire generated response.
                self.send_response(response.status)
                self.send_header(
                    "Content-Type", content_type
                )
                self.send_header(
                    "Cache-Control", "no-cache"
                )
                self.send_header(
                    "Transfer-Encoding", "chunked"
                )
                self.end_headers()
                self.close_connection = True

                try:
                    while chunk := response.read1(65536):
                        header = (
                            f"{len(chunk):X}\r\n"
                        ).encode()

                        self.wfile.write(header)
                        self.wfile.write(
                            chunk + b"\r\n"
                        )
                        self.wfile.flush()

                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    OSError,
                ):
                    # Closing the upstream stream
                    # releases this proxy request.
                    pass

            else:
                content = response.read(
                    8 * 1024 * 1024 + 1
                )

                if len(content) > 8 * 1024 * 1024:
                    return self.reply(
                        502,
                        {
                            "error":
                                "upstream response too large"
                        },
                    )

                self.reply(
                    response.status,
                    content,
                    content_type,
                )

    def do_GET(self):
        if self.path == "/healthz":
            return self.reply(
                200, {"status": "alive"}
            )

        if not self.authorized():
            return

        if self.path in ("/readyz", "/status"):
            result = self.health()

            code = (
                200
                if self.path == "/status"
                or result["ready"]
                else 503
            )

            return self.reply(code, result)

        if self.path == "/v1/models":
            if not self.health()["ready"]:
                return self.reply(
                    503, {"error": "engine unavailable"}
                )

            return self.proxy("GET")

        self.reply(
            404, {"error": "unsupported route"}
        )

    def do_POST(self):
        if not self.authorized():
            return

        if self.path not in (
            "/v1/completions",
            "/v1/chat/completions",
        ):
            return self.reply(
                404, {"error": "unsupported route"}
            )

        if not self.health()["ready"]:
            return self.reply(
                503, {"error": "engine unavailable"}
            )

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )
        except ValueError:
            return self.reply(
                400, {"error": "invalid content length"}
            )

        if length < 1 or length > 1024 * 1024:
            return self.reply(
                413, {"error": "invalid request size"}
            )

        body = self.rfile.read(length)

        try:
            parsed = json.loads(body)
        except ValueError:
            return self.reply(
                400, {"error": "invalid JSON"}
            )

        if not isinstance(parsed, dict):
            return self.reply(
                400, {"error": "expected JSON object"}
            )

        self.proxy("POST", body)


def main():
    server = CoordinatorServer(
        (
            "0.0.0.0",
            int(os.getenv("PORT", "8080")),
        ),
        upstream=os.environ["VLLM_UPSTREAM_URL"],
        recovery=os.getenv(
            "RECOVERY_STATUS_URL",
            "http://qwen3-recovery.llm.svc.cluster.local:8081",
        ),
        require_recovery=(
            os.getenv(
                "REQUIRE_RECOVERY", "true"
            ).lower() == "true"
        ),
        api_key=os.getenv("GATEWAY_API_KEY"),
    )

    print(
        "V3.3 coordinator listening",
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
