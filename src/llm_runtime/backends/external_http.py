"""Minimal HTTP transport for an external inference backend."""

import json
from http.server import BaseHTTPRequestHandler


def handler_for(adapter):
    class Handler(BaseHTTPRequestHandler):

        def respond(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header(
                "Content-Type", "application/json"
            )
            self.send_header(
                "Content-Length", str(len(data))
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "Not found"})
                return

            ready = adapter.health()
            self.respond(
                200 if ready else 503,
                {"ready": ready},
            )

        def do_POST(self):
            if self.path != "/v1/completions":
                self.respond(404, {"error": "Not found"})
                return

            if not adapter.health():
                self.respond(503, {"error": "Backend unavailable"})
                return

            try:
                size = int(
                    self.headers.get("Content-Length", "0")
                )
                if not 0 < size <= 1048576:
                    raise ValueError("Invalid request size")

                request = json.loads(
                    self.rfile.read(size)
                )

                if not isinstance(request, dict):
                    raise ValueError("Expected JSON object")

                if request.get("stream") is True:
                    raise NotImplementedError(
                        "Streaming is not supported by this backend"
                    )

                result = adapter.generate(request)

            except NotImplementedError as exc:
                self.respond(422, {"error": str(exc)})
                return
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
                return
            except Exception:
                self.respond(
                    500, {"error": "Backend execution failed"}
                )
                return

            self.respond(200, result)

        def log_message(self, *args):
            pass

    return Handler
