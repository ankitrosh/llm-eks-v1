import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RANK_URLS = [
    url.strip().rstrip("/")
    for url in os.environ["RANK_URLS"].split(",")
]
UPSTREAM = os.environ["UPSTREAM_URL"].rstrip("/")
PORT = int(os.environ.get("PORT", "8080"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))
HEALTH_TIMEOUT = 2.0

if len(RANK_URLS) != 2:
    raise ValueError("V1 requires exactly two rank endpoints")


def check_pipeline():
    ranks = []

    for rank, base_url in enumerate(RANK_URLS):
        result = {"rank": rank, "ready": False}

        try:
            with urlopen(
                base_url + "/readyz",
                timeout=HEALTH_TIMEOUT,
            ) as response:
                result["ready"] = response.status == 200
                result["http_status"] = response.status

        except HTTPError as exc:
            result["http_status"] = exc.code
            result["error"] = str(exc)

        except Exception as exc:
            result["error"] = str(exc)

        ranks.append(result)

    controller = {"phase": "UNAVAILABLE", "ready": False}

    try:
        with urlopen(
            "http://qwen3-recovery.llm.svc.cluster.local:8081/status",
            timeout=HEALTH_TIMEOUT,
        ) as response:
            status = json.load(response)

        controller = {
            "phase": status.get("phase", "UNKNOWN"),
            "generation": status.get("generation"),
            "ready": status.get("phase") == "SERVING",
        }

    except Exception as exc:
        controller["error"] = str(exc)

    return {
        "ready": (
            all(rank["ready"] for rank in ranks)
            and controller["ready"]
        ),
        "ranks": ranks,
        "controller": controller,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_bytes(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        body = json.dumps(value).encode("utf-8")
        self.send_bytes(status, body)

    def do_GET(self):
        if self.path == "/healthz":
            return self.send_json(200, {"status": "alive"})

        if self.path in ("/readyz", "/status"):
            status = check_pipeline()

            if self.path == "/readyz":
                code = 200 if status["ready"] else 503
            else:
                code = 200

            return self.send_json(code, status)

        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self.send_json(404, {"error": "not found"})

        pipeline = check_pipeline()

        if not pipeline["ready"]:
            return self.send_json(
                503,
                {
                    "error": "distributed pipeline unavailable",
                    "pipeline": pipeline,
                },
            )

        try:
            size = int(self.headers.get("Content-Length", "0"))

            if size <= 0 or size > 1024 * 1024:
                return self.send_json(
                    413, {"error": "invalid request body size"}
                )

            payload = self.rfile.read(size)

            request = Request(
                UPSTREAM + "/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                body = response.read(8 * 1024 * 1024 + 1)

                if len(body) > 8 * 1024 * 1024:
                    return self.send_json(
                        502, {"error": "upstream response too large"}
                    )

                self.send_bytes(response.status, body)

        except HTTPError as exc:
            self.send_bytes(exc.code, exc.read(1024 * 1024))

        except (URLError, TimeoutError, OSError) as exc:
            self.send_json(
                502,
                {"error": "inference upstream failed", "detail": str(exc)},
            )


print(f"Coordinator listening on port {PORT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
