"""CPU-only end-to-end custom backend integration test."""

import json
import threading
import urllib.error
import urllib.request

from pathlib import Path
from http.server import ThreadingHTTPServer

from llm_runtime.extensions import (
    PluginRegistry,
    load_plugin_context,
)
from llm_runtime.extensions.external import (
    load_external_extensions,
)
from llm_runtime.backends.external_adapter import (
    BackendContractError,
    ExternalBackendAdapter,
)
from llm_runtime.backends.external_http import handler_for


root = Path(
    "infrastructure/profiles/v3-3/qwen3-14b"
)

plan = json.loads(
    (root / "execution_plan.json").read_text()
)

context = load_plugin_context(
    model_capabilities_path=root / "model_capabilities.json",
    model_layout_path=root / "model_layout.json",
    backend="research_pytorch",
    runtime={"execution_plan": plan},
)

registry = PluginRegistry()

load_external_extensions(
    registry,
    ["examples.v3_4_1.custom_backend"],
)

selected = registry.build(
    "backend",
    "user-test-backend",
    context,
)

# Reject implementations that don't fulfill our interface.
try:
    ExternalBackendAdapter(object(), context, plan)
    raise AssertionError("Broken backend was accepted")
except BackendContractError:
    print("PASS: Incomplete backend rejected")

adapter = ExternalBackendAdapter(
    selected, context, plan
)

server = None
thread = None

try:
    adapter.start()
    assert adapter.health()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(adapter),
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    url = (
        f"http://127.0.0.1:{server.server_port}"
        "/v1/completions"
    )

    request_body = {
        "model": "user-test-backend",
        "prompt": "Hello from custom backend",
        "max_tokens": 16,
        "stream": False,
    }

    def send(payload):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(
            request, timeout=10
        )

    with send(request_body) as response:
        result = json.load(response)

    output = result["choices"][0]["text"]

    assert output == (
        "USER_BACKEND_EXECUTED: "
        "Hello from custom backend"
    )
    assert selected.calls == 1

    print("PASS: User factory selected")
    print("PASS: Backend started through adapter")
    print("PASS: HTTP request reached user implementation")
    print("Response:", output)

    unsupported = dict(request_body)
    unsupported["stream"] = True

    try:
        send(unsupported)
        raise AssertionError(
            "Unsupported streaming request was accepted"
        )
    except urllib.error.HTTPError as exc:
        assert exc.code == 422

    print("PASS: Unsupported streaming rejected explicitly")

finally:
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None:
        thread.join(timeout=5)
    adapter.stop()

assert not adapter.health()
print("PASS: Backend shutdown")
print("V341_CUSTOM_BACKEND_INTEGRATION_PASS")
