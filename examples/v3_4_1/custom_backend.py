"""Example external execution backend.

This is a CPU-only test implementation, not an LLM.
"""

from llm_runtime.extensions.contracts import PluginSpec


class UserBackend:
    def __init__(self):
        self.running = False
        self.calls = 0

    def validate(self, context):
        if context.backend != "research_pytorch":
            raise ValueError("Unsupported execution backend")

    def prepare(self, execution_plan):
        if not isinstance(execution_plan, dict):
            raise ValueError("Invalid execution plan")

    def start(self):
        self.running = True
        print("USER_BACKEND_STARTED", flush=True)

    def generate(self, request):
        if not self.running:
            raise RuntimeError("Backend not running")

        prompt = request.get("prompt")

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        if request.get("stream", False):
            raise NotImplementedError(
                "Streaming not implemented"
            )

        self.calls += 1
        print("USER_BACKEND_GENERATE_CALLED", flush=True)

        output = "USER_BACKEND_EXECUTED: " + prompt

        return {
            "id": f"user-test-{self.calls}",
            "object": "text_completion",
            "model": "user-test-backend",
            "choices": [{
                "index": 0,
                "text": output,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def cancel(self, request_id):
        return False

    def health(self):
        return self.running

    def stop(self):
        self.running = False
        print("USER_BACKEND_STOPPED", flush=True)


def create_backend(context):
    return UserBackend()


def register_extensions(registry):
    registry.register(
        PluginSpec(
            kind="backend",
            name="user-test-backend",
            supported_backends=frozenset({
                "research_pytorch"
            }),
            factory=create_backend,
        )
    )
