"""Framework-owned interface for user-defined inference engines."""

from collections.abc import Mapping


class BackendContractError(TypeError):
    pass


REQUIRED_METHODS = (
    "validate",
    "prepare",
    "start",
    "generate",
    "cancel",
    "health",
    "stop",
)


class ExternalBackendAdapter:
    """Adapter around a user-supplied execution backend."""

    def __init__(self, backend, context, execution_plan):
        missing = [
            method for method in REQUIRED_METHODS
            if not callable(getattr(backend, method, None))
        ]
        if missing:
            raise BackendContractError(
                f"Backend missing methods: {missing}"
            )

        self.backend = backend
        self.started = False

        # Do not start an engine until configuration is valid.
        backend.validate(context)
        backend.prepare(execution_plan)

    def start(self):
        if self.started:
            raise BackendContractError("Backend already started")
        self.backend.start()
        self.started = True

    def generate(self, request):
        if not self.started:
            raise BackendContractError("Backend is not running")
        if not isinstance(request, Mapping):
            raise BackendContractError("Request must be a mapping")

        result = self.backend.generate(dict(request))

        if (
            not isinstance(result, dict)
            or not isinstance(result.get("choices"), list)
            or not result["choices"]
            or not isinstance(
                result["choices"][0].get("text"), str
            )
            or not isinstance(result.get("usage"), dict)
        ):
            raise BackendContractError(
                "Backend returned invalid completion response"
            )

        return result

    def cancel(self, request_id):
        if not self.started:
            raise BackendContractError("Backend is not running")
        return self.backend.cancel(request_id)

    def health(self):
        return (
            self.started
            and bool(self.backend.health())
        )

    def stop(self):
        if self.started:
            try:
                self.backend.stop()
            finally:
                self.started = False
