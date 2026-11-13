"""Load trusted, out-of-tree framework extensions."""

import importlib


class ExtensionLoadError(ValueError):
    pass


def load_external_extensions(registry, modules):
    if not isinstance(modules, list):
        raise ExtensionLoadError(
            "modules must be a list of Python module names"
        )

    if len(set(modules)) != len(modules):
        raise ExtensionLoadError(
            "Duplicate extension module"
        )

    for name in modules:
        if not isinstance(name, str) or not name.strip():
            raise ExtensionLoadError(
                "Invalid extension module name"
            )

        # Import errors are intentionally propagated:
        # we never silently fall back.
        module = importlib.import_module(name)

        register = getattr(
            module, "register_extensions", None
        )

        if not callable(register):
            raise ExtensionLoadError(
                f"{name} must define register_extensions(registry)"
            )

        register(registry)
