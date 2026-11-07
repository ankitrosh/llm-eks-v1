"""Create V3.1 Qwen3 execution stages from a launch configuration.

Supports unquantized stages, the explicit CPU GPTQ reference
backend, and trusted user-defined linear quantization providers.

Production GPTQ GPU kernels are not implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping

from llm_runtime.v3_1_launch_spec import (
    V31LaunchSpec,
    compile_v31_launch_spec,
)
from llm_runtime.v3_1_quantization_api import (
    QuantizationContext,
    QuantizationRegistry,
)
from llm_runtime.providers.v3_1_gptq_int8_torch import (
    GPTQInt8TorchReferenceProvider,
)
from llm_runtime.providers.v3_1_qwen3_nrank_factory import (
    Qwen3NRankStageOnlyFactory,
)


def create_v31_qwen3_stage_factory(
    config,
    *,
    launch_spec=None,
    device="cpu",
    dtype="float32",
    gpu_memory_gib=None,
):
    """Return an executable stage factory selected by user config.

    A provided launch_spec is assumed to have passed startup
    negotiation. The quantization profile is re-inspected to
    detect metadata drift before constructing stages.
    """

    if not isinstance(config, Mapping):
        raise ValueError("Configuration must be a mapping")

    model = config.get("model")

    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")

    checkpoint = model.get("checkpoint_dir")

    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("model.checkpoint_dir is required")

    if launch_spec is None:
        launch_spec = compile_v31_launch_spec(
            config,
            gpu_memory_gib=gpu_memory_gib,
        )
    elif gpu_memory_gib is not None:
        raise ValueError(
            "Do not override memory when supplying a launch spec"
        )

    if not isinstance(launch_spec, V31LaunchSpec):
        raise TypeError("Expected V31LaunchSpec")

    selector = model.get("quantization")
    declared = launch_spec.quantization
    provider = None

    if selector is None:
        if declared is not None:
            raise ValueError(
                "Launch spec expects a quantization provider"
            )
    else:
        if not isinstance(selector, Mapping):
            raise ValueError(
                "model.quantization must be a mapping"
            )

        if not isinstance(declared, Mapping):
            raise ValueError(
                "Launch spec has no quantization information"
            )

        provider_name = selector.get("provider")
        options = selector.get("options", {})

        if not isinstance(options, Mapping):
            raise ValueError(
                "quantization.options must be a mapping"
            )

        options = dict(options)

        if (
            declared.get("provider") != provider_name
            or declared.get("options") != options
        ):
            raise ValueError(
                "Quantization selection differs from launch spec"
            )

        if (
            "activation_dtype" in options
            and options["activation_dtype"] != dtype
        ):
            raise ValueError(
                "Configured activation dtype differs from runtime"
            )

        context = QuantizationContext(
            checkpoint_dir=checkpoint,
            device=device,
            activation_dtype=dtype,
            options=options,
        )

        if provider_name == "gptq":
            backend = options.get(
                "execution_backend", "auto"
            )

            if backend == "torch-reference":
                provider = GPTQInt8TorchReferenceProvider(
                    context
                )

            elif backend in ("auto", "marlin"):
                if device.split(":")[0] != "cuda":
                    raise ValueError(
                        "GPTQ execution_backend requires CUDA "
                        "for automatic Marlin selection; "
                        "use torch-reference explicitly for CPU"
                    )

                from llm_runtime.providers.v3_1_gptq_marlin_provider import (
                    GPTQMarlinProvider,
                )

                provider = GPTQMarlinProvider(context)

            else:
                raise ValueError(
                    f"Unsupported GPTQ execution_backend: {backend}"
                )

        elif provider_name == "custom":
            factory_path = selector.get("factory")

            if (
                not isinstance(factory_path, str)
                or declared.get("factory") != factory_path
            ):
                raise ValueError(
                    "Custom provider factory differs from launch spec"
                )

            provider = QuantizationRegistry().create(
                factory_path,
                context,
            )

        else:
            raise ValueError(
                f"Unsupported execution provider: {provider_name}"
            )

        # Unlike offline planning, execution validates device,
        # activation dtype and provider capabilities.
        profile = QuantizationRegistry.inspect(
            provider, context
        )

        if profile != launch_spec.checkpoint_profile:
            raise ValueError(
                "Quantization checkpoint profile changed "
                "after launch planning"
            )

        if profile.execution_mode != "linear":
            raise NotImplementedError(
                "Stage-level quantization execution adapters "
                "are not yet integrated"
            )

    return Qwen3NRankStageOnlyFactory(
        checkpoint_dir=checkpoint,
        layer_boundaries=tuple(
            launch_spec.layout.boundaries
        ),
        device=device,
        dtype=dtype,
        quantization_provider=provider,
    )
