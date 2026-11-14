"""Example external architecture inspector. CPU-only demonstration."""

from llm_runtime.inspection.registry import InspectorSpec


ARCHITECTURE = "UserDemoForCausalLM"


def inspect_model(config, model_id, weight_index=None):
    if weight_index is not None:
        raise ValueError(
            "This example cannot verify checkpoint weights"
        )

    layers = config["num_hidden_layers"]

    if type(layers) is not int or layers < 1:
        raise ValueError("Invalid layer count")

    layout = {
        "schema_version": 1,
        "kind": "logical_checkpoint_layout",
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "num_layers": layers,
        "transformer_container": "transformer.blocks",
        "verification": "config_inferred_unverified",
        "layers": [
            {
                "index": i,
                "module_path": f"transformer.blocks.{i}",
            }
            for i in range(layers)
        ],
    }

    capabilities = {
        "schema_version": 1,
        "kind": "model_capabilities",
        "architecture": ARCHITECTURE,
        "model_id": model_id,
        "num_layers": layers,
    }

    return layout, capabilities


def register_inspectors(registry):
    registry.register(
        InspectorSpec(
            architecture=ARCHITECTURE,
            inspect=inspect_model,
        )
    )
