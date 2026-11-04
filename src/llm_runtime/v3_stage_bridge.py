"""Bridge V3 model adapters to the existing V2 serving contract.

No PyTorch imports during module import or static validation.
"""

import json
from pathlib import Path

from .v3_adapter_registry import create_adapter
from .v3_extension_contracts import StageSpec


def load_qwen3_stage_v3(
    model_dir,
    rank,
    split,
    *,
    stage_factory=None,
):
    """Return a V2-compatible Qwen3Stage using the V3 adapter."""

    root = Path(model_dir)

    if not root.is_dir():
        raise ValueError(
            f"Checkpoint directory does not exist: {root}"
        )

    config_path = root / "config.json"

    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot read checkpoint metadata: {config_path}"
        ) from exc

    if config.get("model_type") != "qwen3":
        raise ValueError(
            "Qwen3Adapter requires model_type=qwen3"
        )

    if config.get("tie_word_embeddings", False):
        raise ValueError(
            "Current Qwen3 loader requires untied embeddings"
        )

    num_layers = config.get("num_hidden_layers")

    if (
        type(num_layers) is not int
        or num_layers < 2
    ):
        raise ValueError("Invalid checkpoint layer count")

    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Current Qwen3 runtime requires 2 ranks")

    if (
        type(split) is not int
        or not 0 < split < num_layers
    ):
        raise ValueError(
            "Split must be strictly within model layer range"
        )

    specs = [
        StageSpec(
            rank=0,
            start_layer=0,
            end_layer=split,
            first_stage=True,
            final_stage=False,
        ),
        StageSpec(
            rank=1,
            start_layer=split,
            end_layer=num_layers,
            first_stage=False,
            final_stage=True,
        ),
    ]

    adapter = create_adapter(
        "qwen3",
        num_layers=num_layers,
        stage_factory=stage_factory,
    )

    adapter.validate_partition(specs)

    loaded = adapter.create_stage(
        specs[rank],
        checkpoint=str(root),
        device="cuda",
        dtype="bfloat16",
    )

    # For V3.1.5A, retain the exact object expected by
    # the existing NCCL, prefix-cache, and timing code.
    return loaded.module
