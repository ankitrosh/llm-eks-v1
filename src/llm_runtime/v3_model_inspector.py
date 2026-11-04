"""Inspect checkpoint metadata without loading model weights."""

import hashlib
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class InspectionError(ValueError):
    pass


# A registry entry does NOT imply every checkpoint is supported.
REGISTRY = {
    "qwen3": {
        "adapter": "qwen3",
        "status": "IMPLEMENTED_V2_TWO_RANK_GPU",
        "architectures": {"Qwen3ForCausalLM"},
        "layer_key": "num_hidden_layers",
    },
    "gpt2": {
        "adapter": "gpt2",
        "status": "PLANNED_V3_2",
        "architectures": {"GPT2LMHeadModel"},
        "layer_key": "n_layer",
    },
}


def fetch_config(model_id, revision="main"):
    """Download only config.json from Hugging Face."""

    if not isinstance(model_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/"
        r"[A-Za-z0-9][A-Za-z0-9_.-]*",
        model_id,
    ):
        raise InspectionError(
            "Expected Hugging Face model ID: org/repo"
        )

    if not isinstance(revision, str) or not revision.strip():
        raise InspectionError(
            "model.revision must be a nonempty string"
        )

    url = (
        "https://huggingface.co/"
        + model_id
        + "/resolve/"
        + quote(revision, safe="")
        + "/config.json"
    )

    headers = {
        "User-Agent": "llm-pipeline-metadata-inspector/0.1"
    }

    token = os.environ.get("HF_TOKEN")

    if token:
        headers["Authorization"] = "Bearer " + token

    try:
        with urlopen(
            Request(url, headers=headers), timeout=20
        ) as response:
            content = response.read(512 * 1024 + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise InspectionError(
            f"Unable to retrieve config.json for {model_id}; "
            "check repository access, revision and connectivity"
        ) from exc

    if len(content) > 512 * 1024:
        raise InspectionError("config.json exceeds 512 KiB")

    try:
        return json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise InspectionError(
            "Downloaded config.json is invalid"
        ) from exc


def inspect_config(
    model_id,
    config,
    *,
    expected_layers=None,
    revision="main",
):
    """Return metadata and support classification."""

    if not isinstance(config, dict):
        raise InspectionError("config.json must be an object")

    family = config.get("model_type")

    if not isinstance(family, str) or not family:
        raise InspectionError("Missing model_type")

    entry = REGISTRY.get(family)

    architectures = config.get("architectures") or []

    if not isinstance(architectures, list) or any(
        not isinstance(a, str) for a in architectures
    ):
        raise InspectionError(
            "architectures must be a list of strings"
        )

    architecture = (
        architectures[0] if architectures else None
    )

    recognized = bool(
        entry and architecture in entry["architectures"]
    )

    layer_key = (
        entry["layer_key"]
        if entry
        else (
            "num_hidden_layers"
            if "num_hidden_layers" in config
            else "n_layer"
        )
    )

    num_layers = config.get(layer_key)

    if type(num_layers) is not int or num_layers < 1:
        raise InspectionError(
            f"Missing or invalid layer count: {layer_key}"
        )

    if (
        expected_layers is not None
        and expected_layers != num_layers
    ):
        raise InspectionError(
            f"Manifest declares {expected_layers} layers, "
            f"but checkpoint declares {num_layers}"
        )

    quant = config.get("quantization_config")

    if quant is not None and not isinstance(quant, dict):
        raise InspectionError(
            "quantization_config must be an object"
        )

    quant = quant or {}
    method = quant.get("quant_method", "none")

    # Known GPTQ INT8 metadata shape. This is NOT a
    # guarantee that a particular CUDA kernel will work.
    gptq_int8 = (
        method == "gptq"
        and type(quant.get("bits")) is int
        and quant.get("bits") == 8
        and quant.get("group_size") == 128
        and quant.get("sym") is True
        and quant.get("desc_act") is False
        and quant.get("checkpoint_format", "gptq")
            == "gptq"
    )

    if not quant:
        quant_status = "NO_QUANTIZATION_DECLARED"
    elif gptq_int8 and family == "qwen3":
        quant_status = "GPTQ_INT8_LAYOUT_CANDIDATE"
    else:
        quant_status = "NOT_VERIFIED"

    # Checkpoint families previously exercised in V2.
    tested_family = (
        recognized
        and num_layers == 40
        and (
            (
                model_id ==
                "JunHowie/Qwen3-14B-GPTQ-Int8"
                and gptq_int8
            )
            or (
                model_id == "Qwen/Qwen3-14B"
                and not quant
            )
        )
    )

    if tested_family:
        checkpoint_status = (
            "PREVIOUSLY_TESTED_CHECKPOINT_FAMILY"
        )
    elif recognized:
        checkpoint_status = "UNVERIFIED_CHECKPOINT"
    elif entry:
        checkpoint_status = (
            "ARCHITECTURE_VARIANT_UNSUPPORTED"
        )
    else:
        checkpoint_status = "ADAPTER_REQUIRED"

    if entry is None:
        adapter_status = "ADAPTER_REQUIRED"
    elif not recognized:
        adapter_status = (
            "ARCHITECTURE_VARIANT_UNSUPPORTED"
        )
    else:
        adapter_status = entry["status"]

    not_run = [
        "Model weights not loaded",
        "Inference correctness not tested",
        "GPU memory feasibility not tested",
        "Distributed execution not tested",
    ]

    if revision == "main":
        not_run.append(
            "Revision not pinned; main can change"
        )

    if adapter_status == "PLANNED_V3_2":
        not_run.append(
            "GPT-2 adapter and CPU/Gloo execution pending"
        )

    if quant_status == "NOT_VERIFIED":
        not_run.append(
            "Quantization backend compatibility unverified"
        )

    return {
        "model_id": model_id,
        "revision": revision,
        "model_type": family,
        "architecture": architecture,
        "num_layers": num_layers,
        "hidden_size": config.get(
            "hidden_size", config.get("n_embd")
        ),
        "num_attention_heads": config.get(
            "num_attention_heads", config.get("n_head")
        ),
        "num_key_value_heads": config.get(
            "num_key_value_heads"
        ),
        "quantization": {
            "method": method,
            "bits": quant.get("bits"),
            "layout_status": quant_status,
        },
        "adapter": entry["adapter"] if entry else None,
        "adapter_status": adapter_status,
        "checkpoint_status": checkpoint_status,
        "metadata_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "not_run": not_run,
    }
