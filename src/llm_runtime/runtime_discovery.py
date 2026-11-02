"""Read-only discovery of installed inference runtime capabilities.

An available backend is only an import/API candidate,
NOT a validated GPU kernel.
"""

from dataclasses import asdict, dataclass
import importlib
from importlib import metadata
import platform


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class RuntimeInfo:
    python_version: str
    torch_version: str | None
    transformers_version: str | None
    flash_attn_version: str | None

    torch_import_ok: bool
    torch_cuda_build: str | None

    device_type: str
    gpu_name: str | None
    compute_capability: tuple[int, int] | None

    sdpa_api_available: bool
    flash_attn_importable: bool

    backend_candidates: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_runtime() -> RuntimeInfo:
    """Inspect this Python process without downloading model weights."""

    torch_version = _package_version("torch")
    transformers_version = _package_version("transformers")
    flash_attn_version = _package_version("flash-attn")

    notes = []
    torch_import_ok = False
    torch_cuda_build = None
    device_type = "cpu_or_unavailable"
    gpu_name = None
    capability = None
    sdpa_available = False
    flash_importable = False

    if torch_version is None:
        notes.append(
            "PyTorch is not installed in this Python environment"
        )
    else:
        try:
            torch = importlib.import_module("torch")
            torch_import_ok = True

            torch_cuda_build = getattr(
                getattr(torch, "version", None),
                "cuda",
                None,
            )

            sdpa_available = callable(
                getattr(
                    torch.nn.functional,
                    "scaled_dot_product_attention",
                    None,
                )
            )

            if torch.cuda.is_available():
                device_type = "cuda"
                capability = tuple(
                    torch.cuda.get_device_capability(0)
                )
                gpu_name = torch.cuda.get_device_name(0)
            else:
                notes.append(
                    "No accessible CUDA GPU in this Python process"
                )

        except Exception as exc:
            notes.append(
                f"PyTorch inspection failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if flash_attn_version is None:
        notes.append(
            "flash-attn distribution is not installed"
        )
    elif device_type != "cuda":
        notes.append(
            "FlashAttention import check deferred: "
            "no accessible CUDA GPU"
        )
    else:
        try:
            module = importlib.import_module(
                "flash_attn.flash_attn_interface"
            )

            flash_importable = callable(
                getattr(module, "flash_attn_func", None)
            )

            if not flash_importable:
                notes.append(
                    "flash-attn import succeeded but "
                    "flash_attn_func is missing"
                )

        except Exception as exc:
            notes.append(
                f"FlashAttention import failed: "
                f"{type(exc).__name__}: {exc}"
            )

    candidates = []

    if sdpa_available:
        candidates.append("sdpa")

    if device_type == "cuda" and flash_importable:
        candidates.append("flash_attention_2")

    notes.append(
        "Backend candidates are not verified fused-kernel "
        "execution or model compatibility"
    )

    return RuntimeInfo(
        python_version=platform.python_version(),
        torch_version=torch_version,
        transformers_version=transformers_version,
        flash_attn_version=flash_attn_version,
        torch_import_ok=torch_import_ok,
        torch_cuda_build=torch_cuda_build,
        device_type=device_type,
        gpu_name=gpu_name,
        compute_capability=capability,
        sdpa_api_available=sdpa_available,
        flash_attn_importable=flash_importable,
        backend_candidates=tuple(candidates),
        notes=tuple(notes),
    )
