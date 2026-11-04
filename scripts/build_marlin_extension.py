import os
import shutil
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

root = Path("/tmp/gptqmodel-src")
marlin = root / "gptqmodel_ext/marlin"

assert torch.version.cuda == "13.0", torch.version.cuda
assert CUDA_HOME, "CUDA toolkit not detected"

print("PyTorch:", torch.__version__, flush=True)
print("CUDA toolkit:", CUDA_HOME, flush=True)
print("GPU target:", os.environ["TORCH_CUDA_ARCH_LIST"], flush=True)

build_dir = Path("/tmp/marlin-compile")
build_dir.mkdir(parents=True, exist_ok=True)

module = load(
    name="gptqmodel_marlin_kernels",
    sources=[
        str(marlin / "marlin_cuda.cpp"),
        str(marlin / "marlin_cuda_kernel.cu"),
        str(marlin / "marlin_repack.cu"),
    ],
    extra_include_paths=[str(marlin)],
    extra_cflags=[
        "-O2",
        "-std=c++20",
        "-DENABLE_BF16",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-DENABLE_BF16",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    with_cuda=True,
    build_directory=str(build_dir),
    verbose=True,
)

output = Path("/opt/marlin-output")
output.mkdir(parents=True, exist_ok=True)

shutil.copy2(module.__file__, output / Path(module.__file__).name)

print("PASS: Marlin CUDA extension compiled", flush=True)
