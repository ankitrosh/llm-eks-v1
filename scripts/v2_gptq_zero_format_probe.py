"""CPU-only test of GPTQ v1 to v2 zero-point conversion."""

import torch

from gptqmodel.nn_modules.qlinear.tritonv2 import (
    TritonV2QuantLinear,
)
from gptqmodel.utils.model import (
    convert_gptq_v1_to_v2_format_module,
)


def main():
    layer = TritonV2QuantLinear(
        bits=8,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        bias=False,
        pack_dtype=torch.int32,
    )

    # Four distinct zero points packed into one INT32.
    expected = [120, 125, 129, 130]

    packed_v1 = sum(
        (value - 1) << (8 * i)
        for i, value in enumerate(expected)
    )

    if packed_v1 >= 2**31:
        packed_v1 -= 2**32

    with torch.no_grad():
        layer.qzeros.fill_(packed_v1)

        convert_gptq_v1_to_v2_format_module(
            module=layer,
            bits=8,
            pack_dtype=torch.int32,
        )

    word = layer.qzeros[0, 0].item()

    actual = [
        (word >> (8 * i)) & 255
        for i in range(4)
    ]

    assert actual == expected, (
        f"Expected {expected}, got {actual}"
    )
    assert layer.qzero_format() == 2

    print("Original zero points:", expected)
    print("Recovered zero points:", actual)
    print("PASS: GPTQ v1 to v2 conversion")
    print("PASS: Packed INT32 representation preserved")


if __name__ == "__main__":
    main()
