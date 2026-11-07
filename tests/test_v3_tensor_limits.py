import unittest

from llm_runtime.v3_ordered_rank_transport import (
    TensorTransferLimits,
    TransportFailure,
    _shape,
)


class TensorLimitsTests(unittest.TestCase):

    def test_gpt2_shape(self):
        self.assertEqual(
            _shape([1, 128, 768], "float32"),
            (1, 128, 768),
        )

    def test_qwen_hidden_states(self):
        self.assertEqual(
            _shape([1, 128, 5120], "bfloat16"),
            (1, 128, 5120),
        )

    def test_large_vocabulary(self):
        self.assertEqual(
            _shape([1, 1, 151936], "float32"),
            (1, 1, 151936),
        )

    def test_non_three_dimensional_tensor(self):
        self.assertEqual(
            _shape([4, 16], "float16"),
            (4, 16),
        )

    def test_transfer_limit(self):
        limits = TensorTransferLimits(
            max_tensor_bytes=1024,
        )

        with self.assertRaises(TransportFailure):
            _shape(
                [1, 128, 5120],
                "bfloat16",
                limits=limits,
            )

    def test_invalid_shape(self):
        with self.assertRaises(TransportFailure):
            _shape([1, 0, 5120], "float32")

    def test_dimension_limit(self):
        limits = TensorTransferLimits(
            max_dimensions=2,
        )

        with self.assertRaises(TransportFailure):
            _shape(
                [1, 2, 3],
                "float32",
                limits=limits,
            )

    def test_unsupported_dtype(self):
        with self.assertRaises(TransportFailure):
            _shape([1, 128, 5120], "complex64")


if __name__ == "__main__":
    unittest.main()
