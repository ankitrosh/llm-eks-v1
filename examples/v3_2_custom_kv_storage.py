"""Example user-supplied KV storage adapter."""

from llm_runtime.v3_2_kv_storage import TorchKVBlockStorage


class CountingKVStorage(TorchKVBlockStorage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.write_calls = 0

    def write(self, layer, block_id, offset, keys, values):
        super().write(
            layer, block_id, offset, keys, values
        )
        self.write_calls += 1
