import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src" / "gpu"),
)

from prefix_kv import snapshot_prefix, restore_prefix


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, 1, len(self.values), 1)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.values.copy())

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 2

    def __getitem__(self, index):
        # The adapter slices the token dimension.
        return FakeTensor(self.values[index[-2]])


class DynamicLayer:
    is_sliding = False

    def __init__(self):
        self.keys = None
        self.values = None


class FakeCache:
    def __init__(self):
        self.layers = [DynamicLayer() for _ in range(4)]

    def update(self, keys, values, index):
        self.layers[index].keys = keys
        self.layers[index].values = values

    def get_seq_length(self, index):
        keys = self.layers[index].keys
        return 0 if keys is None else keys.shape[-2]


class SnapshotTests(unittest.TestCase):

    def make_filled_cache(self):
        cache = FakeCache()

        for idx in (1, 2):
            cache.update(
                FakeTensor([10, 20, 30, 40]),
                FakeTensor([50, 60, 70, 80]),
                idx,
            )

        return cache

    def test_snapshot_layer_range_and_memory(self):
        cache = self.make_filled_cache()

        snapshot = snapshot_prefix(cache, 1, 3, 4)

        self.assertEqual(len(snapshot.layers), 2)
        self.assertEqual(snapshot.token_count, 4)
        self.assertEqual(snapshot.nbytes, 32)

    def test_snapshot_is_independent(self):
        cache = self.make_filled_cache()
        snapshot = snapshot_prefix(cache, 1, 3, 4)

        cache.layers[1].keys.values[0] = 999

        self.assertEqual(
            snapshot.layers[0].keys.values[0], 10
        )

    def test_restore_partial_prefix(self):
        snapshot = snapshot_prefix(
            self.make_filled_cache(), 1, 3, 4
        )

        restored = restore_prefix(
            FakeCache, snapshot, reuse_tokens=2
        )

        self.assertEqual(restored.get_seq_length(1), 2)
        self.assertEqual(restored.get_seq_length(2), 2)
        self.assertEqual(
            restored.layers[1].keys.values,
            [10, 20],
        )

    def test_restore_does_not_modify_snapshot(self):
        snapshot = snapshot_prefix(
            self.make_filled_cache(), 1, 3, 4
        )

        restored = restore_prefix(FakeCache, snapshot, 3)

        restored.layers[1].keys.values[0] = 999

        self.assertEqual(
            snapshot.layers[0].keys.values[0], 10
        )

    def test_reject_invalid_reuse(self):
        snapshot = snapshot_prefix(
            self.make_filled_cache(), 1, 3, 4
        )

        with self.assertRaises(ValueError):
            restore_prefix(FakeCache, snapshot, 5)

    def test_reject_missing_layer(self):
        cache = self.make_filled_cache()

        with self.assertRaises(ValueError):
            snapshot_prefix(cache, 0, 3, 4)


if __name__ == "__main__":
    unittest.main()
