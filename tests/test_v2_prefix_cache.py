import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src" / "gpu"),
)

from prefix_cache import PrefixCache


class PrefixCacheTests(unittest.TestCase):

    def test_miss_then_hit(self):
        c = PrefixCache(100, min_reuse_tokens=2)

        assert c.lookup(
            [1, 2, 3, 4], model_key="A"
        ) is None

        assert c.put(
            [1, 2, 3],
            "kv1",
            nbytes=40,
            model_key="A",
        )

        hit = c.lookup(
            [1, 2, 3, 9], model_key="A"
        )

        assert hit.matched_tokens == 3
        assert hit.payload == "kv1"

    def test_partial_prefix(self):
        c = PrefixCache(100, min_reuse_tokens=2)

        c.put(
            [1, 2, 3, 4, 5],
            "snapshot",
            nbytes=10,
            model_key="A",
        )

        hit = c.lookup(
            [1, 2, 3, 99], model_key="A"
        )

        assert hit.matched_tokens == 3
        assert hit.cached_tokens == 5

    def test_identical_prompt_replays_final_token(self):
        c = PrefixCache(100, min_reuse_tokens=2)

        c.put(
            [1, 2, 3, 4],
            "kv",
            nbytes=10,
            model_key="A",
        )

        hit = c.lookup(
            [1, 2, 3, 4], model_key="A"
        )

        assert hit.matched_tokens == 3

    def test_model_and_namespace_isolation(self):
        c = PrefixCache(100, min_reuse_tokens=2)

        c.put(
            [1, 2, 3],
            "kv",
            nbytes=10,
            model_key="A",
            namespace="user1",
        )

        assert c.lookup(
            [1, 2, 3, 4],
            model_key="B",
            namespace="user1",
        ) is None

        assert c.lookup(
            [1, 2, 3, 4],
            model_key="A",
            namespace="user2",
        ) is None

    def test_lru_eviction(self):
        c = PrefixCache(
            max_bytes=20,
            max_entries=2,
            min_reuse_tokens=2,
        )

        c.put([1, 2, 3], "A", nbytes=10, model_key="M")
        c.put([4, 5, 6], "B", nbytes=10, model_key="M")

        # A becomes most recently used.
        assert c.lookup(
            [1, 2, 3, 7], model_key="M"
        ) is not None

        # Inserting C evicts B.
        c.put([7, 8, 9], "C", nbytes=10, model_key="M")

        assert c.lookup(
            [4, 5, 6, 7], model_key="M"
        ) is None

        assert c.lookup(
            [1, 2, 3, 7], model_key="M"
        ) is not None

        assert c.evictions == 1
        assert c.used_bytes == 20

    def test_oversized_entry(self):
        c = PrefixCache(20, min_reuse_tokens=2)

        c.put([1, 2, 3], "A", nbytes=10, model_key="M")

        assert not c.put(
            [4, 5, 6],
            "B",
            nbytes=21,
            model_key="M",
        )

        assert c.size == 1

    def test_longest_prefix_wins(self):
        c = PrefixCache(100, min_reuse_tokens=2)

        c.put([1, 2, 5], "short", nbytes=10, model_key="M")
        c.put([1, 2, 3, 4], "long", nbytes=10, model_key="M")

        hit = c.lookup(
            [1, 2, 3, 99], model_key="M"
        )

        assert hit.payload == "long"
        assert hit.matched_tokens == 3


if __name__ == "__main__":
    unittest.main()
