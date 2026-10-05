import pickle
import unittest
from unittest.mock import patch

import mockredis
import numpy as np

from datasketch.lsh import MinHashLSH
from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.minhash import MinHash


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


class TestMinHashLSHEnsemble(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        self.assertTrue(lsh.is_empty())

    def _data(self, count):
        sizes = np.random.randint(1, 100, count)
        for key, size in enumerate(sizes):
            m = MinHash()
            for i in range(size):
                m.update(("%d" % i).encode("utf8"))
            yield (key, m, size)

    def _minhash(self, *tokens):
        minhash = MinHash()
        for token in tokens:
            minhash.update(token.encode("utf8"))
        return minhash

    def _partition_for_key(self, lsh, key):
        for partition, index in enumerate(lsh.indexes):
            if any(key in inner_lsh for inner_lsh in index.values()):
                return partition
        return None

    def test_index(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        lsh.index(self._data(64))
        self.assertFalse(lsh.is_empty())
        self.assertTrue(41 in lsh)

    def test_query(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(64))
        lsh.index(data)
        for key, minhash, size in data:
            keys = lsh.query(minhash, size)
            self.assertTrue(key in keys)

    def test_update_session_commit_rebuilds_across_partition_boundary(self):
        old_minhash = self._minhash("old")
        stay_minhash = self._minhash("stay")
        new_minhash = self._minhash("new")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index(
            [
                ("move", old_minhash, 1),
                ("stay", stay_minhash, 2),
            ]
        )
        self.assertEqual(lsh.uppers, [1, 2])
        self.assertEqual(self._partition_for_key(lsh, "move"), 0)

        session = lsh.update_session()
        session.replace("move", new_minhash, 3)

        # Until commit, queries still use the old MinHash, size, and partitions.
        self.assertEqual(lsh.uppers, [1, 2])
        self.assertIn("move", set(lsh.query(old_minhash, 1)))
        self.assertEqual(self._partition_for_key(lsh, "move"), 0)

        session.commit()
        self.assertEqual(lsh.uppers, [2, 3])
        self.assertEqual(self._partition_for_key(lsh, "move"), 1)
        self.assertEqual(self._partition_for_key(lsh, "stay"), 0)

        final_data = [
            ("move", new_minhash, 3),
            ("stay", stay_minhash, 2),
        ]
        rebuilt = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        rebuilt.index(final_data)
        for _key, minhash, size in final_data:
            self.assertEqual(set(lsh.query(minhash, size)), set(rebuilt.query(minhash, size)))

    def test_update_session_cancel_keeps_old_index_queryable(self):
        old_minhash = self._minhash("old")
        stay_minhash = self._minhash("stay")
        new_minhash = self._minhash("new")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index(
            [
                ("move", old_minhash, 1),
                ("stay", stay_minhash, 2),
            ]
        )
        before = set(lsh.query(old_minhash, 1))

        session = lsh.update_session()
        session.replace("move", new_minhash, 3)
        self.assertEqual(lsh.uppers, [1, 2])
        session.cancel()

        self.assertEqual(lsh.uppers, [1, 2])
        self.assertEqual(set(lsh.query(old_minhash, 1)), before)
        self.assertEqual(self._partition_for_key(lsh, "move"), 0)
        self.assertEqual(lsh._entries["move"], (old_minhash, 1))

    def test_update_session_context_manager_cancels_without_commit(self):
        old_minhash = self._minhash("old")
        new_minhash = self._minhash("new")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index([("key", old_minhash, 1)])
        before = lsh._entries["key"]

        with lsh.update_session() as session:
            session.update("key", new_minhash, 2)

        self.assertEqual(lsh._entries["key"], before)
        self.assertIn("key", set(lsh.query(old_minhash, 1)))

    def test_failed_commit_leaves_old_partitions_intact(self):
        old_minhash = self._minhash("old")
        stay_minhash = self._minhash("stay")
        new_minhash = self._minhash("new")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index(
            [
                ("move", old_minhash, 1),
                ("stay", stay_minhash, 2),
            ]
        )
        session = lsh.update_session()
        session.replace("move", new_minhash, 3)

        with patch.object(MinHashLSH, "insert", side_effect=RuntimeError("simulated rebuild failure")):
            with self.assertRaises(RuntimeError):
                session.commit()

        self.assertEqual(lsh.uppers, [1, 2])
        self.assertEqual(self._partition_for_key(lsh, "move"), 0)
        self.assertEqual(self._partition_for_key(lsh, "stay"), 1)
        self.assertIn("move", set(lsh.query(old_minhash, 1)))
        self.assertIn("stay", set(lsh.query(stay_minhash, 2)))

    def test_repeated_updates_for_same_key_collapse_in_session_order(self):
        first_minhash = self._minhash("first")
        second_minhash = self._minhash("second")
        third_minhash = self._minhash("third")
        added_minhash = self._minhash("added")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index([("key", first_minhash, 1)])

        session = lsh.update_session()
        session.replace("key", second_minhash, 2)
        session.replace("key", third_minhash, 3)
        session.commit()
        self.assertEqual(lsh._entries["key"], (third_minhash, 3))

        session = lsh.update_session()
        session.remove("key")
        session.insert("key", second_minhash, 2)
        session.commit()
        self.assertEqual(lsh._entries["key"], (second_minhash, 2))

        session = lsh.update_session()
        session.insert("added", added_minhash, 1)
        session.remove("added")
        session.commit()
        self.assertNotIn("added", lsh)

    def test_update_session_validates_digest_parameters_and_size(self):
        old_minhash = self._minhash("old")
        other_minhash = self._minhash("other")
        legacy_minhash = MinHash(scheme="legacy")
        different_seed_minhash = MinHash(seed=2)
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index(
            [
                ("old", old_minhash, 1),
                ("other", other_minhash, 2),
            ]
        )

        session = lsh.update_session()
        with self.assertRaises(ValueError):
            session.insert("legacy", legacy_minhash, 1)
        with self.assertRaises(ValueError):
            session.replace("old", legacy_minhash, 1)
        with self.assertRaises(ValueError):
            session.replace("old", different_seed_minhash, 1)
        with self.assertRaises(ValueError):
            session.replace("old", old_minhash, 0)
        with self.assertRaises(TypeError):
            session.replace("old", old_minhash, 1.5)

        self.assertEqual(lsh._entries["old"], (old_minhash, 1))
        session.cancel()

    def test_update_session_can_switch_digest_family_only_after_removing_old_entries(self):
        old_minhash = self._minhash("old")
        retained_minhash = self._minhash("retained")
        replacement_minhash = MinHash(scheme="legacy")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index(
            [
                ("old", old_minhash, 1),
                ("retained", retained_minhash, 2),
            ]
        )

        with lsh.update_session() as session:
            session.remove("retained")
            with self.assertRaises(ValueError):
                session.insert("legacy", replacement_minhash, 1)
            session.remove("old")
            session.insert("legacy", replacement_minhash, 1)
            session.commit()

        self.assertEqual(lsh._minhash_digest, ("minhash", "legacy", 1))
        self.assertIn("legacy", lsh)

    def test_committed_session_cancel_does_not_mutate_index(self):
        old_minhash = self._minhash("old")
        new_minhash = self._minhash("new")
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=2, m=8)
        lsh.index([("key", old_minhash, 1)])

        session = lsh.update_session()
        session.replace("key", new_minhash, 2)
        session.commit()
        session.cancel()

        self.assertEqual(lsh._entries["key"], (new_minhash, 2))
        self.assertIn("key", set(lsh.query(new_minhash, 2)))

    def test_update_session_rejects_non_memory_storage(self):
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}})
            with self.assertRaises(ValueError):
                lsh.update_session()

    def test_pickle(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(32))
        lsh.index(data)
        buf = pickle.dumps(lsh)
        lsh2 = pickle.loads(buf)
        for _key, minhash, size in data:
            keys1 = lsh.query(minhash, size)
            keys2 = lsh2.query(minhash, size)
            self.assertTrue(set(keys1) == set(keys2))

    def test_index_redis(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.8, storage_config=storage_config)
            lsh.index(self._data(64))
            self.assertFalse(lsh.is_empty())
            self.assertTrue(41 in lsh)

    def test_query_redis(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.9, storage_config=storage_config)
            data = list(self._data(64))
            lsh.index(data)
            for key, minhash, size in data:
                keys = lsh.query(minhash, size)
                self.assertTrue(key in keys)


if __name__ == "__main__":
    unittest.main()
