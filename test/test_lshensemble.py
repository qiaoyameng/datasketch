import pickle
import unittest
from unittest.mock import patch

import mockredis
import numpy as np

import datasketch.lshensemble
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


class TestMinHashLSHEnsembleUpdateSession(unittest.TestCase):
    def _make_entry(self, key, size, rng):
        m = MinHash(num_perm=128)
        for v in rng.randint(0, 10**9, size=size):
            m.update(str(v).encode("utf8"))
        return (key, m, size)

    def _make_data(self, sizes, seed=42):
        rng = np.random.RandomState(seed)
        return [self._make_entry(key, size, rng) for key, size in enumerate(sizes)]

    def _partitions_of(self, lsh, key):
        return [i for i, index in enumerate(lsh.indexes) if any(key in index[r] for r in index)]

    def test_insert_remove_update_commit(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        before = {key: set(lsh.query(m, size)) for key, m, size in data}

        rng = np.random.RandomState(7)
        new_entry = self._make_entry("new", 15, rng)
        replacement = self._make_entry(3, 20, rng)
        session = lsh.update_session()
        session.insert(*new_entry)
        session.remove(5)
        session.update(3, replacement[1], replacement[2])

        # Staged updates are not visible before commit.
        self.assertTrue("new" not in lsh)
        self.assertTrue(5 in lsh)
        for key, m, size in data:
            self.assertEqual(set(lsh.query(m, size)), before[key])

        session.commit()
        self.assertTrue("new" in lsh)
        self.assertTrue(5 not in lsh)
        self.assertTrue(3 in lsh)
        self.assertEqual(set(lsh._entries), {e[0] for e in data} - {5} | {"new"})
        self.assertEqual(lsh._entries[3], (replacement[1], replacement[2]))
        self.assertEqual(lsh._entries["new"], (new_entry[1], new_entry[2]))
        # Every indexed set finds itself.
        for key, m, size in data:
            if key == 5:
                continue
            m, size = lsh._entries[key]
            self.assertIn(key, set(lsh.query(m, size)))
        m, size = lsh._entries["new"]
        self.assertIn("new", set(lsh.query(m, size)))

    def test_size_update_crossing_partition_boundary(self):
        # One key per size with consecutive sizes, so partition upper bounds
        # are themselves sizes of indexed keys.
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        boundary = lsh.uppers[0]
        self.assertIsNotNone(boundary)
        key = next(k for k, _, size in data if size == boundary)
        self.assertEqual(self._partitions_of(lsh, key), [0])

        # The new size lands exactly on the other side of the boundary:
        # uppers[0] + 1 is the lower bound of partition 1.
        rng = np.random.RandomState(11)
        _, m, new_size = self._make_entry(key, boundary + 1, rng)
        with lsh.update_session(strategy="keep") as session:
            session.update(key, m, new_size)

        # The key moved across the boundary into the next partition, and
        # the partition boundaries themselves are unchanged.
        self.assertEqual(self._partitions_of(lsh, key), [1])
        self.assertEqual(lsh.uppers[0], boundary)
        self.assertEqual(lsh._entries[key], (m, new_size))
        for k, _, _ in data:
            if k != key:
                self.assertEqual(len(self._partitions_of(lsh, k)), 1)

    def test_commit_matches_full_rebuild(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        boundary = lsh.uppers[0]
        key = next(k for k, _, size in data if size == boundary)

        rng = np.random.RandomState(13)
        moved = self._make_entry(key, boundary + 1, rng)
        added = [self._make_entry("added-%d" % i, s, rng) for i, s in enumerate((5, 60, 25))]
        updated = self._make_entry(20, 33, rng)
        with lsh.update_session(strategy="rebuild") as session:
            session.update(key, moved[1], moved[2])
            session.remove(0)
            session.remove(41 - 10)
            session.update(20, updated[1], updated[2])
            for entry in added:
                session.insert(*entry)

        final = {k: (m, s) for k, m, s in data}
        final.pop(0)
        final.pop(41 - 10)
        final[key] = (moved[1], moved[2])
        final[20] = (updated[1], updated[2])
        for k, m, s in added:
            final[k] = (m, s)

        rebuilt = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        rebuilt.index([(k, m, s) for k, (m, s) in final.items()])

        # Same partitioning and same candidate sets as a from-scratch index
        # of the final data, for indexed and held-out queries alike.
        self.assertEqual(lsh.lowers, rebuilt.lowers)
        self.assertEqual(lsh.uppers, rebuilt.uppers)
        queries = [(m, s) for m, s in final.values()]
        rng = np.random.RandomState(99)
        queries += [(e[1], e[2]) for e in (self._make_entry("q%d" % i, s, rng) for i, s in enumerate((12, 30, 55)))]
        for m, size in queries:
            self.assertEqual(set(lsh.query(m, size)), set(rebuilt.query(m, size)))

    def test_cancel_leaves_index_queryable(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        before = {key: set(lsh.query(m, size)) for key, m, size in data}

        rng = np.random.RandomState(17)
        session = lsh.update_session()
        session.insert(*self._make_entry("new", 15, rng))
        session.remove(3)
        session.update(7, *self._make_entry(7, 40, rng)[1:])
        session.cancel()

        self.assertTrue("new" not in lsh)
        self.assertTrue(3 in lsh)
        for key, m, size in data:
            self.assertTrue(key in lsh)
            self.assertEqual(set(lsh.query(m, size)), before[key])
        # A cancelled session cannot be reused or committed.
        self.assertRaises(ValueError, session.commit)
        self.assertRaises(ValueError, session.remove, 1)

    def test_exception_in_session_discards_updates(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        before = {key: set(lsh.query(m, size)) for key, m, size in data}

        rng = np.random.RandomState(19)
        try:
            with lsh.update_session() as session:
                session.insert(*self._make_entry("new", 15, rng))
                session.remove(3)
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        self.assertTrue("new" not in lsh)
        self.assertTrue(3 in lsh)
        for key, m, size in data:
            self.assertEqual(set(lsh.query(m, size)), before[key])

    def test_failed_commit_leaves_old_index_intact(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        before = {key: set(lsh.query(m, size)) for key, m, size in data}

        rng = np.random.RandomState(23)
        session = lsh.update_session()
        session.insert(*self._make_entry("new", 15, rng))
        session.remove(3)
        session.update(7, *self._make_entry(7, 40, rng)[1:])
        with patch.object(datasketch.lshensemble, "optimal_partitions", side_effect=RuntimeError("rebuild failed")):
            self.assertRaises(RuntimeError, session.commit)

        # The failed commit must not strand keys: nothing was removed from
        # the old partitions and nothing was half-added to new ones.
        self.assertTrue("new" not in lsh)
        for key, m, size in data:
            self.assertTrue(key in lsh)
            self.assertEqual(self._partitions_of(lsh, key), self._partitions_of(lsh, key)[:1])
            self.assertEqual(set(lsh.query(m, size)), before[key])

    def test_same_key_updates_fold_in_session_order(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)

        rng = np.random.RandomState(29)
        first = self._make_entry("k", 12, rng)
        second = self._make_entry("k", 25, rng)
        third = self._make_entry("k", 38, rng)
        with lsh.update_session() as session:
            session.insert(*first)
            session.update("k", second[1], second[2])
            session.remove("k")
            session.insert(*third)
        self.assertEqual(lsh._entries["k"], (third[1], third[2]))
        self.assertIn("k", set(lsh.query(third[1], third[2])))

        # update-then-remove on a committed key removes it.
        with lsh.update_session() as session:
            session.update(4, *self._make_entry(4, 30, rng)[1:])
            session.remove(4)
        self.assertTrue(4 not in lsh)

        # remove-then-insert on a committed key re-inserts it.
        replacement = self._make_entry(6, 18, rng)
        with lsh.update_session() as session:
            session.remove(6)
            session.insert(*replacement)
        self.assertEqual(lsh._entries[6], (replacement[1], replacement[2]))
        self.assertIn(6, set(lsh.query(replacement[1], replacement[2])))

    def test_validation(self):
        data = self._make_data(range(10, 42))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        m = MinHash(num_perm=128)
        m.update(b"x")

        session = lsh.update_session()
        self.assertRaises(ValueError, session.insert, "bad-size", m, 0)
        self.assertRaises(ValueError, session.insert, "bad-size", m, -3)
        self.assertRaises(ValueError, session.update, 1, m, 0)
        wrong_perm = MinHash(num_perm=64)
        wrong_perm.update(b"x")
        self.assertRaises(ValueError, session.insert, "wrong-perm", wrong_perm, 10)
        self.assertRaises(ValueError, session.insert, 1, m, 10)  # duplicate key
        self.assertRaises(ValueError, session.remove, "missing")
        self.assertRaises(ValueError, session.update, "missing", m, 10)
        session.cancel()

        self.assertRaises(ValueError, lsh.update_session, strategy="bogus")

    def test_scheme_mismatch_rejected(self):
        data = self._make_data(range(10, 42))  # default scheme: affine32
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)

        legacy = MinHash(num_perm=128, scheme="legacy")
        legacy.update(b"x")
        session = lsh.update_session()
        self.assertRaises(ValueError, session.insert, "legacy", legacy, 10)
        self.assertRaises(ValueError, session.update, 1, legacy, 10)
        session.cancel()

        # Mixing schemes within a single session is also rejected, even on
        # an empty index.
        empty = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        affine = MinHash(num_perm=128, scheme="affine32")
        affine.update(b"x")
        session = empty.update_session()
        session.insert("a", affine, 10)
        self.assertRaises(ValueError, session.insert, "b", legacy, 10)
        session.cancel()

        # index() itself rejects mixed schemes across partitions.
        mixed = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        self.assertRaises(ValueError, mixed.index, [("a", affine, 10), ("b", legacy, 10)])

    def test_remove_all_empties_index(self):
        data = self._make_data(range(10, 15))
        lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4)
        lsh.index(data)
        with lsh.update_session() as session:
            for key, _, _ in data:
                session.remove(key)
        self.assertTrue(lsh.is_empty())
        for _, m, size in data:
            self.assertEqual(list(lsh.query(m, size)), [])

    def test_update_session_requires_dict_storage(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.9, num_perm=128, num_part=4, storage_config=storage_config)
            lsh.index(self._make_data(range(10, 20)))
            self.assertRaises(ValueError, lsh.update_session)


if __name__ == "__main__":
    unittest.main()
