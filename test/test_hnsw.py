import io
import json
import os
import tempfile
import unittest
import warnings
import zipfile

import numpy as np
import pytest

from datasketch.hnsw import HNSW, HNSWSnapshotError
from datasketch.minhash import MinHash


def l2_distance(x, y):
    return np.linalg.norm(x - y)


def jaccard_distance(x, y):
    return 1.0 - float(len(np.intersect1d(x, y, assume_unique=False))) / float(len(np.union1d(x, y)))


class TestHNSW(unittest.TestCase):
    def _create_random_points(self, n=100, dim=10):
        return np.random.rand(n, dim)

    def _create_index(self, vecs, keys=None):
        hnsw = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, vecs, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return self._search_index_dist(index, queries, l2_distance, k)

    def _insert_points(self, index, points, keys=None):
        original_length = len(index)

        if keys is None:
            keys = list(range(len(points)))

        for i, (key, point) in enumerate(zip(keys, points)):
            # Test insert.
            if i % 2 == 0:
                index.insert(key, point)
            else:
                index[key] = point
            # Make sure the entry point is set.
            self.assertTrue(index._entry_point is not None)
            # Test contains.
            self.assertIn(key, index)
            if original_length == 0:
                self.assertNotIn(key + 1, index)
            # Test get.
            self.assertTrue(np.array_equal(index.get(key), point))
            self.assertTrue(np.array_equal(index[key], point))

        if original_length == 0:
            # Test length.
            self.assertEqual(len(index), len(points))

            # Test order.
            for key_indexed, key in zip(index, keys):
                self.assertEqual(key_indexed, key)
            for key_indexed, key in zip(index.keys(), keys):
                self.assertEqual(key_indexed, key)
            for vec_indexed, vec in zip(index.values(), points):
                self.assertTrue(np.array_equal(vec_indexed, vec))
            for (key_indexed, vec_indexed), key, vec in zip(index.items(), keys, points):
                self.assertEqual(key_indexed, key)
                self.assertTrue(np.array_equal(vec_indexed, vec))

    def _search_index_dist(self, index, queries, distance_func, k=10):
        for i in range(len(queries)):
            results = index.query(queries[i], 10)
            # Check graph connectivity.
            self.assertEqual(len(results), 10)
            for j in range(len(results) - 1):
                self.assertLessEqual(
                    distance_func(index[results[j][0]], queries[i]),
                    distance_func(index[results[j + 1][0]], queries[i]),
                )

    def test_search(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        self._search_index(hnsw, data)

    def test_upsert(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        new_data = self._create_random_points(n=10, dim=10)
        self._insert_points(hnsw, new_data)
        self._search_index(hnsw, new_data)

    def test_update(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        new_data = self._create_random_points(n=10, dim=10)
        hnsw.update({i: new_data[i] for i in range(len(new_data))})
        self._search_index(hnsw, new_data)

    def test_merge(self):
        data1 = self._create_random_points()
        data2 = self._create_random_points()
        hnsw1 = self._create_index(data1, keys=list(range(len(data1))))
        hnsw2 = self._create_index(data2, keys=list(range(len(data1), len(data1) + len(data2))))
        new_index = hnsw1.merge(hnsw2)
        self._search_index(new_index, data1)
        self._search_index(new_index, data2)
        for i in range(len(data1)):
            self.assertIn(i, new_index)
            self.assertTrue(np.array_equal(new_index[i], data1[i]))
        for i in range(len(data2)):
            self.assertIn(i + len(data1), new_index)
            self.assertTrue(np.array_equal(new_index[i + len(data1)], data2[i]))

    def test_pickle(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        import pickle

        hnsw2 = pickle.loads(pickle.dumps(hnsw))
        self.assertEqual(hnsw, hnsw2)

    def test_copy(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        hnsw2 = hnsw.copy()
        self.assertEqual(hnsw, hnsw2)

        hnsw.remove(0)
        self.assertTrue(0 not in hnsw)
        self.assertTrue(0 in hnsw2)

    def test_soft_remove_and_pop_and_clean(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        # Remove all points except the last one.
        for i in range(len(data) - 1):
            if i % 2 == 0:
                hnsw.remove(i)
            else:
                point = hnsw.pop(i)
                self.assertTrue(np.array_equal(point, data[i]))
            self.assertNotIn(i, hnsw)
            self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.pop, i)
            # Test idempotency.
            hnsw.remove(i)
            hnsw.remove(i)
            hnsw.remove(i)
            results = hnsw.query(data[i], 10)
            # Check graph connectivity.
            expected_result_size = min(10, len(data) - i - 1)
            if len(results) != expected_result_size:
                warnings.warn(
                    f"Issue encountered at i={i} during soft remove unit test: "
                    f"expected {expected_result_size} results, "
                    f"got {len(results)} results. "
                    "Potential graph connectivity issue.",
                    stacklevel=2,
                )
                # NOTE: we are not getting the expected number of results.
                # Try hard remove all previous soft removed points.
                hnsw.clean()
                results = hnsw.query(data[i], 10)
                self.assertEqual(len(results), min(10, len(data) - i - 1))
        # Remove last point.
        hnsw.remove(len(data) - 1)
        self.assertNotIn(len(data) - 1, hnsw)
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.pop, len(data) - 1)
        self.assertRaises(KeyError, hnsw.remove, len(data) - 1)
        # Test search on empty index.
        self.assertRaises(ValueError, hnsw.query, data[0])
        # Test clean.
        hnsw.clean()
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.remove, 0)
        self.assertRaises(ValueError, hnsw.query, data[0])

    @pytest.mark.flaky(reruns=3)
    def test_hard_remove_and_pop_and_clean(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        # Remove all points except the last one.
        for i in range(len(data) - 1):
            if i % 2 == 0:
                hnsw.remove(i, hard=True)
            else:
                point = hnsw.pop(i, hard=True)
                self.assertTrue(np.array_equal(point, data[i]))
            self.assertNotIn(i, hnsw)
            self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.pop, i)
            self.assertRaises(KeyError, hnsw.remove, i)
            results = hnsw.query(data[i], 10)
            # Check graph connectivity.
            self.assertEqual(len(results), min(10, len(data) - i - 1))
        # Remove last point.
        hnsw.remove(len(data) - 1, hard=True)
        self.assertNotIn(len(data) - 1, hnsw)
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.pop, len(data) - 1)
        self.assertRaises(KeyError, hnsw.remove, len(data) - 1)
        # Test search on empty index.
        self.assertRaises(ValueError, hnsw.query, data[0])
        # Test clean.
        hnsw.clean()
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(KeyError, hnsw.remove, 0)
        self.assertRaises(ValueError, hnsw.query, data[0])

    def test_popitem_last(self):
        data = self._create_random_points()
        for hard in [True, False]:
            hnsw = self._create_index(data)
            for i in range(len(data)):
                key, point = hnsw.popitem(hard=hard)
                self.assertTrue(np.array_equal(point, data[key]))
                self.assertEqual(key, len(data) - i - 1)
                self.assertTrue(np.array_equal(point, data[len(data) - i - 1]))
                self.assertNotIn(key, hnsw)
                self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.popitem)

    def test_popitem_first(self):
        data = self._create_random_points()
        for hard in [True, False]:
            hnsw = self._create_index(data)
            for i in range(len(data)):
                key, point = hnsw.popitem(last=False, hard=hard)
                self.assertTrue(np.array_equal(point, data[key]))
                self.assertEqual(key, i)
                self.assertTrue(np.array_equal(point, data[i]))
                self.assertNotIn(key, hnsw)
                self.assertEqual(len(hnsw), len(data) - i - 1)
            self.assertRaises(KeyError, hnsw.popitem)

    def test_clear(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        hnsw.clear()
        self.assertEqual(len(hnsw), 0)
        self.assertRaises(StopIteration, next, iter(hnsw))
        self.assertRaises(StopIteration, next, iter(hnsw.keys()))
        self.assertRaises(StopIteration, next, iter(hnsw.values()))
        self.assertRaises(KeyError, hnsw.pop, 0)
        self.assertRaises(KeyError, hnsw.__getitem__, 0)
        self.assertRaises(KeyError, hnsw.popitem)
        self.assertRaises(ValueError, hnsw.query, data[0])


class TestHNSWLayerWithReversedEdges(TestHNSW):
    def _create_index(self, vecs, keys=None):
        hnsw = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
            reversed_edges=True,
        )
        self._insert_points(hnsw, vecs, keys)
        return hnsw


class TestHNSWJaccard(TestHNSW):
    def _create_random_points(self, high=50, n=100, dim=10):
        return np.random.randint(0, high, (n, dim))

    def _create_index(self, sets, keys=None):
        hnsw = HNSW(
            distance_func=jaccard_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, sets, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return super()._search_index_dist(index, queries, jaccard_distance, k)


def minhash_jaccard_distance(x, y) -> float:
    return 1.0 - x.jaccard(y)


class TestHNSWMinHashJaccard(TestHNSW):
    def _create_random_points(self, high=50, n=100, dim=10):
        sets = np.random.randint(0, high, (n, dim))
        return MinHash.bulk(sets, num_perm=128)

    def _create_index(self, minhashes, keys=None):
        hnsw = HNSW(
            distance_func=minhash_jaccard_distance,
            m=16,
            ef_construction=100,
        )
        self._insert_points(hnsw, minhashes, keys)
        return hnsw

    def _search_index(self, index, queries, k=10):
        return super()._search_index_dist(index, queries, minhash_jaccard_distance, k)

def _rewrite_snapshot(snapshot_bytes, metadata_updates=None, members=None):
    members = members or {}
    metadata_updates = metadata_updates or {}
    input_file = zipfile.ZipFile(io.BytesIO(snapshot_bytes), mode="r")
    output = io.BytesIO()
    with input_file as source, zipfile.ZipFile(output, mode="w") as target:
        for info in source.infolist():
            if info.filename == _TEST_SNAPSHOT_METADATA:
                metadata = json.loads(source.read(info))
                metadata.update(metadata_updates)
                data = json.dumps(metadata).encode("utf-8")
            elif info.filename in members:
                data = members[info.filename]
            else:
                data = source.read(info)
            target.writestr(info.filename, data)
    return output.getvalue()


def _array_member(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


_TEST_SNAPSHOT_METADATA = "metadata.json"


class TestHNSWSnapshot(unittest.TestCase):
    def test_snapshot_roundtrip_queries_insertions_and_soft_delete(self):
        rng = np.random.RandomState(7)
        points = rng.randn(60, 6).astype(np.float32)
        keys = [i if i % 2 == 0 else f"key-{i}" for i in range(len(points))]
        original = HNSW(l2_distance, m=5, ef_construction=24, m0=8, seed=123)
        for key, point in zip(keys, points):
            original.insert(key, point)
        original.remove(0)
        query_points = points[1:6]
        expected_results = [original.query(point, k=5) for point in query_points]

        snapshot = io.BytesIO()
        original.save_snapshot(snapshot, "l2-distance")
        snapshot.seek(0)
        loaded = HNSW.load_snapshot(snapshot, l2_distance, "l2-distance")

        self.assertEqual(original, loaded)
        self.assertEqual(len(loaded), 59)
        self.assertNotIn(0, loaded)
        self.assertTrue(loaded._nodes[0].is_deleted)
        self.assertEqual([loaded.query(point, k=5) for point in query_points], expected_results)

        extra_points = rng.randn(10, 6).astype(np.float32)
        original_extra = original.copy()
        loaded_extra = loaded.copy()
        for index, point in enumerate(extra_points):
            key = f"new-{index}"
            original_extra.insert(key, point)
            loaded_extra.insert(key, point)
        self.assertEqual(original_extra, loaded_extra)

    def test_snapshot_path_roundtrip(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        index = HNSW(l2_distance, m=3, ef_construction=8, seed=5)
        index.update({i: point for i, point in enumerate(points)})
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
            path = os.path.join(directory, "hnsw.snapshot")
            index.save_snapshot(path, "l2")
            loaded = HNSW.load_snapshot(path, l2_distance, "l2")
        self.assertEqual(index, loaded)

    def test_empty_snapshot_roundtrip(self):
        index = HNSW(l2_distance, m=4, seed=3)
        snapshot = io.BytesIO()
        index.save_snapshot(snapshot, "l2")
        snapshot.seek(0)
        loaded = HNSW.load_snapshot(snapshot, l2_distance, "l2")
        self.assertEqual(index, loaded)

    def test_unsupported_keys_and_vector_types_are_rejected(self):
        points = np.array([[1.0, 2.0]], dtype=np.float64)
        index = HNSW(l2_distance, m=3)
        index[("not", "supported")] = points[0]
        with self.assertRaises(HNSWSnapshotError):
            index.save_snapshot(io.BytesIO(), "l2")

        integer_points = np.array([[1, 2]], dtype=np.int8)
        index = HNSW(l2_distance, m=3)
        index[0] = integer_points[0]
        with self.assertRaises(HNSWSnapshotError):
            index.save_snapshot(io.BytesIO(), "l2")

        with self.assertRaises(HNSWSnapshotError):
            HNSW(l2_distance, m=3, seed=1).save_snapshot(io.BytesIO(), "bad identifier!")

    def test_rejected_snapshot_does_not_replace_existing_index(self):
        points = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        index = HNSW(l2_distance, m=3, seed=1)
        index.update({0: points[0], 1: points[1]})
        snapshot = io.BytesIO()
        index.save_snapshot(snapshot, "l2")
        snapshot.seek(0)
        corrupt = _rewrite_snapshot(snapshot.getvalue(), metadata_updates={"distance_identifier": "cosine"})

        existing = HNSW(l2_distance, m=3, seed=9)
        existing.insert("untouched", np.array([9.0, 9.0]))
        before = existing.copy()
        with self.assertRaises(HNSWSnapshotError):
            existing.load_snapshot(io.BytesIO(corrupt), l2_distance, "l2")
        self.assertEqual(existing, before)

    def test_snapshot_rejects_duplicate_key_and_entry_point_corruption(self):
        index = self._valid_snapshot_index()
        snapshot = self._valid_snapshot_bytes(index)
        metadata = self._snapshot_metadata(snapshot)
        metadata["keys"][1] = metadata["keys"][0]
        duplicate_key_snapshot = self._replace_metadata(snapshot, metadata)
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(duplicate_key_snapshot), l2_distance, "l2")

        for bad_entry_point in (-1, 4):
            bad_snapshot = _rewrite_snapshot(snapshot, {"entry_point": bad_entry_point})
            with self.assertRaises(HNSWSnapshotError):
                HNSW.load_snapshot(io.BytesIO(bad_snapshot), l2_distance, "l2")

    def test_snapshot_rejects_graph_and_array_corruption(self):
        index = self._valid_snapshot_index()
        snapshot = self._valid_snapshot_bytes(index)
        metadata = self._snapshot_metadata(snapshot)
        targets_name = metadata["layers"][1]["targets_name"]
        targets = self._read_snapshot_array(snapshot, targets_name)
        self.assertGreater(targets.size, 0)
        targets[0] = 2
        cross_layer_snapshot = _rewrite_snapshot(snapshot, members={targets_name: _array_member(targets)})
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(cross_layer_snapshot), l2_distance, "l2")

        nonexistent_targets = targets.copy()
        nonexistent_targets[0] = 99
        nonexistent_snapshot = _rewrite_snapshot(
            snapshot,
            members={targets_name: _array_member(nonexistent_targets)},
        )
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(nonexistent_snapshot), l2_distance, "l2")

        deleted_flags = self._read_snapshot_array(snapshot, "deleted.npy")
        deleted_flags[0] = True
        deleted_entry_snapshot = _rewrite_snapshot(
            snapshot,
            members={"deleted.npy": _array_member(deleted_flags)},
        )
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(deleted_entry_snapshot), l2_distance, "l2")

        offsets_name = metadata["layers"][0]["offsets_name"]
        bad_offsets = self._read_snapshot_array(snapshot, offsets_name)[:-1]
        bad_offsets_snapshot = _rewrite_snapshot(snapshot, members={offsets_name: _array_member(bad_offsets)})
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(bad_offsets_snapshot), l2_distance, "l2")

        points = np.array([index._nodes[i].point for i in range(4)])
        bad_points = np.vstack([points, np.zeros((1, 2), dtype=points.dtype)])
        bad_shape_snapshot = _rewrite_snapshot(snapshot, members={"points.npy": _array_member(bad_points)})
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(bad_shape_snapshot), l2_distance, "l2")

        bad_deleted = np.array([False, False, False], dtype=bool)
        bad_deleted_snapshot = _rewrite_snapshot(snapshot, members={"deleted.npy": _array_member(bad_deleted)})
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(bad_deleted_snapshot), l2_distance, "l2")

    def test_snapshot_rejects_invalid_distance_and_budgets(self):
        index = self._valid_snapshot_index()
        snapshot = io.BytesIO()
        index.save_snapshot(snapshot, "l2")
        snapshot.seek(0)

        def nan_distance(x, y):
            return float("nan")

        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(snapshot, nan_distance, "l2")

        snapshot.seek(0)
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(snapshot, l2_distance, "l2", max_points=3)

        snapshot.seek(0)
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(snapshot, l2_distance, "l2", max_dimension=1)

    def test_snapshot_rejects_invalid_metadata_and_npy_bytes(self):
        index = self._valid_snapshot_index()
        snapshot = self._valid_snapshot_bytes(index)
        for updates in (
            {"version": 2},
            {"format": "pickle"},
            {"m": 0},
            {"dimension": 3},
        ):
            bad_snapshot = _rewrite_snapshot(snapshot, updates)
            with self.assertRaises(HNSWSnapshotError):
                HNSW.load_snapshot(io.BytesIO(bad_snapshot), l2_distance, "l2")

        bad_bytes = bytearray(snapshot)
        bad_bytes[len(bad_bytes) // 2] ^= 0xFF
        with self.assertRaises(HNSWSnapshotError):
            HNSW.load_snapshot(io.BytesIO(bytes(bad_bytes)), l2_distance, "l2")

    @staticmethod
    def _valid_snapshot_index():
        points = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.float64,
        )
        index = HNSW(l2_distance, m=3, ef_construction=8, seed=2)
        for i, point in enumerate(points[:2]):
            index.insert(i, point, level=1)
        for i, point in enumerate(points[2:], start=2):
            index.insert(i, point, level=0)
        return index

    @staticmethod
    def _valid_snapshot_bytes(index):
        snapshot = io.BytesIO()
        index.save_snapshot(snapshot, "l2")
        return snapshot.getvalue()

    @staticmethod
    def _snapshot_metadata(snapshot_bytes):
        with zipfile.ZipFile(io.BytesIO(snapshot_bytes)) as snapshot_file:
            return json.loads(snapshot_file.read(_TEST_SNAPSHOT_METADATA))

    @staticmethod
    def _replace_metadata(snapshot_bytes, metadata):
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(snapshot_bytes), mode="r") as source, zipfile.ZipFile(
            output, mode="w"
        ) as target:
            for info in source.infolist():
                if info.filename == _TEST_SNAPSHOT_METADATA:
                    data = json.dumps(metadata).encode("utf-8")
                else:
                    data = source.read(info)
                target.writestr(info.filename, data)
        return output.getvalue()

    @staticmethod
    def _read_snapshot_array(snapshot_bytes, name):
        with zipfile.ZipFile(io.BytesIO(snapshot_bytes)) as zf:
            with zf.open(name) as array_file:
                return np.load(array_file, allow_pickle=False)
