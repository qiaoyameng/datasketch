import io
import json
import os
import tempfile
import unittest
import warnings

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


class TestHNSWSnapshot(unittest.TestCase):
    def _create_index(self, n=200, dim=16, seed=42, index_seed=12345, **kwargs):
        data = np.random.RandomState(seed).rand(n, dim)
        index = HNSW(
            distance_func=l2_distance,
            m=16,
            ef_construction=100,
            seed=index_seed,
            **kwargs,
        )
        for i, point in enumerate(data):
            index.insert(i, point)
        return index, data

    def _save_to_bytes(self, index, distance_func_id="l2"):
        buf = io.BytesIO()
        index.save_snapshot(buf, distance_func_id)
        return buf.getvalue()

    def _load_from_bytes(self, data, distance_func_id="l2", **kwargs):
        return HNSW.load_snapshot(
            io.BytesIO(data),
            distance_func=l2_distance,
            distance_func_id=distance_func_id,
            **kwargs,
        )

    def _read_snapshot_arrays(self, data):
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def _write_snapshot_arrays(self, arrays):
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    def _edit_snapshot(self, data, array_edits=None, metadata_edit=None):
        arrays = self._read_snapshot_arrays(data)
        if array_edits:
            arrays.update(array_edits)
        if metadata_edit is not None:
            metadata = json.loads(arrays["metadata"].tobytes().decode("utf-8"))
            metadata_edit(metadata)
            arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
        return self._write_snapshot_arrays(arrays)

    def test_roundtrip_fixed_seed(self):
        index, data = self._create_index()
        self.assertGreaterEqual(len(index._graphs), 2)
        # Soft-remove some points to exercise the soft-deletion flags.
        index.remove(3)
        index.remove(17)
        loaded = self._load_from_bytes(self._save_to_bytes(index))
        self.assertEqual(index, loaded)
        # Queries return identical results.
        for query in data[:20]:
            self.assertEqual(index.query(query, 10), loaded.query(query, 10))
        # Subsequent inserts evolve both indexes identically because the
        # random state is restored.
        new_data = np.random.RandomState(7).rand(50, data.shape[1])
        for j, point in enumerate(new_data):
            index.insert(1000 + j, point)
            loaded.insert(1000 + j, point)
        self.assertEqual(index, loaded)
        for query in new_data[:10]:
            self.assertEqual(index.query(query, 10), loaded.query(query, 10))

    def test_roundtrip_str_keys(self):
        index, data = self._create_index(n=100, dim=8)
        str_index = HNSW(distance_func=l2_distance, m=16, ef_construction=100, seed=12345)
        for i, point in enumerate(data):
            str_index.insert(f"key-{i}", point)
        loaded = self._load_from_bytes(self._save_to_bytes(str_index))
        self.assertEqual(str_index, loaded)
        self.assertEqual(str_index.query(data[0], 5), loaded.query(data[0], 5))

    def test_roundtrip_reversed_edges(self):
        index, data = self._create_index(n=100, dim=8, reversed_edges=True)
        index.remove(5, hard=True)
        index.remove(6)
        loaded = self._load_from_bytes(self._save_to_bytes(index))
        self.assertEqual(index, loaded)

    def test_roundtrip_empty_index(self):
        index = HNSW(distance_func=l2_distance, seed=1)
        loaded = self._load_from_bytes(self._save_to_bytes(index))
        self.assertEqual(index, loaded)
        self.assertEqual(len(loaded), 0)

    def test_roundtrip_file_path(self):
        index, data = self._create_index(n=100, dim=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.snapshot")
            index.save_snapshot(path, distance_func_id="l2")
            loaded = HNSW.load_snapshot(path, distance_func=l2_distance, distance_func_id="l2")
        self.assertEqual(index, loaded)

    def test_distance_func_id_mismatch(self):
        index, _ = self._create_index(n=50, dim=8)
        snapshot = self._save_to_bytes(index)
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(snapshot, distance_func_id="cosine")

    def test_max_nodes_budget(self):
        index, _ = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(snapshot, max_nodes=99)
        loaded = self._load_from_bytes(snapshot, max_nodes=100)
        self.assertEqual(index, loaded)

    def test_duplicate_keys_rejected(self):
        index, _ = self._create_index(n=50, dim=8)
        snapshot = self._save_to_bytes(index)

        def duplicate_first_key(metadata):
            metadata["keys"][1] = metadata["keys"][0]

        tampered = self._edit_snapshot(snapshot, metadata_edit=duplicate_first_key)
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_tampered_neighbor_out_of_range(self):
        index, _ = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        arrays = self._read_snapshot_arrays(snapshot)
        indices = arrays["layer_0_indices"].copy()
        indices[0] = 100  # Nonexistent node.
        tampered = self._edit_snapshot(snapshot, array_edits={"layer_0_indices": indices})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_tampered_cross_layer_reference(self):
        index, _ = self._create_index(n=200, dim=16)
        self.assertGreaterEqual(len(index._graphs), 2)
        snapshot = self._save_to_bytes(index)
        arrays = self._read_snapshot_arrays(snapshot)
        members = arrays["layer_1_members"]
        non_member = int(np.flatnonzero(~members)[0])
        indices = arrays["layer_1_indices"].copy()
        # Point an existing edge of layer 1 at a node that is not a member
        # of layer 1.
        indices[0] = non_member
        tampered = self._edit_snapshot(snapshot, array_edits={"layer_1_indices": indices})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_tampered_entry_point(self):
        index, _ = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        # Entry point out of range.
        tampered = self._edit_snapshot(snapshot, metadata_edit=lambda m: m.update(entry_point=100))
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)
        # Entry point that is not a member of the top layer.
        arrays = self._read_snapshot_arrays(snapshot)
        top_members = arrays[f"layer_{len(index._graphs) - 1}_members"]
        non_member = int(np.flatnonzero(~top_members)[0])
        tampered = self._edit_snapshot(snapshot, metadata_edit=lambda m: m.update(entry_point=non_member))
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_tampered_points_shape(self):
        index, data = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        arrays = self._read_snapshot_arrays(snapshot)
        bad_points = np.zeros((100, 7), dtype=np.float64)
        tampered = self._edit_snapshot(snapshot, array_edits={"points": bad_points})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)
        bad_dtype = arrays["points"].astype(np.float32)
        tampered = self._edit_snapshot(snapshot, array_edits={"points": bad_dtype})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_tampered_deleted_shape(self):
        index, _ = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        bad_deleted = np.zeros(101, dtype=np.uint8)
        tampered = self._edit_snapshot(snapshot, array_edits={"deleted": bad_deleted})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_tampered_layer_structure(self):
        index, _ = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        arrays = self._read_snapshot_arrays(snapshot)
        # Non-monotonic index pointer array.
        indptr = arrays["layer_0_indptr"].copy()
        indptr[1] = indptr[2] + 1
        tampered = self._edit_snapshot(snapshot, array_edits={"layer_0_indptr": indptr})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)
        # Base layer no longer contains every node.
        members = arrays["layer_0_members"].copy()
        members[0] = False
        tampered = self._edit_snapshot(snapshot, array_edits={"layer_0_members": members})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_unsupported_version_rejected(self):
        index, _ = self._create_index(n=50, dim=8)
        snapshot = self._save_to_bytes(index)
        tampered = self._edit_snapshot(snapshot, metadata_edit=lambda m: m.update(version=2))
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)

    def test_corrupt_file_rejected(self):
        index, _ = self._create_index(n=50, dim=8)
        snapshot = self._save_to_bytes(index)
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(snapshot[: len(snapshot) // 2])
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(b"not a snapshot at all")

    def test_bad_snapshot_does_not_publish(self):
        index, data = self._create_index(n=100, dim=8)
        snapshot = self._save_to_bytes(index)
        existing, _ = self._create_index(n=50, dim=8, seed=1, index_seed=2)
        existing_before = existing.copy()
        arrays = self._read_snapshot_arrays(snapshot)
        indices = arrays["layer_0_indices"].copy()
        indices[0] = 100
        tampered = self._edit_snapshot(snapshot, array_edits={"layer_0_indices": indices})
        with self.assertRaises(HNSWSnapshotError):
            self._load_from_bytes(tampered)
        # The existing index is untouched and no half-loaded object exists.
        self.assertEqual(existing, existing_before)
        # A subsequent valid load still works.
        loaded = self._load_from_bytes(snapshot)
        self.assertEqual(index, loaded)

    def test_unsupported_key_type_save(self):
        index = HNSW(distance_func=l2_distance, seed=1)
        index.insert(("a", 1), np.random.rand(8))
        with self.assertRaises(TypeError):
            self._save_to_bytes(index)

    def test_unsupported_point_dtype_save(self):
        index = HNSW(distance_func=l2_distance, seed=1)
        index.insert(0, np.arange(8))
        with self.assertRaises(TypeError):
            self._save_to_bytes(index)
