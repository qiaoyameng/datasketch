import unittest
import warnings
from collections import OrderedDict

import numpy as np
import pytest

from datasketch.hnsw import HNSW
from datasketch.hnsw import _Layer, _Node
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

    def _build_bridge_index(self):
        # A tiny graph in 1-D: E (the entry point) is connected only to B,
        # which in turn connects to T. The query is at 0, so the nearest
        # qualified answer T is reachable only by navigating through B.
        hnsw = HNSW(distance_func=l2_distance, m=16, ef_construction=10, seed=1)
        points = {"E": 10.0, "B": 1.0, "T": 2.0}
        hnsw._nodes = OrderedDict((key, _Node(key, np.array(value))) for key, value in points.items())
        layer = _Layer("E")
        layer["E"] = {"B": l2_distance(np.array(10.0), np.array(1.0))}
        layer["B"] = {
            "E": l2_distance(np.array(1.0), np.array(10.0)),
            "T": l2_distance(np.array(1.0), np.array(2.0)),
        }
        layer["T"] = {"B": l2_distance(np.array(2.0), np.array(1.0))}
        hnsw._graphs = [layer]
        hnsw._entry_point = "E"
        return hnsw

    def test_filter_navigates_through_rejected_nodes(self):
        hnsw = self._build_bridge_index()
        query = np.array(0.0)
        # T is the only allowed key and is reachable solely via B, which the
        # filter rejects. It must still be found.
        results, exhausted = hnsw.query(
            query,
            k=1,
            ef=10,
            filter_func=lambda key: key == "T",
            return_exhausted=True,
        )
        self.assertEqual(results, [("T", l2_distance(query, np.array(2.0)))])
        self.assertFalse(exhausted)

    def test_filter_only_returns_allowed_keys(self):
        hnsw = self._build_bridge_index()
        query = np.array(0.0)
        results = hnsw.query(
            query,
            k=5,
            ef=10,
            filter_func=lambda key: key in ("E", "B"),
        )
        self.assertEqual([key for key, _ in results], ["B", "E"])
        for j in range(len(results) - 1):
            self.assertLessEqual(results[j][1], results[j + 1][1])

    def test_filter_returns_actual_count_when_fewer_than_k(self):
        hnsw = self._build_bridge_index()
        query = np.array(0.0)
        results, exhausted = hnsw.query(
            query,
            k=10,
            ef=10,
            filter_func=lambda key: key == "T",
            return_exhausted=True,
        )
        self.assertEqual(results, [("T", l2_distance(query, np.array(2.0)))])
        self.assertFalse(exhausted)

    def test_filter_empty_qualified_set(self):
        hnsw = self._build_bridge_index()
        results = hnsw.query(np.array(0.0), k=5, ef=10, filter_func=lambda key: False)
        self.assertEqual(results, [])
        results, exhausted = hnsw.query(
            np.array(0.0),
            k=5,
            ef=10,
            filter_func=lambda key: False,
            return_exhausted=True,
        )
        self.assertEqual(results, [])
        # The whole connected graph is explored within the budget.
        self.assertFalse(exhausted)

    def test_filter_k_zero(self):
        hnsw = self._build_bridge_index()
        self.assertEqual(hnsw.query(np.array(0.0), k=0, ef=10, filter_func=lambda key: True), [])

    def test_filter_exhausted_budget(self):
        hnsw = self._build_bridge_index()
        # ef=1 lets only the entry point E be explored, so k=3 cannot be met.
        results, exhausted = hnsw.query(
            np.array(0.0),
            k=3,
            ef=1,
            filter_func=lambda key: True,
            return_exhausted=True,
        )
        self.assertTrue(exhausted)
        self.assertLess(len(results), 3)

    def test_filter_exhausted_budget_large_graph(self):
        rng = np.random.RandomState(99)
        data = rng.rand(500, 8)
        hnsw = HNSW(distance_func=l2_distance, m=8, ef_construction=100, seed=11)
        for key, point in enumerate(data):
            hnsw.insert(key, point)
        query = rng.rand(8)
        results, exhausted = hnsw.query(
            query,
            k=50,
            ef=1,
            filter_func=lambda key: True,
            return_exhausted=True,
        )
        self.assertTrue(exhausted)
        self.assertLess(len(results), 50)
        results, exhausted = hnsw.query(
            query,
            k=10,
            ef=500,
            filter_func=lambda key: True,
            return_exhausted=True,
        )
        self.assertEqual(len(results), 10)
        self.assertFalse(exhausted)

    def test_filter_matches_brute_force(self):
        rng = np.random.RandomState(2024)
        n, dim = 200, 6
        data = rng.rand(n, dim)
        hnsw = HNSW(distance_func=l2_distance, m=8, ef_construction=100, seed=5)
        for key, point in enumerate(data):
            hnsw.insert(key, point)
        tenant_of = {key: key % 4 for key in range(n)}
        for _ in range(20):
            query = rng.rand(dim)
            allowed_tenants = set(rng.choice(4, size=int(rng.randint(1, 5)), replace=False).tolist())
            allowed = {key for key in range(n) if tenant_of[key] in allowed_tenants}
            k = 7
            results = hnsw.query(
                query,
                k=k,
                ef=n,
                filter_func=lambda key: key in allowed,
            )
            expected = sorted((l2_distance(query, data[key]), key) for key in allowed)[:k]
            self.assertEqual(
                [(key, dist) for dist, key in expected],
                results,
            )
            self.assertTrue(all(key in allowed for key, _ in results))
            self.assertLessEqual(len(results), k)

    def test_filter_soft_deleted_nodes_traversed_but_not_returned(self):
        rng = np.random.RandomState(7)
        data = rng.rand(40, 3)
        hnsw = HNSW(distance_func=l2_distance, m=6, ef_construction=60, seed=3)
        for key, point in enumerate(data):
            hnsw.insert(key, point)
        query = rng.rand(3)
        allowed = {2, 17, 31}
        hnsw.remove(17)
        results = hnsw.query(query, k=10, ef=40, filter_func=lambda key: key in allowed)
        self.assertTrue(all(key != 17 for key, _ in results))
        expected = sorted((l2_distance(query, data[key]), key) for key in (2, 31))
        self.assertEqual(results, [(key, dist) for dist, key in expected])

    def test_no_filter_preserves_behavior(self):
        data = self._create_random_points()
        hnsw = self._create_index(data)
        for i in (0, 10, 50):
            plain = hnsw.query(data[i], k=10)
            flagged, exhausted = hnsw.query(data[i], k=10, return_exhausted=True)
            self.assertEqual(plain, flagged)
            self.assertFalse(exhausted)

    def test_equal_distances_with_incomparable_keys(self):
        # str and tuple keys raise TypeError when compared; equal distances
        # must never force such a comparison during search or sorting.
        hnsw = HNSW(distance_func=l2_distance, m=4, ef_construction=20, seed=7)
        points = {
            "a": np.array([1.0]),
            "b": np.array([-1.0]),
            (1,): np.array([1.0]),
            (2,): np.array([-1.0]),
            "z": np.array([0.5]),
            "y": np.array([-0.5]),
        }
        for key, point in points.items():
            hnsw.insert(key, point)
        query = np.array([0.0])
        self.assertEqual(len(hnsw.query(query, k=6, ef=50)), 6)
        self.assertEqual(len(hnsw.query(query, k=4, ef=50, filter_func=lambda key: True)), 4)
        self.assertEqual(len(hnsw.query(query, k=None, ef=50)), 6)


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
