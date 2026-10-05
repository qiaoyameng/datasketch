import unittest
import warnings

import numpy as np
import pytest

from datasketch.hnsw import HNSW
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

    def test_query_filter_traverses_unqualified_nodes(self):
        # A small 1-D graph in which the only qualifying point is the
        # farthest one: the search must navigate through unqualified
        # nodes to reach it.
        points = [np.array([float(i)]) for i in range(1, 11)]
        index = HNSW(distance_func=l2_distance, m=4, ef_construction=50, seed=42)
        for key, point in enumerate(points):
            index.insert(key, point)
        results = index.query(np.array([0.0]), k=3, ef=10, filter_func=lambda key: key == 9)
        self.assertEqual([key for key, _ in results], [9])
        self.assertAlmostEqual(results[0][1], 10.0)
        # All reachable nodes were explored, so the search did not stop
        # early even though fewer than k results qualify.
        self.assertFalse(results.stopped_early)

    def test_query_filter_budget_exhaustion(self):
        # Many unqualified points near the query exhaust the candidate
        # budget before any qualifying point is reached.
        points = [np.array([float(i)]) for i in range(1, 21)]
        index = HNSW(distance_func=l2_distance, m=4, ef_construction=50, seed=42)
        for key, point in enumerate(points):
            index.insert(key, point)
        qualifying = set(range(15, 20))
        filter_func = lambda key: key in qualifying
        # With a small ef, the budget is spent on closer unqualified points.
        results = index.query(np.array([0.0]), k=3, ef=5, filter_func=filter_func)
        self.assertLess(len(results), 3)
        self.assertTrue(results.stopped_early)
        # With a large enough budget, the qualifying points are found.
        results = index.query(np.array([0.0]), k=3, ef=20, filter_func=filter_func)
        self.assertEqual([key for key, _ in results], [15, 16, 17])
        self.assertFalse(results.stopped_early)

    def test_query_filter_matches_exhaustive_search(self):
        rng = np.random.RandomState(42)
        data = rng.random_sample((200, 8))
        index = HNSW(distance_func=l2_distance, m=16, ef_construction=100, seed=42)
        index.update({i: data[i] for i in range(len(data))})
        allowed = set(range(0, len(data), 3))
        k = 10
        for query_point in data[:10]:
            results = index.query(query_point, k=k, ef=len(data), filter_func=lambda key: key in allowed)
            expected = sorted(
                ((key, l2_distance(query_point, data[key])) for key in allowed),
                key=lambda item: item[1],
            )[:k]
            self.assertEqual([key for key, _ in results], [key for key, _ in expected])
            for (_, dist), (_, expected_dist) in zip(results, expected):
                self.assertAlmostEqual(dist, expected_dist)
        # Without k, all qualifying neighbors are returned.
        results = index.query(data[0], ef=len(data), filter_func=lambda key: key in allowed)
        expected = sorted(
            ((key, l2_distance(data[0], data[key])) for key in allowed),
            key=lambda item: item[1],
        )
        self.assertEqual(len(results), len(allowed))
        self.assertEqual([key for key, _ in results], [key for key, _ in expected])

    def test_query_without_filter_unchanged(self):
        data = self._create_random_points()
        index = self._create_index(data)
        for query_point in data[:10]:
            expected = index.query(query_point, 10)
            self.assertIsInstance(expected, list)
            self.assertEqual(expected, index.query(query_point, 10, filter_func=None))
            self.assertEqual(expected, index.query(query_point, 10, filter_func=lambda key: True))

    def test_query_filter_excludes_soft_removed(self):
        data = self._create_random_points()
        index = self._create_index(data)
        index.remove(0)
        results = index.query(data[0], 10, filter_func=lambda key: True)
        self.assertNotIn(0, [key for key, _ in results])

    def test_query_tied_distances_with_unorderable_keys(self):
        # Mixed-type keys cannot be ordered; equal distances must not
        # trigger comparisons between keys.
        keys = [0, "a", 1, "b", 2, "c", 3, "d"]
        index = HNSW(distance_func=l2_distance, m=4, ef_construction=10, seed=42)
        for key in keys:
            index.insert(key, np.zeros(2))
        results = index.query(np.zeros(2), k=4)
        self.assertEqual(len(results), 4)
        results = index.query(np.zeros(2), k=4, filter_func=lambda key: isinstance(key, str))
        self.assertEqual(sorted(key for key, _ in results), ["a", "b", "c", "d"])
        # Update a point to trigger graph repair with tied distances.
        index[0] = np.array([1.0, 1.0])
        results = index.query(np.zeros(2), k=4)
        self.assertEqual(len(results), 4)


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
