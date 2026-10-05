import pickle
import unittest

import numpy as np

from datasketch import KMV
from test.utils import fake_hash_func


class TestKMV(unittest.TestCase):
    def test_retains_smallest_k_distinct_hashes(self):
        sketch = KMV(k=3, seed=7, hashfunc=fake_hash_func)

        for hv in (5, 1, 4, 1, 4):
            sketch.update(hv)

        self.assertEqual(set(sketch.digest()), {1, 4, 5})
        self.assertTrue(sketch.is_saturated())
        self.assertEqual(sketch.count(), (3 - 1) * (2**32) / 5)

        sketch.update(2)
        self.assertEqual(set(sketch.digest()), {1, 2, 4})
        self.assertEqual(sketch.threshold(), 4)
        self.assertEqual(sketch.count(), (3 - 1) * (2**32) / 4)

    def test_unsaturated_count_is_exact(self):
        sketch = KMV(k=3, seed=7, hashfunc=fake_hash_func)
        sketch.update_batch((5, 1, 5))
        self.assertFalse(sketch.is_saturated())
        self.assertEqual(sketch.count(), 2.0)

    def test_eviction_is_stable_for_larger_hashes(self):
        sketch = KMV(k=3, seed=7, hashfunc=fake_hash_func)
        sketch.update_batch((2, 9, 4, 8, 5, 1))
        self.assertEqual(set(sketch.digest()), {1, 2, 4})

    def test_hash_collisions_are_deduplicated(self):
        sketch = KMV(k=4, seed=7, hashfunc=lambda value: 1)
        sketch.update_batch((b"a", b"b", b"c"))
        self.assertEqual(set(sketch.digest()), {1})
        self.assertEqual(sketch.count(), 1.0)
        self.assertFalse(sketch.is_saturated())

    def test_saturated_estimate(self):
        sketch = KMV(k=3, seed=7, hashfunc=fake_hash_func)
        sketch.update_batch((1, 2, 4))
        expected = (3 - 1) * (2**32) / 4
        self.assertEqual(sketch.count(), expected)

    def test_empty_and_zero_threshold_bounds(self):
        sketch = KMV(k=2, seed=7, hashfunc=fake_hash_func)
        self.assertEqual(sketch.count(), 0.0)
        self.assertEqual(sketch.jaccard(sketch.copy()), 1.0)
        with self.assertRaisesRegex(ValueError, "zero threshold"):
            sketch._estimate(2, 0)

        with self.assertRaisesRegex(ValueError, "no smaller than 2"):
            KMV(k=1)

    def test_invalid_hash_value(self):
        sketch = KMV(k=2, seed=7, hashfunc=fake_hash_func)
        with self.assertRaises(ValueError):
            sketch.update(2**32)

    def test_merge_common_capacity_and_deduplication(self):
        left = KMV(k=4, seed=7, hashfunc=fake_hash_func)
        right = KMV(k=3, seed=7, hashfunc=fake_hash_func)
        left.update_batch((1, 3, 5, 9))
        right.update_batch((2, 3, 4))

        left.merge(right)

        self.assertEqual(left.k, 3)
        self.assertEqual(set(left.digest()), {1, 2, 3})

    def test_similarity_lower_capacity_saturated_sketch(self):
        left = KMV(k=3, seed=7, hashfunc=fake_hash_func)
        right = KMV(k=4, seed=7, hashfunc=fake_hash_func)
        left.update_batch((1, 2, 3))
        right.update_batch((1, 2, 4, 5))

        self.assertEqual(left.jaccard(right), 2.0 / 3.0)
        self.assertGreaterEqual(left.intersection_count(right), 0.0)

    def test_union_common_capacity(self):
        left = KMV(k=4, seed=7, hashfunc=fake_hash_func)
        right = KMV(k=3, seed=7, hashfunc=fake_hash_func)
        left.update_batch((1, 3, 5, 9))
        right.update_batch((2, 3, 4))

        merged = KMV.union(left, right)

        self.assertEqual(merged.k, 3)
        self.assertEqual(set(merged.digest()), {1, 2, 3})
        self.assertEqual(set(left.digest()), {1, 3, 5, 9})

    def test_exact_intersection_and_similarity_below_capacity(self):
        left = KMV(k=8, seed=7, hashfunc=fake_hash_func)
        right = KMV(k=8, seed=7, hashfunc=fake_hash_func)
        left.update_batch((1, 2, 3, 4))
        right.update_batch((3, 4, 5))

        self.assertEqual(left.intersection_count(right), 2.0)
        self.assertEqual(left.jaccard(right), 2.0 / 5.0)

    def test_disjoint_saturated_sets_have_nonnegative_intersection(self):
        left = KMV(k=2, seed=7, hashfunc=fake_hash_func)
        right = KMV(k=2, seed=7, hashfunc=fake_hash_func)
        left.update_batch((1, 100))
        right.update_batch((2, 99))

        self.assertEqual(left.intersection_count(right), 0.0)
        self.assertEqual(left.jaccard(right), 0.0)

    def test_rejects_different_seed_scheme_width_and_hash_function(self):
        base = KMV(k=2, seed=1, hashfunc=fake_hash_func)
        cases = [
            KMV(k=2, seed=2, hashfunc=fake_hash_func),
            KMV(k=2, seed=1, hashfunc=lambda hv: hv),
            KMV(k=2, seed=1, hashfunc=fake_hash_func, width=64),
            KMV(k=2, seed=1, hash_scheme="sha1-64"),
        ]
        for other in cases:
            with self.subTest(other=other):
                with self.assertRaises(ValueError):
                    base.merge(other)

    def test_copy_equality_and_pickle(self):
        sketch = KMV(k=4, seed=7, hash_scheme="sha1-64")
        sketch.update_batch((b"a", b"b"))
        copied = sketch.copy()

        self.assertEqual(sketch, copied)
        copied.update(b"c")
        self.assertNotEqual(sketch, copied)

        restored = pickle.loads(pickle.dumps(sketch))
        self.assertEqual(restored, sketch)
        np.testing.assert_array_equal(restored.digest(), sketch.digest())

    def test_synthetic_estimates_stay_within_error_bounds(self):
        rng = np.random.default_rng(20261005)
        shared = rng.choice(200_000, size=5_000, replace=False)
        left_ids = np.concatenate([shared, rng.choice(200_000, size=5_000, replace=False)])
        right_ids = np.concatenate([shared, rng.choice(200_000, size=5_000, replace=False)])

        left = KMV(k=2_048, seed=42, hash_scheme="sha1-64")
        right = KMV(k=2_048, seed=42, hash_scheme="sha1-64")
        left.update_batch(str(item).encode() for item in left_ids)
        right.update_batch(str(item).encode() for item in right_ids)

        for count in (left.count(), right.count()):
            self.assertGreater(count, 8_500)
            self.assertLess(count, 11_500)

        intersection = left.intersection_count(right)
        self.assertGreater(intersection, 3_500)
        self.assertLess(intersection, 6_500)

        similarity = left.jaccard(right)
        self.assertGreater(similarity, 0.22)
        self.assertLess(similarity, 0.45)


if __name__ == "__main__":
    unittest.main()
