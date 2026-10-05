import math
import unittest

import numpy as np

from datasketch import KMV, KMinimumValues


def test_kmv_alias():
    assert KMinimumValues is KMV


def identity_hash64(value):
    return value


def another_hash64(value):
    return value + 1


class TestKMV(unittest.TestCase):
    def test_retains_smallest_distinct_hashes(self):
        sketch = KMV(k=4, seed=7, width=64, hashfunc=identity_hash64)
        sketch.update_batch([5, 10, 5, 2, 8, 1, 20])

        self.assertEqual(sketch.digest().tolist(), [1, 2, 5, 8])
        self.assertEqual(sketch.threshold, 8)
        self.assertTrue(sketch.is_full())
        self.assertEqual(len(sketch), 4)

    def test_exact_count_before_capacity_and_empty_boundary(self):
        sketch = KMV(k=4, width=64, hashfunc=identity_hash64)
        self.assertEqual(sketch.count(), 0.0)
        self.assertTrue(sketch.is_empty())

        sketch.update_batch([3, 3, 1])
        self.assertEqual(sketch.count(), 2.0)
        self.assertFalse(sketch.is_full())

    def test_count_estimate_after_capacity(self):
        sketch = KMV(k=4, width=64, hashfunc=identity_hash64, hashvalues=[0, 1, 2, 3])

        self.assertEqual(sketch.count(), 3.0 * (2**64) / 4.0)
        self.assertTrue(math.isfinite(sketch.count()))

    def test_rejects_invalid_hash_values(self):
        sketch = KMV(k=4, width=32, hashfunc=identity_hash64)
        with self.assertRaises(ValueError):
            sketch.update(1 << 32)
        with self.assertRaises(ValueError):
            sketch.update(-1)

    def test_hash_collision_is_deduped(self):

        def constant_hash(_value):
            return 42

        sketch = KMV(k=4, width=64, hashfunc=constant_hash)
        sketch.update_batch([b"a", b"b", b"c"])

        self.assertEqual(len(sketch), 1)
        self.assertEqual(sketch.count(), 1.0)

    def test_merge_reduces_to_common_capacity(self):
        first = KMV(k=4, width=64, seed=3, hashfunc=identity_hash64)
        second = KMV(k=3, width=64, seed=3, hashfunc=identity_hash64)
        first.update_batch([1, 4, 5, 9])
        second.update_batch([2, 4, 10])

        first.merge(second)

        self.assertEqual(first.k, 3)
        self.assertEqual(first.digest().tolist(), [1, 2, 4])
        self.assertEqual(len(first), 3)

    def test_union_classmethod_deduplicates(self):
        first = KMV(k=3, width=64, seed=3, hashfunc=identity_hash64)
        second = KMV(k=3, width=64, seed=3, hashfunc=identity_hash64)
        first.update_batch([1, 4, 5])
        second.update_batch([1, 2, 4])

        merged = KMV.union(first, second)

        self.assertEqual(merged.k, 3)
        self.assertEqual(merged.digest().tolist(), [1, 2, 4])

    def test_merge_rejects_different_seed_scheme_width_or_function(self):
        base = KMV(k=3, seed=1, width=64, hashfunc=identity_hash64)
        different_seed = KMV(k=3, seed=2, width=64, hashfunc=identity_hash64)
        different_width = KMV(k=3, seed=1, width=32, hashfunc=identity_hash64)
        different_scheme = KMV(k=3, seed=1, width=64, hashfunc=identity_hash64, hash_scheme="other")
        different_function = KMV(k=3, seed=1, width=64, hashfunc=another_hash64)

        for incompatible in (different_seed, different_width, different_scheme, different_function):
            with self.assertRaises(ValueError):
                base.merge(incompatible)
            with self.assertRaises(ValueError):
                base.jaccard(incompatible)

    def test_exact_similarity_when_union_is_below_capacity(self):
        first = KMV(k=10, width=64, seed=1, hashfunc=identity_hash64)
        second = KMV(k=10, width=64, seed=1, hashfunc=identity_hash64)
        first.update_batch([1, 2, 3, 4, 5])
        second.update_batch([4, 5, 6, 7, 8])

        self.assertEqual(first.union_count(second), 8.0)
        self.assertEqual(first.intersection_count(second), 2.0)
        self.assertEqual(first.jaccard(second), 0.25)

    def test_estimates_are_non_negative_and_bounded(self):
        first = KMV(k=4, width=64, seed=1, hashfunc=identity_hash64)
        second = KMV(k=4, width=64, seed=1, hashfunc=identity_hash64)
        first.update_batch([1, 3, 5, 7])
        second.update_batch([2, 4, 6, 8])

        self.assertEqual(first.intersection_count(second), 0.0)
        self.assertEqual(first.jaccard(second), 0.0)

        identical = first.copy()
        self.assertEqual(first.jaccard(identical), 1.0)

    def test_copy_clear_and_equality(self):
        sketch = KMV(k=4, seed=9, width=64)
        sketch.update_batch([b"a", b"b"])
        copied = sketch.copy()

        self.assertEqual(sketch, copied)
        self.assertIs(sketch.hashfunc, copied.hashfunc)
        copied.update(b"c")
        self.assertNotEqual(sketch, copied)
        copied.clear()
        self.assertTrue(copied.is_empty())
        self.assertNotEqual(sketch, copied)

    def test_synthetic_sets_have_reasonable_estimation_error(self):
        true_a = 2000
        true_b = 2200
        true_intersection = 1200
        true_union = true_a + true_b - true_intersection
        true_jaccard = true_intersection / true_union

        for seed in (101, 202, 303):
            rng = np.random.RandomState(seed)
            values_a = set(rng.permutation(np.arange(10_000))[:true_a].tolist())
            values_b = set(sorted(values_a)[:true_intersection])
            values_b.update(rng.permutation(np.arange(10_000, 20_000))[: true_b - true_intersection])

            sketch_a = KMV(k=512, seed=seed, width=64)
            sketch_b = KMV(k=512, seed=seed, width=64)
            sketch_a.update_batch(("item-%d" % value).encode("ascii") for value in values_a)
            sketch_b.update_batch(("item-%d" % value).encode("ascii") for value in values_b)

            self.assertLess(abs(sketch_a.count() - true_a) / true_a, 0.20)
            self.assertLess(abs(sketch_b.count() - true_b) / true_b, 0.20)
            self.assertLess(abs(sketch_a.union_count(sketch_b) - true_union) / true_union, 0.20)
            self.assertLess(abs(sketch_a.intersection_count(sketch_b) - true_intersection) / true_intersection, 0.35)
            self.assertLess(abs(sketch_a.jaccard(sketch_b) - true_jaccard), 0.10)


if __name__ == "__main__":
    unittest.main()
