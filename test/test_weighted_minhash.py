import pickle
import unittest
from unittest import mock

import numpy as np
import scipy as sp
import scipy.sparse

from datasketch.weighted_minhash import (
    UpdatableWeightedMinHash,
    WeightedMinHash,
    WeightedMinHashGenerator,
    WeightedMinHashUpdater,
)


class TestWeightedMinHash(unittest.TestCase):
    def test_pickle(self):
        mg = WeightedMinHashGenerator(4, 10, 1)
        m = mg.minhash([1, 2, 3, 4])
        p = pickle.loads(pickle.dumps(m))
        self.assertEqual(p.seed, m.seed)
        self.assertTrue(np.array_equal(p.hashvalues, m.hashvalues))


class TestWeightedMinHashGenerator(unittest.TestCase):
    def test_init(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        self.assertEqual(len(mg.rs), 4)
        self.assertEqual(len(mg.ln_cs), 4)
        self.assertEqual(len(mg.betas), 4)
        self.assertEqual(mg.seed, 1)
        self.assertEqual(mg.sample_size, 4)

    def test_minhash(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m = mg.minhash([1, 3])
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_does_not_mutate_float32_input(self):
        mg = WeightedMinHashGenerator(3, 4, 1)
        v = np.array([1, 0, 3], dtype=np.float32)
        expected = v.copy()
        mg.minhash(v)
        np.testing.assert_array_equal(v, expected)

    def test_minhash_many_dense_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([1, 3]).reshape(1, 2))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_dense_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1, 3], [1, 3]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_dense_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        m_list = mg.minhash_many(np.array([[1, 3], [0, 0]]))
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)

    def test_minhash_many_sparse_onerow(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([1, 3]).reshape(1, 2))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 1)
        self.assertIsInstance(m_list[0], WeightedMinHash)
        self.assertEqual(len(m_list[0].hashvalues), 4)
        self.assertEqual(len(m_list[0]), 4)
        self.assertTrue(m_list[0].hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1, 3], [1, 3]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)
        for m in m_list:
            self.assertIsInstance(m, WeightedMinHash)
            self.assertEqual(len(m.hashvalues), 4)
            self.assertEqual(len(m), 4)
            self.assertTrue(m.hashvalues.dtype == int)

    def test_minhash_many_sparse_tworows_with_null(self):
        mg = WeightedMinHashGenerator(2, 4, 1)
        X = sp.sparse.csr_matrix(np.array([[1, 3], [0, 0]]))
        m_list = mg.minhash_many(X)
        self.assertIsInstance(m_list, list)
        self.assertEqual(len(m_list), 2)

        m = m_list[0]
        self.assertIsInstance(m, WeightedMinHash)
        self.assertEqual(len(m.hashvalues), 4)
        self.assertEqual(len(m), 4)
        self.assertTrue(m.hashvalues.dtype == int)

        self.assertIs(m_list[1], None)


class TestUpdatableWeightedMinHash(unittest.TestCase):
    def assert_matches_full_minhash(self, generator, updatable, vector):
        if np.count_nonzero(vector) == 0:
            self.assertIsNone(updatable.minhash())
            self.assertIsNone(updatable.digest())
            return

        expected = generator.minhash(vector)
        actual = updatable.minhash()
        self.assertIsInstance(actual, WeightedMinHash)
        np.testing.assert_array_equal(actual.hashvalues, expected.hashvalues)

    def test_updates_match_full_minhash_including_all_zeros(self):
        generator = WeightedMinHashGenerator(6, sample_size=16, seed=123)
        vector = np.array([0, 3, 0, 2, 0, 0], dtype=np.float32)
        updatable = generator.updatable_minhash(vector)

        changes = [
            (0, 1),
            (2, 4),
            (3, 1),
            (1, 0),
            (2, 0),
            (0, 0),
            (3, 0),
        ]
        self.assert_matches_full_minhash(generator, updatable, vector)
        for index, weight in changes:
            vector[index] = weight
            self.assertIs(updatable.update(index, weight), updatable)
            np.testing.assert_array_equal(updatable.weights, vector)
            self.assert_matches_full_minhash(generator, updatable, vector)

        self.assertTrue(updatable.empty)

        vector[4] = 5
        updatable.update(4, 5)
        self.assert_matches_full_minhash(generator, updatable, vector)

    def test_empty_initial_vector_can_be_populated_and_cleared(self):
        generator = WeightedMinHashGenerator(3, sample_size=8, seed=42)
        vector = np.zeros(3, dtype=np.float32)
        updatable = UpdatableWeightedMinHash.from_generator(generator)

        self.assertTrue(updatable.empty)
        self.assertIsNone(updatable.minhash())

        updatable.set_weight(1, 2)
        vector[1] = 2
        self.assert_matches_full_minhash(generator, updatable, vector)

        updatable.update(1, 0)
        vector[1] = 0
        self.assertTrue(updatable.empty)
        self.assertIsNone(updatable.digest())

    def test_single_dimension_update_does_not_regenerate_random_matrices(self):
        generator = WeightedMinHashGenerator(5, sample_size=8, seed=7)
        updatable = generator.updatable_minhash(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        rs, ln_cs, betas = generator.rs, generator.ln_cs, generator.betas

        class CountingRandomState(np.random.RandomState):
            gamma_calls = 0
            uniform_calls = 0

            def gamma(self, *args, **kwargs):
                type(self).gamma_calls += 1
                return super().gamma(*args, **kwargs)

            def uniform(self, *args, **kwargs):
                type(self).uniform_calls += 1
                return super().uniform(*args, **kwargs)

        with mock.patch("datasketch.weighted_minhash.np.random.RandomState", CountingRandomState):
            updatable.update(2, 6)

        self.assertEqual(CountingRandomState.gamma_calls, 0)
        self.assertEqual(CountingRandomState.uniform_calls, 0)
        self.assertIs(generator.rs, rs)
        self.assertIs(generator.ln_cs, ln_cs)
        self.assertIs(generator.betas, betas)

        expected = generator.minhash(np.array([1, 2, 6, 4, 5], dtype=np.float32))
        np.testing.assert_array_equal(updatable.minhash().hashvalues, expected.hashvalues)

    def test_invalid_update_is_atomic(self):
        generator = WeightedMinHashGenerator(4, sample_size=8, seed=9)
        vector = np.array([1, 2, 0, 4], dtype=np.float32)
        updatable = generator.updatable_minhash(vector)
        before = pickle.loads(pickle.dumps(updatable))

        with self.assertRaises(IndexError):
            updatable.update(-1, 2)
        with self.assertRaises(IndexError):
            updatable.update(4, 2)
        with self.assertRaises(TypeError):
            updatable.update(1.0, 2)
        for invalid_weight in (-1, np.nan, np.inf, -np.inf, [2]):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaises((TypeError, ValueError)):
                    updatable.update(1, invalid_weight)

        self.assertEqual(updatable, before)
        np.testing.assert_array_equal(updatable.weights, vector)
        self.assert_matches_full_minhash(generator, updatable, vector)

    def test_invalid_initial_vector_is_rejected(self):
        for invalid_vector in (
            [-1, 1],
            [np.nan, 1],
            [np.inf, 1],
            [[1, 1]],
            [1],
        ):
            with self.subTest(invalid_vector=invalid_vector):
                with self.assertRaises((TypeError, ValueError)):
                    UpdatableWeightedMinHash(2, sample_size=4, seed=1, v=invalid_vector)

    def test_candidate_memory_scales_with_dimension_and_sample_size(self):
        updatable = UpdatableWeightedMinHash(7, sample_size=11, seed=1, v=np.arange(7, dtype=np.float32))
        self.assertEqual(updatable._t.shape, (11, 7))
        self.assertEqual(updatable._ln_a.shape, (11, 7))
        self.assertEqual(updatable.candidate_memory_bytes, 11 * 7 * 2 * np.dtype(np.float32).itemsize)

    def test_copy_constructor_and_alias(self):
        generator = WeightedMinHashGenerator(3, sample_size=5, seed=3)
        vector = np.array([1, 2, 3], dtype=np.float32)
        updatable = WeightedMinHashUpdater.from_generator(generator, vector)
        copied = updatable.copy()

        self.assertEqual(updatable, copied)
        self.assertIsNot(updatable, copied)
        copied.update(0, 5)
        self.assertNotEqual(updatable, copied)
        np.testing.assert_array_equal(updatable.weights, vector)


if __name__ == "__main__":
    unittest.main()
