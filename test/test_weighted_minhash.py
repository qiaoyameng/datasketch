import pickle
import unittest
from unittest.mock import patch

import numpy as np
import scipy as sp
import scipy.sparse

from datasketch.weighted_minhash import (
    UpdatableWeightedMinHash,
    WeightedMinHash,
    WeightedMinHashGenerator,
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
    def test_updatable_minhash_matches_full_minhash_after_each_update(self):
        dim = 6
        sample_size = 32
        seed = 3
        generator = WeightedMinHashGenerator(dim, sample_size, seed)
        full_generator = WeightedMinHashGenerator(dim, sample_size, seed)
        values = np.array([0, 1, 0, 4, 0, 0], dtype=np.float32)
        sketch = generator.updatable_minhash(values)

        updates = [
            (0, 2),
            (1, 0),
            (3, 2),
            (3, 7),
            (0, 0),
            (3, 0),
            (2, 5),
        ]

        expected = full_generator.minhash(values)
        np.testing.assert_array_equal(sketch.digest(), expected.hashvalues)

        for dimension, weight in updates:
            values[dimension] = weight
            sketch.update(dimension, weight)

            if np.any(values):
                expected = full_generator.minhash(values)
                np.testing.assert_array_equal(sketch.digest(), expected.hashvalues)
                self.assertFalse(sketch.is_empty)
                self.assertEqual(sketch.minhash(), expected)
            else:
                self.assertTrue(sketch.is_empty)
                self.assertIsNone(sketch.digest())
                self.assertIsNone(sketch.minhash())

        np.testing.assert_array_equal(sketch.weights(), values)

    def test_update_does_not_regenerate_random_parameters(self):
        calls = []

        class CountingRandomState(np.random.RandomState):
            def gamma(self, *args, **kwargs):
                calls.append("gamma")
                return super().gamma(*args, **kwargs)

            def uniform(self, *args, **kwargs):
                calls.append("uniform")
                return super().uniform(*args, **kwargs)

        with patch("datasketch.weighted_minhash.np.random.RandomState", CountingRandomState):
            generator = WeightedMinHashGenerator(6, 32, 3)
            self.assertEqual(calls.count("gamma"), 2)
            self.assertEqual(calls.count("uniform"), 1)

            calls.clear()
            sketch = generator.updatable_minhash(np.array([0, 1, 0, 4, 0, 0], dtype=np.float32))
            sketch.update(0, 2)
            sketch.update(1, 0)
            sketch.update(3, 0)

        self.assertEqual(calls, [])

    def test_candidate_state_scales_with_dimension_and_sample_count(self):
        generator = WeightedMinHashGenerator(7, 11, 3)
        sketch = UpdatableWeightedMinHash(generator, np.array([0, 1, 0, 4, 0, 0, 0], dtype=np.float32))

        self.assertEqual(sketch._candidate_t.shape, (11, 7))
        self.assertEqual(sketch._candidate_ln_a.shape, (11, 7))
        self.assertEqual(sketch._candidate_t.dtype, np.float32)
        self.assertEqual(sketch._candidate_ln_a.dtype, np.float32)

    def test_invalid_update_leaves_all_state_unchanged(self):
        generator = WeightedMinHashGenerator(4, 12, 3)
        sketch = generator.updatable_minhash(np.array([0, 1, 0, 4], dtype=np.float32))
        invalid_updates = [
            (-1, 1),
            (4, 1),
            (1.5, 1),
            (1, -1),
            (1, np.nan),
            (1, np.inf),
            (1, -np.inf),
            (1, np.array([1])),
        ]

        for dimension, weight in invalid_updates:
            with self.subTest(dimension=dimension, weight=weight):
                weights_before = sketch.weights()
                hashvalues_before = sketch.digest()
                candidate_t_before = sketch._candidate_t.copy()
                candidate_ln_a_before = sketch._candidate_ln_a.copy()

                with self.assertRaises(ValueError):
                    sketch.update(dimension, weight)

                np.testing.assert_array_equal(sketch.weights(), weights_before)
                np.testing.assert_array_equal(sketch.digest(), hashvalues_before)
                np.testing.assert_array_equal(sketch._candidate_t, candidate_t_before)
                np.testing.assert_array_equal(sketch._candidate_ln_a, candidate_ln_a_before)

    def test_initial_vector_rejects_invalid_weights(self):
        generator = WeightedMinHashGenerator(3, 4, 1)

        for values in ([1, -1, 0], [1, np.nan, 0], [1, np.inf, 0], [1, 2], [[1, 2, 0]]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    UpdatableWeightedMinHash(generator, values)


if __name__ == "__main__":
    unittest.main()
