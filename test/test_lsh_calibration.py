import unittest

import numpy as np

from datasketch import MinHash, MinHashLSH
from datasketch.lsh_calibration import (
    LSHCalibrationPair,
    calibrate_lsh_params,
)


def make_minhash(values):
    values = np.asarray(values)
    minhash = MinHash(num_perm=len(values))
    minhash.hashvalues = values.astype(minhash.hashvalues.dtype)
    return minhash


def brute_force_best(pairs, num_perm, miss_cost, false_alarm_cost):
    """Independent reference implementation used to cross-check the tool."""
    total_weight = sum(pair[3] for pair in pairs)
    best = None
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            loss = 0.0
            for query, candidate, label, weight in pairs:
                recalled = any(
                    all(query[band * r + i] == candidate[band * r + i] for i in range(r))
                    for band in range(b)
                )
                if label and not recalled:
                    loss += miss_cost * weight
                elif not label and recalled:
                    loss += false_alarm_cost * weight
            loss /= total_weight
            if best is None or loss < best[0]:
                best = (loss, b, r)
    return best


def random_signature(rng, num_perm):
    return rng.randint(0, 2**31, size=num_perm).astype(np.uint64)


def similar_pair(rng, num_perm, shared_ratio):
    query = random_signature(rng, num_perm)
    candidate = random_signature(rng, num_perm)
    shared = rng.rand(num_perm) < shared_ratio
    candidate[shared] = query[shared]
    return query, candidate


def make_dataset(rng, num_perm, num_positive, num_negative):
    pairs = []
    for _ in range(num_positive):
        query, candidate = similar_pair(rng, num_perm, shared_ratio=0.8)
        pairs.append((query, candidate, True, 1.0))
    for _ in range(num_negative):
        query, candidate = similar_pair(rng, num_perm, shared_ratio=0.1)
        pairs.append((query, candidate, False, 1.0))
    return pairs


class TestCalibrateLSHParams(unittest.TestCase):
    def test_exhaustive_small_budget_matches_brute_force(self):
        num_perm = 12
        rng = np.random.RandomState(42)
        pairs = make_dataset(rng, num_perm, num_positive=30, num_negative=30)
        result = calibrate_lsh_params(
            pairs, num_perm=num_perm, miss_cost=2.0, false_alarm_cost=1.0
        )
        loss, b, r = brute_force_best(pairs, num_perm, 2.0, 1.0)
        self.assertEqual(result.params, (b, r))
        self.assertAlmostEqual(result.train.expected_loss, loss)
        self.assertEqual(result.train.b, b)
        self.assertEqual(result.train.r, r)

    def test_enumerates_all_legal_combinations(self):
        num_perm = 8
        rng = np.random.RandomState(7)
        pairs = make_dataset(rng, num_perm, 5, 5)
        result = calibrate_lsh_params(pairs, num_perm=num_perm)
        expected_count = sum(num_perm // b for b in range(1, num_perm + 1))
        self.assertEqual(len(result.train_scores), expected_count)
        for score in result.train_scores:
            self.assertGreaterEqual(score.b, 1)
            self.assertGreaterEqual(score.r, 1)
            self.assertLessEqual(score.b * score.r, num_perm)
            self.assertEqual(score.used_permutations, score.b * score.r)
            self.assertEqual(score.unused_permutations, num_perm - score.b * score.r)
        losses = [score.expected_loss for score in result.train_scores]
        self.assertEqual(losses, sorted(losses))

    def test_unused_permutations_reported(self):
        # num_perm = 10: the (3, 3) combination leaves 1 permutation unused.
        query = np.array([1, 1, 1, 1, 5, 6, 7, 8, 9, 10], dtype=np.uint64)
        candidate = np.array([1, 1, 1, 1, 5, 6, 7, 8, 9, 11], dtype=np.uint64)
        pairs = [(query, candidate, True, 1.0)]
        result = calibrate_lsh_params(pairs, num_perm=10)
        b, r = result.params
        self.assertEqual(result.unused_permutations, 10 - b * r)
        self.assertEqual(result.train.unused_permutations, result.unused_permutations)

    def test_validation_split_reported_separately(self):
        num_perm = 16
        rng = np.random.RandomState(123)
        train_pairs = make_dataset(rng, num_perm, 40, 40)
        validation_pairs = make_dataset(rng, num_perm, 40, 40)
        result = calibrate_lsh_params(
            train_pairs,
            num_perm=num_perm,
            miss_cost=3.0,
            false_alarm_cost=1.0,
            validation_pairs=validation_pairs,
        )
        self.assertIsNotNone(result.validation)
        self.assertIsNotNone(result.validation_best)
        self.assertIsNotNone(result.validation_scores)
        self.assertEqual(
            (result.validation.b, result.validation.r), result.params
        )
        self.assertEqual(result.validation.num_pairs, len(validation_pairs))
        # The training-chosen parameters cannot beat the validation oracle.
        self.assertLessEqual(
            result.validation_best.expected_loss,
            result.validation.expected_loss + 1e-12,
        )
        self.assertEqual(len(result.validation_scores), len(result.train_scores))
        # Validation loss must match an independent computation on the
        # validation split, not the training numbers.
        loss, _, _ = brute_force_best(
            [tuple(p) for p in validation_pairs], num_perm, 3.0, 1.0
        )
        self.assertAlmostEqual(result.validation_best.expected_loss, loss)

    def test_identical_signatures_always_recalled(self):
        # Similarity boundary 1: identical signatures collide in every band.
        signature = np.arange(16, dtype=np.uint64)
        pairs = [(signature, signature, True, 1.0)]
        result = calibrate_lsh_params(pairs, num_perm=16)
        self.assertEqual(result.train.expected_loss, 0.0)
        self.assertEqual(result.train.missed_weight, 0.0)
        for score in result.train_scores:
            self.assertEqual(score.expected_loss, 0.0)

    def test_disjoint_signatures_never_recalled(self):
        # Similarity boundary 0: signatures differing at every position never
        # collide, whatever (b, r) is.
        query = np.arange(16, dtype=np.uint64)
        candidate = np.arange(100, 116, dtype=np.uint64)
        pairs = [(query, candidate, False, 1.0)]
        result = calibrate_lsh_params(pairs, num_perm=16)
        self.assertEqual(result.train.expected_loss, 0.0)
        for score in result.train_scores:
            self.assertEqual(score.false_alarm_weight, 0.0)
        # The same pair labeled should-recall is always missed.
        result = calibrate_lsh_params(
            [(query, candidate, True, 2.0)], num_perm=16, miss_cost=5.0
        )
        self.assertEqual(result.train.expected_loss, 5.0)
        self.assertEqual(result.train.missed_weight, 2.0)

    def test_empty_minhash_samples(self):
        # Two empty MinHashes have identical (all-max) hash values and are
        # therefore always recalled; an empty and a non-empty one are not.
        empty_a = MinHash(num_perm=16)
        empty_b = MinHash(num_perm=16)
        non_empty = MinHash(num_perm=16)
        non_empty.update(b"token")
        result = calibrate_lsh_params(
            [(empty_a, empty_b, True, 1.0), (empty_a, non_empty, False, 1.0)],
            num_perm=16,
        )
        self.assertEqual(result.train.expected_loss, 0.0)
        self.assertEqual(result.train.missed_weight, 0.0)
        self.assertEqual(result.train.false_alarm_weight, 0.0)

    def test_all_positive_labels(self):
        rng = np.random.RandomState(3)
        num_perm = 8
        pairs = [
            (*similar_pair(rng, num_perm, 0.8), True, 1.0) for _ in range(10)
        ]
        result = calibrate_lsh_params(pairs, num_perm=num_perm)
        self.assertEqual(result.train.false_alarm_loss, 0.0)
        self.assertEqual(result.train.false_alarm_weight, 0.0)
        self.assertTrue(np.isfinite(result.train.expected_loss))

    def test_all_negative_labels(self):
        rng = np.random.RandomState(4)
        num_perm = 8
        pairs = [
            (*similar_pair(rng, num_perm, 0.1), False, 1.0) for _ in range(10)
        ]
        result = calibrate_lsh_params(pairs, num_perm=num_perm)
        self.assertEqual(result.train.miss_loss, 0.0)
        self.assertEqual(result.train.missed_weight, 0.0)
        self.assertTrue(np.isfinite(result.train.expected_loss))

    def test_weights_and_costs_determine_loss(self):
        # One positive pair recalled only when r == 1, one negative pair
        # never recalled. With num_perm = 2 the legal combos are (1, 1),
        # (1, 2) and (2, 1).
        query = np.array([7, 1], dtype=np.uint64)
        candidate = np.array([7, 2], dtype=np.uint64)
        far = np.array([100, 200], dtype=np.uint64)
        pairs = [
            (query, candidate, True, 2.0),
            (query, far, False, 1.0),
        ]
        result = calibrate_lsh_params(
            pairs, num_perm=2, miss_cost=10.0, false_alarm_cost=1.0
        )
        scores = {(s.b, s.r): s for s in result.train_scores}
        # r = 2 misses the positive pair: loss = 10 * 2 / 3.
        self.assertAlmostEqual(scores[(1, 2)].expected_loss, 10.0 * 2.0 / 3.0)
        self.assertAlmostEqual(scores[(1, 2)].missed_weight, 2.0)
        # r = 1 recalls everything relevant and nothing else.
        self.assertAlmostEqual(scores[(1, 1)].expected_loss, 0.0)
        self.assertAlmostEqual(scores[(2, 1)].expected_loss, 0.0)
        # Tie broken toward the smallest b.
        self.assertEqual(result.params, (1, 1))
        # Zero miss cost makes the missed pair free.
        result = calibrate_lsh_params(pairs, num_perm=2, miss_cost=0.0)
        self.assertAlmostEqual(result.train.expected_loss, 0.0)

    def test_accepts_minhash_objects_and_dataclass_pairs(self):
        rng = np.random.RandomState(9)
        num_perm = 8
        query, candidate = similar_pair(rng, num_perm, 0.8)
        pairs = [
            LSHCalibrationPair(
                query=make_minhash(query),
                candidate=make_minhash(candidate),
                should_recall=True,
                weight=1.5,
            )
        ]
        result = calibrate_lsh_params(pairs, num_perm=num_perm)
        self.assertEqual(result.train.total_weight, 1.5)
        self.assertEqual(result.train.num_pairs, 1)

    def test_recommended_params_match_real_index_behavior(self):
        num_perm = 16
        rng = np.random.RandomState(2024)
        pairs = make_dataset(rng, num_perm, 20, 20)
        result = calibrate_lsh_params(pairs, num_perm=num_perm)
        b, r = result.params
        lsh = MinHashLSH(threshold=0.5, num_perm=num_perm, params=(b, r))
        for i, (query, candidate, _label, _weight) in enumerate(pairs):
            lsh.insert("key-%d" % i, make_minhash(candidate))
        for query, candidate, _label, _weight in pairs:
            recalled_by_index = len(lsh.query(make_minhash(query))) > 0
            recalled_by_model = any(
                np.array_equal(
                    query[band * r:(band + 1) * r],
                    candidate[band * r:(band + 1) * r],
                )
                for band in range(b)
            )
            # The index contains many candidates, so it can only recall more;
            # for the disjoint/identical synthetic pairs here the band model
            # must agree with a single-candidate index.
            single = MinHashLSH(threshold=0.5, num_perm=num_perm, params=(b, r))
            single.insert("only", make_minhash(candidate))
            self.assertEqual(
                len(single.query(make_minhash(query))) > 0, recalled_by_model
            )
            self.assertTrue(recalled_by_index or not recalled_by_model)

    def test_invalid_inputs(self):
        pair = (np.arange(4, dtype=np.uint64), np.arange(4, dtype=np.uint64), True, 1.0)
        with self.assertRaises(ValueError):
            calibrate_lsh_params([pair], num_perm=0)
        with self.assertRaises(ValueError):
            calibrate_lsh_params([pair], num_perm=4, miss_cost=-1.0)
        with self.assertRaises(ValueError):
            calibrate_lsh_params([pair], num_perm=4, false_alarm_cost=-1.0)
        with self.assertRaises(ValueError):
            calibrate_lsh_params([], num_perm=4)
        with self.assertRaises(ValueError):
            # Signature length does not match num_perm.
            calibrate_lsh_params([pair], num_perm=8)
        with self.assertRaises(ValueError):
            # Zero total weight.
            calibrate_lsh_params(
                [(pair[0], pair[1], True, 0.0)], num_perm=4
            )
        with self.assertRaises(ValueError):
            # 2-D hashvalues (e.g. WeightedMinHash) are rejected.
            calibrate_lsh_params(
                [(np.ones((4, 2)), np.ones((4, 2)), True, 1.0)], num_perm=4
            )
        with self.assertRaises(ValueError):
            calibrate_lsh_params(
                [pair], num_perm=4, validation_pairs=[]
            )


if __name__ == "__main__":
    unittest.main()
