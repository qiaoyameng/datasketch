import unittest

import numpy as np

from datasketch import MinHash, MinHashLSH
from datasketch.lsh_calibration import (
    CalibrationSample,
    calibrate_lsh_params,
)


def minhash_from_mask_values(values):
    """Build a MinHash with exactly the given hash values."""
    return MinHash(hashvalues=np.asarray(values, dtype=np.uint64), scheme="legacy")


def pair_from_masks(mask_a, mask_b):
    """A pair whose per-permutation equality mask is mask_a == mask_b."""
    return minhash_from_mask_values(mask_a), minhash_from_mask_values(mask_b)


def brute_force_best(num_perm, samples, fp_cost, fn_cost):
    """Independent exhaustive evaluation of every legal (b, r) combination."""
    records = {}
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            total_weight = sum(s[3] for s in samples)
            fp = fn = 0.0
            for ma, mb, label, weight in samples:
                va = np.asarray(ma.hashvalues[: b * r]).reshape(b, r)
                vb = np.asarray(mb.hashvalues[: b * r]).reshape(b, r)
                recalled = bool((va == vb).all(axis=1).any())
                if recalled and not label:
                    fp += weight
                elif not recalled and label:
                    fn += weight
            loss = (fp_cost * fp + fn_cost * fn) / total_weight if total_weight else 0.0
            records[(b, r)] = (loss, fp, fn)
    best = min(records, key=lambda br: (records[br][0], br[0] * br[1], br[0], br[1]))
    return best, records


def make_samples(num_perm, specs, seed_offset=0):
    """Turn (mask, label, weight) specs into samples with concrete values."""
    samples = []
    for i, (mask, label, weight) in enumerate(specs):
        base = 1000 * (i + 1 + seed_offset)
        values_a = [base + j for j in range(num_perm)]
        values_b = [v if m else v + 500 for j, (v, m) in enumerate(zip(values_a, mask))]
        ma, mb = pair_from_masks(values_a, values_b)
        samples.append((ma, mb, label, weight))
    return samples


class TestCalibrateLshParams(unittest.TestCase):
    def test_exhaustive_small_budget_matches_brute_force(self):
        num_perm = 6
        rng = np.random.RandomState(42)
        specs = [
            (list(rng.randint(0, 2, num_perm)), bool(label), float(weight))
            for label, weight in [(True, 1.0), (True, 2.0), (False, 1.0), (False, 3.0)]
        ]
        train = make_samples(num_perm, specs)
        fp_cost, fn_cost = 2.0, 5.0
        result = calibrate_lsh_params(
            num_perm=num_perm,
            train_samples=train,
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
        )
        expected_best, records = brute_force_best(num_perm, train, fp_cost, fn_cost)
        # Every legal combination is evaluated exactly once.
        self.assertEqual({(e.b, e.r) for e in result.evaluations}, set(records))
        for e in result.evaluations:
            loss, fp, fn = records[(e.b, e.r)]
            self.assertAlmostEqual(e.train_loss, loss)
            self.assertAlmostEqual(e.train_false_positives, fp)
            self.assertAlmostEqual(e.train_false_negatives, fn)
            self.assertEqual(e.used_permutations, e.b * e.r)
            self.assertEqual(e.unused_permutations, num_perm - e.b * e.r)
        self.assertEqual(result.params, expected_best)
        self.assertEqual(result.unused_permutations, num_perm - expected_best[0] * expected_best[1])

    def test_selection_uses_train_only_and_reports_validation(self):
        num_perm = 4
        # Train: best combination is (1, 1).
        train = make_samples(
            num_perm,
            [
                ([1, 0, 0, 0], True, 1.0),
                ([0, 1, 1, 1], False, 1.0),
            ],
        )
        # Validation: (1, 1) is the worst choice there, (1, 2) is better.
        validation = make_samples(
            num_perm,
            [
                ([0, 0, 0, 0], True, 1.0),
                ([1, 0, 0, 0], False, 1.0),
            ],
            seed_offset=100,
        )
        result = calibrate_lsh_params(
            num_perm=num_perm,
            train_samples=train,
            validation_samples=validation,
        )
        # Selection follows the training split even though it is worse on
        # the validation split.
        self.assertEqual(result.params, (1, 1))
        self.assertAlmostEqual(result.best.train_loss, 0.0)
        self.assertAlmostEqual(result.best.validation_loss, 1.0)
        # The validation split is fully reported for every combination, so
        # the better validation choice is visible in the output.
        best_val = min(e.validation_loss for e in result.evaluations)
        self.assertAlmostEqual(best_val, 0.5)
        self.assertEqual(result.num_train_samples, 2)
        self.assertEqual(result.num_validation_samples, 2)
        as_dict = result.to_dict()
        self.assertEqual(as_dict["best"]["b"], 1)
        self.assertEqual(len(as_dict["evaluations"]), len(result.evaluations))

    def test_identical_and_disjoint_pairs_are_deterministic(self):
        num_perm = 8
        same = MinHash(num_perm=num_perm, seed=3)
        same.update_batch([b"a", b"b", b"c"])
        disjoint_a = MinHash(num_perm=num_perm, seed=3)
        disjoint_a.update_batch([b"x1", b"x2", b"x3"])
        disjoint_b = MinHash(num_perm=num_perm, seed=3)
        disjoint_b.update_batch([b"y1", b"y2", b"y3"])
        self.assertEqual(disjoint_a.jaccard(disjoint_b), 0.0)

        train = [(same, same, True, 1.0), (disjoint_a, disjoint_b, False, 1.0)]
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=train)
        # Jaccard 1.0 pairs are always recalled, Jaccard 0.0 pairs never:
        # every combination achieves zero loss.
        for e in result.evaluations:
            self.assertEqual(e.train_loss, 0.0)
            self.assertEqual(e.train_false_positives, 0.0)
            self.assertEqual(e.train_false_negatives, 0.0)
        # Ties resolve to the cheapest combination deterministically.
        self.assertEqual(result.params, (1, 1))
        self.assertEqual(result.unused_permutations, num_perm - 1)

    def test_empty_splits_have_deterministic_results(self):
        num_perm = 4
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=[])
        self.assertEqual(result.params, (1, 1))
        self.assertEqual(result.best.train_loss, 0.0)
        self.assertIsNone(result.best.validation_loss)
        self.assertEqual(result.num_train_samples, 0)
        self.assertEqual(result.num_validation_samples, 0)

        train = make_samples(num_perm, [([1, 1, 1, 1], True, 1.0)])
        result = calibrate_lsh_params(
            num_perm=num_perm, train_samples=train, validation_samples=[]
        )
        self.assertEqual(result.best.validation_loss, 0.0)
        self.assertEqual(result.best.validation_false_positives, 0.0)
        self.assertEqual(result.best.validation_false_negatives, 0.0)
        self.assertEqual(result.num_validation_samples, 0)

    def test_all_same_labels(self):
        num_perm = 4
        # Only positive labels: the false positive term is always zero.
        train = make_samples(
            num_perm,
            [([1, 1, 1, 1], True, 1.0), ([0, 0, 0, 0], True, 2.0)],
        )
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=train)
        for e in result.evaluations:
            self.assertEqual(e.train_false_positives, 0.0)
        # The all-zero mask is never recalled, so the weighted false
        # negative mass is 2 of 3 for any combination that recalls the
        # identical pair; the cheapest such combination wins.
        self.assertEqual(result.params, (1, 1))
        self.assertAlmostEqual(result.best.train_loss, 2.0 / 3.0)

        # Only negative labels: the false negative term is always zero.
        train = make_samples(
            num_perm,
            [([1, 1, 1, 1], False, 1.0), ([0, 0, 0, 0], False, 1.0)],
            seed_offset=7,
        )
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=train)
        for e in result.evaluations:
            self.assertEqual(e.train_false_negatives, 0.0)
        # The identical pair is recalled by every combination and the
        # disjoint pair by none, so every combination has the same loss
        # and the cheapest one wins.
        self.assertEqual(result.params, (1, 1))
        self.assertAlmostEqual(result.best.train_loss, 0.5)

    def test_weights_and_costs_change_selection(self):
        num_perm = 4
        # Positive pair only recalled with r = 1; negative pair recalled
        # whenever r = 1 as well.
        train = make_samples(
            num_perm,
            [
                ([1, 0, 0, 0], True, 1.0),
                ([1, 0, 0, 0], False, 1.0),
            ],
        )
        # Equal costs: r = 1 recalls both (loss 0.5 via FP), r > 1 recalls
        # neither (loss 0.5 via FN); tie goes to fewest used permutations.
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=train)
        self.assertEqual(result.params, (1, 1))
        # Missing the positive pair is very expensive: r = 1 wins outright.
        result = calibrate_lsh_params(
            num_perm=num_perm, train_samples=train, false_negative_cost=100.0
        )
        self.assertEqual(result.params, (1, 1))
        self.assertEqual(result.best.train_false_negatives, 0.0)
        # Recalling the negative pair is very expensive: avoid r = 1.
        result = calibrate_lsh_params(
            num_perm=num_perm, train_samples=train, false_positive_cost=100.0
        )
        self.assertEqual(result.params, (1, 2))
        self.assertEqual(result.best.train_false_positives, 0.0)
        self.assertEqual(result.best.unused_permutations, num_perm - 2)

        # Sample weights scale the loss as expected.
        weighted = [
            CalibrationSample(train[0][0], train[0][1], True, 3.0),
            CalibrationSample(train[1][0], train[1][1], False, 1.0),
        ]
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=weighted)
        eval_1_1 = next(e for e in result.evaluations if (e.b, e.r) == (1, 1))
        # (1, 1) recalls both: only the negative pair (weight 1 of 4) errs.
        self.assertAlmostEqual(eval_1_1.train_loss, 0.25)

    def test_matches_real_index_behavior(self):
        num_perm = 16
        rng = np.random.RandomState(7)
        specs = [
            (list(rng.randint(0, 2, num_perm)), bool(label), 1.0)
            for label in (True, True, False, False)
        ]
        train = make_samples(num_perm, specs)
        result = calibrate_lsh_params(num_perm=num_perm, train_samples=train)
        # MinHashLSH requires b >= 2; check every such combination against
        # the recall behavior of a real index.
        for e in result.evaluations:
            if e.b < 2:
                continue
            lsh = MinHashLSH(num_perm=num_perm, params=(e.b, e.r))
            for i, (ma, _mb, _label, _w) in enumerate(train):
                lsh.insert(f"a{i}", ma)
            for i, (ma, mb, _label, _w) in enumerate(train):
                recalled = f"a{i}" in lsh.query(mb)
                mask = np.asarray(ma.hashvalues) == np.asarray(mb.hashvalues)
                expected = bool(mask[: e.b * e.r].reshape(e.b, e.r).all(axis=1).any())
                self.assertEqual(recalled, expected)

    def test_invalid_inputs(self):
        ma, mb = pair_from_masks([1, 2], [1, 3])
        with self.assertRaises(ValueError):
            calibrate_lsh_params(num_perm=0, train_samples=[(ma, mb, True)])
        with self.assertRaises(ValueError):
            calibrate_lsh_params(
                num_perm=2, train_samples=[(ma, mb, True)], false_positive_cost=-1.0
            )
        with self.assertRaises(ValueError):
            calibrate_lsh_params(
                num_perm=2, train_samples=[(ma, mb, True)], false_negative_cost=-1.0
            )
        with self.assertRaises(ValueError):
            calibrate_lsh_params(num_perm=2, train_samples=[(ma, mb, True, 0.0)])
        with self.assertRaises(ValueError):
            calibrate_lsh_params(num_perm=2, train_samples=[(ma, mb, "yes")])
        with self.assertRaises(ValueError):
            calibrate_lsh_params(num_perm=2, train_samples=[(ma, mb)])
        with self.assertRaises(TypeError):
            calibrate_lsh_params(num_perm=2, train_samples=["not a sample"])
        with self.assertRaises(ValueError):
            # MinHash has fewer permutations than the budget.
            calibrate_lsh_params(num_perm=4, train_samples=[(ma, mb, True)])
        with self.assertRaises(TypeError):
            calibrate_lsh_params(num_perm=2, train_samples=[(object(), mb, True)])


if __name__ == "__main__":
    unittest.main()
