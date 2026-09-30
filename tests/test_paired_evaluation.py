"""Paired evaluation tests use synthetic predictions only."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from kljpaudit.paired_evaluation import evaluate_paired

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "examples" / "synthetic_paired_predictions.json"
)


class PairedEvaluationTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(FIXTURE.read_text("utf-8"))
        self.assertTrue(fixture["synthetic"])
        self.original = fixture["original"]
        self.changed = fixture["transformed"]

    def evaluate(self, original=None, changed=None, **kwargs):
        return evaluate_paired(
            self.original if original is None else original,
            self.changed if changed is None else changed,
            iterations=30, seed=42, **kwargs,
        )

    def test_point_estimates(self):
        result = self.evaluate()
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["original"]["joint_em"], 1.0)
        self.assertAlmostEqual(result["transformed"]["joint_em"], 1 / 3)
        self.assertAlmostEqual(result["delta"]["joint_em"], -2 / 3)
        self.assertAlmostEqual(result["flip"]["joint_flip"], 2 / 3)
        self.assertAlmostEqual(result["flip"]["fine_lv_flip"], 1 / 3)
        self.assertEqual(result["flip"]["imprisonment_without_labor_lv_flip"], 0)

    def test_case_order_invariance(self):
        self.assertEqual(
            self.evaluate(),
            self.evaluate(changed=list(reversed(self.changed))),
        )

    def test_fixed_full_label_space(self):
        records = [self.original[0]]
        result = self.evaluate(original=records, changed=records)
        self.assertAlmostEqual(result["original"]["fine_lv_macro_f1"], 1 / 5)
        self.assertAlmostEqual(
            result["original"]["imprisonment_with_labor_lv_macro_f1"], 1 / 6
        )

    def test_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            self.evaluate(original=self.original + [deepcopy(self.original[0])])

    def test_case_set_mismatch_fails(self):
        with self.assertRaises(ValueError):
            self.evaluate(changed=self.changed[:-1])

    def test_reference_mismatch_fails(self):
        changed = deepcopy(self.changed)
        changed[0]["y_true"] = [1, 0, 0]
        with self.assertRaises(ValueError):
            self.evaluate(changed=changed)

    def test_invalid_labels_fail(self):
        for value in [5, -1, True, 1.5, "1"]:
            changed = deepcopy(self.changed)
            changed[0]["y_pred"][0] = value
            with self.assertRaises(ValueError):
                self.evaluate(changed=changed)

    def test_subset_is_explicit(self):
        result = self.evaluate(subset_ids=["synthetic-002"])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["flip"]["joint_flip"], 0)
        self.assertEqual(result["delta_ci95"]["joint_em"], [0.0, 0.0])

    def test_invalid_subsets_fail(self):
        for subset in [[], ["missing"], ["synthetic-001", "synthetic-001"]]:
            with self.assertRaises(ValueError):
                self.evaluate(subset_ids=subset)

    def test_reproducible_aggregate_only(self):
        result = self.evaluate()
        self.assertEqual(result, self.evaluate())
        self.assertNotIn("synthetic-", json.dumps(result))

    def test_identity_has_zero_deltas(self):
        result = self.evaluate(changed=self.original)
        self.assertTrue(all(value == 0 for value in result["delta"].values()))
        self.assertTrue(all(
            interval == [0.0, 0.0] for interval in result["delta_ci95"].values()
        ))


if __name__ == "__main__":
    unittest.main()
