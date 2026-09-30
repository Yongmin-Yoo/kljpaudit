"""Independent agreement tests using synthetic records only."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from kljpaudit.annotation_agreement import summarize_independent_agreement

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "examples" / "synthetic_independent_annotations.json"
)


class AgreementTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(FIXTURE.read_text("utf-8"))
        self.assertTrue(fixture["synthetic"])
        self.r1 = fixture["r1"]
        self.r2 = fixture["r2"]

    def test_aggregate_counts_and_missing_denominator(self):
        result = summarize_independent_agreement(self.r1, self.r2)
        self.assertEqual(result["reviewed_cases"], 4)
        self.assertEqual(result["overall"]["n_possible"], 36)
        self.assertEqual(result["overall"]["n_both_annotated"], 35)
        self.assertEqual(result["overall"]["n_agree"], 33)
        self.assertEqual(result["overall"]["n_disagree"], 2)
        self.assertEqual(result["cases_with_observed_disagreement"], 1)
        self.assertEqual(result["cases_with_missing_cells"], 1)
        self.assertAlmostEqual(
            result["overall"]["percent_agreement"], 100 * 33 / 35
        )

    def test_case_order_does_not_matter(self):
        self.assertEqual(
            summarize_independent_agreement(self.r1, self.r2),
            summarize_independent_agreement(self.r1, list(reversed(self.r2))),
        )

    def test_duplicates_fail(self):
        with self.assertRaises(ValueError):
            summarize_independent_agreement(
                self.r1 + [deepcopy(self.r1[0])], self.r2
            )

    def test_different_case_sets_fail(self):
        with self.assertRaises(ValueError):
            summarize_independent_agreement(self.r1, self.r2[:-1])

    def test_same_reviewer_fails(self):
        r2 = deepcopy(self.r2)
        for record in r2:
            record["reviewer_id"] = self.r1[0]["reviewer_id"]
        with self.assertRaises(ValueError):
            summarize_independent_agreement(self.r1, r2)

    def test_missing_column_fails(self):
        r2 = deepcopy(self.r2)
        del r2[0]["prospective_availability"]
        with self.assertRaises(ValueError):
            summarize_independent_agreement(self.r1, r2)

    def test_invalid_category_fails(self):
        r2 = deepcopy(self.r2)
        r2[0]["outcome_relation"] = "unknown"
        with self.assertRaises(ValueError):
            summarize_independent_agreement(self.r1, r2)

    def test_does_not_emit_identifiers_or_modify_inputs(self):
        before = deepcopy((self.r1, self.r2))
        result = summarize_independent_agreement(self.r1, self.r2)
        self.assertNotIn("synthetic-", json.dumps(result))
        self.assertEqual((self.r1, self.r2), before)

    def test_adjudicated_input_fails(self):
        r1 = deepcopy(self.r1)
        r1[0]["record_type"] = "adjudicated"
        with self.assertRaises(ValueError):
            summarize_independent_agreement(r1, self.r2)

    def test_unclear_is_not_missing(self):
        r1, r2 = deepcopy(self.r1), deepcopy(self.r2)
        r1[0]["outcome_relation"] = "unclear"
        r2[0]["outcome_relation"] = "unclear"
        result = summarize_independent_agreement(r1, r2)
        field = result["fields"]["outcome_relation"]
        self.assertEqual(field["n_both_annotated"], 4)
        self.assertEqual(field["n_agree"], 4)


if __name__ == "__main__":
    unittest.main()
