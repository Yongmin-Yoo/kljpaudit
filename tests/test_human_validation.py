"""Tests use synthetic data only."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from kljpaudit.human_validation import (
    select_validated_case_ids,
    summarize_adjudicated,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "examples" / "synthetic_human_validation.json"
)


class HumanValidationTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(FIXTURE.read_text("utf-8"))
        self.assertTrue(fixture["synthetic"])
        self.records = fixture["adjudicated"]

    def test_three_criteria_are_conjunctive(self):
        self.assertEqual(
            select_validated_case_ids(self.records), ("synthetic-001",)
        )

    def test_duplicate_ids_fail(self):
        records = deepcopy(self.records)
        records.append(deepcopy(records[0]))
        with self.assertRaises(ValueError):
            select_validated_case_ids(records)

    def test_missing_criterion_fails(self):
        records = deepcopy(self.records)
        del records[0]["prospective_availability"]
        with self.assertRaises(ValueError):
            select_validated_case_ids(records)

    def test_unknown_category_fails(self):
        records = deepcopy(self.records)
        records[0]["outcome_relation"] = "unknown"
        with self.assertRaises(ValueError):
            select_validated_case_ids(records)

    def test_independent_records_are_not_adjudication(self):
        records = deepcopy(self.records)
        records[0]["record_type"] = "independent"
        with self.assertRaises(ValueError):
            select_validated_case_ids(records)

    def test_empty_records_fail(self):
        with self.assertRaises(ValueError):
            select_validated_case_ids([])

    def test_blank_or_padded_ids_fail(self):
        for case_uid in ["", "   ", " synthetic-001 "]:
            records = deepcopy(self.records)
            records[0]["case_uid"] = case_uid
            with self.assertRaises(ValueError):
                select_validated_case_ids(records)

    def test_predictions_do_not_affect_selection(self):
        records = deepcopy(self.records)
        for index, record in enumerate(records):
            record["joint_flip"] = bool(index % 2)
            record["model_correct"] = bool((index + 1) % 2)
        self.assertEqual(
            select_validated_case_ids(records),
            select_validated_case_ids(self.records),
        )

    def test_aggregate_does_not_emit_case_ids(self):
        result = summarize_adjudicated(self.records)
        self.assertEqual(result["reviewed_cases"], 4)
        self.assertEqual(result["validated_cases"], 1)
        self.assertNotIn("synthetic-", json.dumps(result))

    def test_each_criterion_can_exclude_a_case(self):
        alternatives = {
            "outcome_relation": "outcome_independent",
            "prospective_availability": "available",
            "facts_preserved_after_removal": "no",
        }
        for field, value in alternatives.items():
            with self.subTest(field=field):
                record = deepcopy(self.records[0])
                record[field] = value
                self.assertEqual(select_validated_case_ids([record]), ())


if __name__ == "__main__":
    unittest.main()
