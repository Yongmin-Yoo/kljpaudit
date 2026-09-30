import unittest
import numpy as np
from kljpaudit.cue_swap import (
    TARGETS, K, CONDITIONS,
    bootstrap_weights, summarize, fixed_macro_f1, evaluate_target,
)


def fixture():
    truth = np.zeros((3, 3), dtype=int)
    predictions = {
        condition: np.zeros((3, 1 if condition in (
            "original", "removed"
        ) else 2, 3), dtype=int)
        for condition in CONDITIONS
    }
    probabilities = {
        condition: np.eye(5)[predictions[condition][:, :, 0]]
        for condition in CONDITIONS
    }
    labels = np.ones((3, 2), dtype=int)
    return truth, predictions, probabilities, labels


class CueSwapTests(unittest.TestCase):
    def test_weights_count_cases(self):
        weights = bootstrap_weights(3, 20, 42)
        self.assertTrue((weights.sum(axis=1) == 3).all())

    def test_weights_reproducible(self):
        np.testing.assert_array_equal(
            bootstrap_weights(3, 20, 42),
            bootstrap_weights(3, 20, 42),
        )

    def test_fixed_label_space(self):
        self.assertAlmostEqual(
            fixed_macro_f1(np.array([0, 1]), np.array([0, 1]), TARGETS[0]),
            2 / 5,
        )

    def test_conditional_empty_is_not_zero(self):
        result = summarize(np.array([np.nan, np.nan]), bootstrap_weights(2, 20))
        self.assertTrue(np.isnan(result["estimate"]))
        self.assertEqual(result["n_valid_recipients"], 0)

    def test_conditional_valid_count(self):
        result = summarize(np.array([1.0, np.nan]), bootstrap_weights(2, 20))
        self.assertEqual(result["n_valid_recipients"], 1)
        self.assertEqual(result["estimate"], 1.0)

    def test_identical_conditions_zero_contrast(self):
        result = evaluate_target(*fixture(), TARGETS[0], bootstrap_repetitions=20)
        self.assertTrue(all(
            abs(row["estimate"]) < 1e-12 for row in result["contrasts"]
        ))

    def test_directional_change(self):
        truth, predictions, probabilities, labels = fixture()
        predictions["different_label"][:, :, 0] = 1
        probabilities["different_label"] = np.eye(5)[
            predictions["different_label"][:, :, 0]
        ]
        result = evaluate_target(
            truth, predictions, probabilities, labels,
            TARGETS[0], bootstrap_repetitions=20,
        )
        row = next(
            row for row in result["contrasts"]
            if row["contrast"] == "different_minus_same"
            and row["metric"] == "target_flip"
        )
        self.assertEqual(row["estimate"], 1.0)

    def test_invalid_probability_fails(self):
        truth, predictions, probabilities, labels = fixture()
        probabilities["same_label"][:] = 0
        with self.assertRaises(ValueError):
            evaluate_target(truth, predictions, probabilities, labels, TARGETS[0])

    def test_same_label_as_different_fails(self):
        truth, predictions, probabilities, labels = fixture()
        labels[:] = 0
        with self.assertRaises(ValueError):
            evaluate_target(truth, predictions, probabilities, labels, TARGETS[0])


if __name__ == "__main__":
    unittest.main()
