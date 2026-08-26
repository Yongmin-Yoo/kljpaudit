"""Paired case-level bootstrap utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def percentile_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def paired_bootstrap_delta(
    y_true: np.ndarray,
    original_pred: np.ndarray,
    transformed_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap transformed-minus-original metric differences."""
    y_true = np.asarray(y_true)
    original_pred = np.asarray(original_pred)
    transformed_pred = np.asarray(transformed_pred)

    if not (
        len(y_true)
        == len(original_pred)
        == len(transformed_pred)
    ):
        raise ValueError("Paired arrays must have equal length.")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(iterations, dtype=float)

    for iteration in range(iterations):
        indices = rng.integers(0, n, size=n)

        original_score = metric(
            y_true[indices],
            original_pred[indices],
        )

        transformed_score = metric(
            y_true[indices],
            transformed_pred[indices],
        )

        deltas[iteration] = transformed_score - original_score

    lower, upper = percentile_interval(deltas)

    return {
        "mean_delta": float(deltas.mean()),
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
        "samples": deltas,
    }
