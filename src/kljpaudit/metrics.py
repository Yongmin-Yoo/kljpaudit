"""Consistent multitask metrics with complete label spaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def target_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
) -> dict[str, float]:
    """Calculate target metrics using the complete training label set."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
    }


def joint_exact_match(
    true_by_target: Mapping[str, Sequence[int]],
    pred_by_target: Mapping[str, Sequence[int]],
) -> float:
    """Proportion for which all target labels are predicted correctly."""
    targets = tuple(true_by_target)

    if not targets:
        raise ValueError("At least one target is required.")

    correctness = [
        np.asarray(true_by_target[target])
        == np.asarray(pred_by_target[target])
        for target in targets
    ]

    return float(np.logical_and.reduce(correctness).mean())


def multitask_metrics(
    true_by_target: Mapping[str, Sequence[int]],
    pred_by_target: Mapping[str, Sequence[int]],
    labels_by_target: Mapping[str, Sequence[int]],
) -> dict:
    """Calculate per-target and aggregate metrics."""
    per_target = {
        target: target_metrics(
            true_by_target[target],
            pred_by_target[target],
            labels_by_target[target],
        )
        for target in true_by_target
    }

    mean_macro_f1 = float(
        np.mean([
            values["macro_f1"]
            for values in per_target.values()
        ])
    )

    return {
        "per_target": per_target,
        "mean_macro_f1": mean_macro_f1,
        "joint_exact_match": joint_exact_match(
            true_by_target,
            pred_by_target,
        ),
    }


def prediction_flip_rate(
    original: Sequence[int],
    transformed: Sequence[int],
) -> float:
    return float(
        (
            np.asarray(original)
            != np.asarray(transformed)
        ).mean()
    )
