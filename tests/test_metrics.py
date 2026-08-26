from kljpaudit.metrics import (
    joint_exact_match,
    prediction_flip_rate,
    target_metrics,
)


def test_target_metrics_use_full_label_space():
    result = target_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 0],
        labels=[0, 1, 2],
    )

    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["accuracy"] == 0.75


def test_joint_exact_match():
    true = {
        "a": [0, 1, 1],
        "b": [0, 0, 1],
    }

    pred = {
        "a": [0, 1, 0],
        "b": [0, 1, 1],
    }

    assert joint_exact_match(true, pred) == 1 / 3


def test_flip_rate():
    assert prediction_flip_rate(
        [0, 1, 2, 3],
        [0, 2, 2, 4],
    ) == 0.5
