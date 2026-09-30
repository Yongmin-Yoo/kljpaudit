"""규칙 탐지 문장 교체 실험의 사례 단위 평가."""

import numpy as np
from sklearn.metrics import f1_score

TARGETS = (
    "fine_lv",
    "imprisonment_with_labor_lv",
    "imprisonment_without_labor_lv",
)
K = dict(zip(TARGETS, (5, 6, 5)))
CONDITIONS = ("original", "removed", "same_label", "different_label")
COMPARISONS = (
    ("different_minus_same", "different_label", "same_label"),
    ("different_minus_original", "different_label", "original"),
    ("same_minus_original", "same_label", "original"),
    ("removed_minus_original", "removed", "original"),
)


def bootstrap_weights(n, repetitions=1000, seed=42):
    """사례를 resample하며 donor 반복을 독립 사례로 세지 않습니다."""
    if n < 1 or repetitions < 1:
        raise ValueError("사례 수와 bootstrap 반복 수는 양수여야 합니다")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(repetitions, n))
    weights = np.zeros((repetitions, n), dtype=np.int64)
    np.add.at(
        weights,
        (
            np.repeat(np.arange(repetitions), n),
            indices.ravel(),
        ),
        1,
    )
    return weights


def summarize(values, weights):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or weights.shape[1] != len(values):
        raise ValueError("사례별 값과 bootstrap 배열 크기가 다릅니다")
    if np.isinf(values).any():
        raise ValueError("무한대 지표 값이 있습니다")

    valid = np.isfinite(values)
    if not valid.any():
        return {
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_valid_recipients": 0,
            "bootstrap_valid_resamples": 0,
        }

    denominator = weights[:, valid].sum(axis=1)
    numerator = weights[:, valid] @ values[valid]
    boot = numerator[denominator > 0] / denominator[denominator > 0]
    low, high = np.quantile(boot, [0.025, 0.975])

    return {
        "estimate": float(values[valid].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid_recipients": int(valid.sum()),
        "bootstrap_valid_resamples": len(boot),
    }


def fixed_macro_f1(truth, prediction, target):
    return float(f1_score(
        truth, prediction,
        labels=list(range(K[target])),
        average="macro",
        zero_division=0,
    ))


def evaluate_target(
    truth, predictions, probabilities, different_labels,
    target, bootstrap_repetitions=1000, bootstrap_seed=42,
):
    """확정된 동일 사례·donor 배열을 평가합니다. 사례 선정은 하지 않습니다."""
    if target not in TARGETS:
        raise ValueError("알 수 없는 target입니다")

    truth = np.asarray(truth)
    different_labels = np.asarray(different_labels)
    if truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("정답 배열은 사례 × 3 target이어야 합니다")
    if different_labels.ndim != 2:
        raise ValueError("donor 라벨은 사례 × 반복 배열이어야 합니다")

    n, repeats = different_labels.shape
    if n < 1 or repeats < 1 or truth.shape[0] != n:
        raise ValueError("사례·반복 배열 크기가 잘못됐습니다")
    if not np.issubdtype(truth.dtype, np.integer):
        raise ValueError("정답 라벨은 정수여야 합니다")
    if not np.issubdtype(different_labels.dtype, np.integer):
        raise ValueError("donor 라벨은 정수여야 합니다")

    number = TARGETS.index(target)

    for j, name in enumerate(TARGETS):
        if ((truth[:, j] < 0) | (truth[:, j] >= K[name])).any():
            raise ValueError("정답이 전체 라벨 공간을 벗어납니다")

    if (
        (different_labels < 0).any()
        or (different_labels >= K[target]).any()
        or (different_labels == truth[:, number, None]).any()
    ):
        raise ValueError("다른 라벨 donor 조건을 위반했습니다")

    if set(predictions) != set(CONDITIONS):
        raise ValueError("예측 조건 구성이 다릅니다")
    if set(probabilities) != set(CONDITIONS):
        raise ValueError("확률 조건 구성이 다릅니다")

    checked_predictions = {}
    checked_probabilities = {}

    for condition in CONDITIONS:
        expected_repeats = 1 if condition in ("original", "removed") else repeats
        pred = np.asarray(predictions[condition])
        prob = np.asarray(probabilities[condition], dtype=float)

        if pred.shape != (n, expected_repeats, 3):
            raise ValueError("예측 배열 크기가 다릅니다")
        if not np.issubdtype(pred.dtype, np.integer):
            raise ValueError("예측 라벨은 정수여야 합니다")
        if prob.shape != (n, expected_repeats, K[target]):
            raise ValueError("확률 배열 크기가 다릅니다")
        if (
            not np.isfinite(prob).all()
            or (prob < 0).any()
            or not np.allclose(prob.sum(axis=-1), 1, atol=1e-5)
        ):
            raise ValueError("확률 벡터가 잘못됐습니다")

        for j, name in enumerate(TARGETS):
            if ((pred[:, :, j] < 0) | (pred[:, :, j] >= K[name])).any():
                raise ValueError("예측이 전체 라벨 공간을 벗어납니다")
        if not np.array_equal(prob.argmax(-1), pred[:, :, number]):
            raise ValueError("확률 argmax와 예측이 다릅니다")

        checked_predictions[condition] = pred
        checked_probabilities[condition] = prob

    weights = bootstrap_weights(n, bootstrap_repetitions, bootstrap_seed)

    original = np.repeat(checked_predictions["original"], repeats, axis=1)
    original_prob = np.repeat(
        checked_probabilities["original"], repeats, axis=1
    )
    original_donor_prob = np.take_along_axis(
        original_prob, different_labels[:, :, None], axis=2
    ).squeeze(2)

    eligible = original[:, :, number] != different_labels
    eligible_count = eligible.sum(axis=1)

    condition_rows, contrast_rows, performance_rows = [], [], []
    case_values = {}
    contrast_vectors = {}

    for condition in CONDITIONS:
        pred = checked_predictions[condition]
        prob = checked_probabilities[condition]

        # 교체 조건 성능은 반복별 계산 후 평균합니다.
        repeat_scores = []
        for repetition in range(pred.shape[1]):
            current = pred[:, repetition, :]
            scores = {}
            for j, name in enumerate(TARGETS):
                scores[f"{name}_macro_f1"] = fixed_macro_f1(
                    truth[:, j], current[:, j], name
                )
                scores[f"{name}_accuracy"] = float(
                    (truth[:, j] == current[:, j]).mean()
                )
            scores["mean_macro_f1"] = float(np.mean([
                scores[f"{name}_macro_f1"] for name in TARGETS
            ]))
            scores["joint_em"] = float(
                (current == truth).all(axis=1).mean()
            )
            repeat_scores.append(scores)

        for metric in repeat_scores[0]:
            performance_rows.append({
                "condition": condition,
                "metric": metric,
                "estimate": float(np.mean([
                    record[metric] for record in repeat_scores
                ])),
                "n_recipients": n,
            })

        if pred.shape[1] == 1:
            pred = np.repeat(pred, repeats, axis=1)
            prob = np.repeat(prob, repeats, axis=1)

        donor_prob = np.take_along_axis(
            prob, different_labels[:, :, None], axis=2
        ).squeeze(2)
        donor_hit = pred[:, :, number] == different_labels

        new_conversion = np.full(n, np.nan)
        np.divide(
            (donor_hit & eligible).sum(axis=1),
            eligible_count,
            out=new_conversion,
            where=eligible_count > 0,
        )

        values = {
            "target_flip": (
                pred[:, :, number] != original[:, :, number]
            ).mean(axis=1),
            "joint_flip": (pred != original).any(axis=2).mean(axis=1),
            "different_donor_label_probability": donor_prob.mean(axis=1),
            "different_donor_probability_change": (
                donor_prob - original_donor_prob
            ).mean(axis=1),
            "different_donor_label_hit": donor_hit.mean(axis=1),
            "new_different_donor_conversion": new_conversion,
            "target_accuracy": (
                pred[:, :, number] == truth[:, number, None]
            ).mean(axis=1),
            "joint_em": (pred == truth[:, None, :]).all(axis=2).mean(axis=1),
        }

        for j, name in enumerate(TARGETS):
            values[f"{name}_flip"] = (
                pred[:, :, j] != original[:, :, j]
            ).mean(axis=1)

        case_values[condition] = values

        for metric, vector in values.items():
            condition_rows.append({
                "condition": condition, "metric": metric,
                "n_recipients": n, **summarize(vector, weights),
            })

    for contrast, left, right in COMPARISONS:
        for metric in case_values[left]:
            vector = case_values[left][metric] - case_values[right][metric]
            contrast_vectors[(contrast, metric)] = vector
            stats = summarize(vector, weights)
            if stats["n_valid_recipients"] == 0:
                continue
            contrast_rows.append({
                "contrast": contrast, "metric": metric,
                "n_recipients": n, **stats,
            })

    return {
        "conditions": condition_rows,
        "contrasts": contrast_rows,
        "performance": performance_rows,
        "contrast_vectors": contrast_vectors,
    }
