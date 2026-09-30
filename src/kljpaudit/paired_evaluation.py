"""Fixed-label-space, ID-aligned paired evaluation."""

from numbers import Integral
import numpy as np

TARGETS = (
    "fine_lv", "imprisonment_with_labor_lv",
    "imprisonment_without_labor_lv",
)
CLASS_COUNTS = (5, 6, 5)


def _labels(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("정답·예측은 대상 순서의 정수 3개여야 합니다")
    for value, count in zip(values, CLASS_COUNTS):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value < count
        ):
            raise ValueError("정답·예측의 형식 또는 label space 오류")
    return tuple(int(value) for value in values)


def _records(records):
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("예측 기록은 객체 형식이어야 합니다")
        uid = record.get("case_uid")
        if (
            not isinstance(uid, str)
            or not uid.strip()
            or uid != uid.strip()
        ):
            raise ValueError("사례 ID 형식 오류")
        if uid in result:
            raise ValueError("사례 ID 중복")
        result[uid] = (
            _labels(record.get("y_true")),
            _labels(record.get("y_pred")),
        )
    if not result:
        raise ValueError("예측 기록이 비어 있습니다")
    return result


def _metrics(y_true, y_pred):
    result = {}
    for index, (target, count) in enumerate(zip(TARGETS, CLASS_COUNTS)):
        matrix = np.bincount(
            count * y_true[:, index] + y_pred[:, index],
            minlength=count * count,
        ).reshape(count, count)
        denominator = matrix.sum(0) + matrix.sum(1)
        class_f1 = np.divide(
            2.0 * matrix.diagonal(), denominator,
            out=np.zeros(count), where=denominator != 0,
        )
        result[f"{target}_macro_f1"] = float(class_f1.mean())
        result[f"{target}_accuracy"] = float(
            (y_true[:, index] == y_pred[:, index]).mean()
        )
    result["mean_macro_f1"] = float(np.mean([
        result[f"{target}_macro_f1"] for target in TARGETS
    ]))
    result["joint_em"] = float(np.all(y_true == y_pred, axis=1).mean())
    return result


def evaluate_paired(original, transformed, subset_ids=None,
                    iterations=1000, seed=42):
    """Use supplied IDs only; never select cases from model responses."""
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations < 1
    ):
        raise ValueError("bootstrap 반복 수는 양의 정수여야 합니다")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or seed < 0
    ):
        raise ValueError("seed는 음수가 아닌 정수여야 합니다")

    original_rows = _records(original)
    transformed_rows = _records(transformed)
    if set(original_rows) != set(transformed_rows):
        raise ValueError("원본·변환의 사례 집합이 다릅니다")

    # subset 밖에서도 같은 사례의 참고 정답이 다르면 실패합니다.
    for uid in original_rows:
        if original_rows[uid][0] != transformed_rows[uid][0]:
            raise ValueError("원본·변환의 참고 정답이 다릅니다")

    if subset_ids is None:
        ids = sorted(original_rows)
    else:
        ids = list(subset_ids)
        if not ids or any(
            not isinstance(uid, str)
            or not uid.strip()
            or uid != uid.strip()
            for uid in ids
        ):
            raise ValueError("subset ID 형식 오류 또는 빈 subset")
        if len(ids) != len(set(ids)):
            raise ValueError("subset ID 중복")
        if not set(ids).issubset(original_rows):
            raise ValueError("예측 파일에 없는 subset ID")
        ids.sort()

    y_true = np.array([original_rows[uid][0] for uid in ids], dtype=np.int64)
    p_original = np.array([original_rows[uid][1] for uid in ids], dtype=np.int64)
    p_changed = np.array([transformed_rows[uid][1] for uid in ids], dtype=np.int64)
    original_metrics = _metrics(y_true, p_original)
    changed_metrics = _metrics(y_true, p_changed)

    names = list(original_metrics)
    delta = {
        name: changed_metrics[name] - original_metrics[name] for name in names
    }
    flips = p_original != p_changed
    flip_names = [f"{target}_flip" for target in TARGETS] + ["joint_flip"]
    flip_point = np.r_[flips.mean(0), flips.any(1).mean()]

    rng = np.random.default_rng(seed)
    delta_samples = np.empty((iterations, len(names)))
    flip_samples = np.empty((iterations, 4))
    for iteration in range(iterations):
        indices = rng.integers(0, len(ids), size=len(ids))
        a = _metrics(y_true[indices], p_original[indices])
        b = _metrics(y_true[indices], p_changed[indices])
        delta_samples[iteration] = [b[name] - a[name] for name in names]
        selected = flips[indices]
        flip_samples[iteration] = np.r_[
            selected.mean(0), selected.any(1).mean()
        ]

    lower, upper = np.quantile(delta_samples, [0.025, 0.975], axis=0)
    flip_lower, flip_upper = np.quantile(
        flip_samples, [0.025, 0.975], axis=0
    )
    return {
        "n": len(ids),
        "original": original_metrics,
        "transformed": changed_metrics,
        "delta": delta,
        "delta_ci95": {
            name: [float(lower[index]), float(upper[index])]
            for index, name in enumerate(names)
        },
        "flip": dict(zip(flip_names, map(float, flip_point))),
        "flip_ci95": {
            name: [float(flip_lower[index]), float(flip_upper[index])]
            for index, name in enumerate(flip_names)
        },
        "bootstrap": {
            "iterations": int(iterations),
            "seed": int(seed),
            "rng": "numpy.default_rng",
            "case_order": "sorted_case_uid",
            "unit": "paired_case",
            "interval": "percentile_95",
        },
    }
