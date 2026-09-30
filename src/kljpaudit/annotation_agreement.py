"""Aggregate independent annotation agreement without adjudication."""

from collections.abc import Mapping
from .human_validation import FIELD_CATEGORIES


def _validate_independent(records):
    rows = {}
    reviewer = None
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"레코드 {position}: 객체 형식이 아닙니다")
        if record.get("record_type") != "independent":
            raise ValueError(f"레코드 {position}: 독립 평가 기록이 아닙니다")

        case_uid = record.get("case_uid")
        reviewer_id = record.get("reviewer_id")
        for value in (case_uid, reviewer_id):
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(f"레코드 {position}: 식별자 형식 오류")

        if case_uid in rows:
            raise ValueError(f"레코드 {position}: 사례 ID 중복")
        if reviewer is not None and reviewer != reviewer_id:
            raise ValueError("한 입력에 여러 평가자의 기록이 섞여 있습니다")
        reviewer = reviewer_id

        for field, categories in FIELD_CATEGORIES.items():
            if field not in record:
                raise ValueError(f"레코드 {position}: 필수 열 누락: {field}")
            value = record[field]
            if value is not None and (
                not isinstance(value, str) or value not in categories
            ):
                raise ValueError(f"레코드 {position}: 판정값 오류: {field}")

        rows[case_uid] = dict(record)

    if not rows:
        raise ValueError("독립 평가 기록이 비어 있습니다")
    return rows, reviewer


def summarize_independent_agreement(records_r1, records_r2):
    """Return raw agreement aggregates; missing cells are not unclear labels."""
    r1, reviewer_r1 = _validate_independent(records_r1)
    r2, reviewer_r2 = _validate_independent(records_r2)

    if reviewer_r1 == reviewer_r2:
        raise ValueError("서로 다른 두 평가자의 입력이 필요합니다")
    if set(r1) != set(r2):
        raise ValueError("두 평가자의 사례 집합이 다릅니다")

    ids = sorted(r1)
    fields = {}
    disagreements = set()
    missing_cases = set()
    overall_both = 0
    overall_agree = 0

    for field in FIELD_CATEGORIES:
        both = agree = missing_r1 = missing_r2 = 0
        for case_uid in ids:
            a, b = r1[case_uid][field], r2[case_uid][field]
            missing_r1 += a is None
            missing_r2 += b is None
            if a is None or b is None:
                missing_cases.add(case_uid)
                continue
            both += 1
            if a == b:
                agree += 1
            else:
                disagreements.add(case_uid)

        fields[field] = {
            "n_total": len(ids),
            "n_both_annotated": both,
            "n_agree": agree,
            "n_disagree": both - agree,
            "missing_r1": missing_r1,
            "missing_r2": missing_r2,
            "percent_agreement": 100.0 * agree / both if both else None,
        }
        overall_both += both
        overall_agree += agree

    return {
        "reviewed_cases": len(ids),
        "annotation_fields": len(FIELD_CATEGORIES),
        "cases_with_observed_disagreement": len(disagreements),
        "cases_with_missing_cells": len(missing_cases),
        "overall": {
            "n_possible": len(ids) * len(FIELD_CATEGORIES),
            "n_both_annotated": overall_both,
            "n_agree": overall_agree,
            "n_disagree": overall_both - overall_agree,
            "percent_agreement": (
                100.0 * overall_agree / overall_both if overall_both else None
            ),
        },
        "fields": fields,
    }
