"""Canonical final-annotation validation and subset selection."""

from collections.abc import Mapping

FIELD_CATEGORIES = {'legal_conclusion_match': ['yes', 'no', 'unclear'], 'meaning_preserved': ['yes', 'no', 'unclear'], 'facts_preserved_paraphrase': ['yes', 'no', 'unclear'], 'grammatically_natural': ['yes', 'no', 'unclear'], 'no_new_outcome_cue': ['yes', 'no', 'unclear'], 'content_type': ['factual_description', 'offense_conclusion', 'sentencing_assessment', 'explicit_outcome', 'mixed_unclear'], 'outcome_relation': ['outcome_independent', 'outcome_contingent_revealing', 'unclear'], 'prospective_availability': ['available', 'not_independently_available', 'unclear'], 'facts_preserved_after_removal': ['yes', 'no', 'unclear']}

CRITERIA = {
    "outcome_relation": "outcome_contingent_revealing",
    "prospective_availability": "not_independently_available",
    "facts_preserved_after_removal": "yes",
}


def validate_adjudicated_records(records):
    """Validate final records without constructing or inferring decisions."""
    validated = []
    seen = set()

    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"레코드 {position}: 객체 형식이 아닙니다")

        case_uid = record.get("case_uid")
        if (
            not isinstance(case_uid, str)
            or not case_uid.strip()
            or case_uid != case_uid.strip()
        ):
            raise ValueError(f"레코드 {position}: ID 형식 오류")

        if case_uid in seen:
            raise ValueError(f"레코드 {position}: ID 중복")
        seen.add(case_uid)

        if record.get("record_type") != "adjudicated":
            raise ValueError(f"레코드 {position}: 최종 판정 기록이 아닙니다")

        for field, categories in FIELD_CATEGORIES.items():
            if field not in record:
                raise ValueError(f"레코드 {position}: 필수 판정 열 누락: {field}")
            value = record[field]
            if not isinstance(value, str) or value not in categories:
                raise ValueError(f"레코드 {position}: 판정값 오류: {field}")

        validated.append(dict(record))

    if not validated:
        raise ValueError("최종 판정 기록이 비어 있습니다")
    return validated


def select_validated_case_ids(records):
    """Apply the three adjudicated criteria using a strict AND filter."""
    validated = validate_adjudicated_records(records)
    return tuple(sorted(
        record["case_uid"]
        for record in validated
        if all(record[field] == expected for field, expected in CRITERIA.items())
    ))


def summarize_adjudicated(records):
    """Return aggregate counts only; never emit case IDs or case-level labels."""
    validated = validate_adjudicated_records(records)
    counts = {
        field: {
            category: sum(record[field] == category for record in validated)
            for category in categories
        }
        for field, categories in FIELD_CATEGORIES.items()
    }
    return {
        "reviewed_cases": len(validated),
        "validated_cases": len(select_validated_case_ids(validated)),
        "field_counts": counts,
    }
