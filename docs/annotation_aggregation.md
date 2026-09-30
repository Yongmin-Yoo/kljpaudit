# Independent annotation aggregation

`kljpaudit.annotation_agreement.summarize_independent_agreement` compares
two canonical independent annotation inputs using the nine retained fields.

Each record requires `case_uid`, `reviewer_id`, `record_type="independent"`,
and all nine fields defined in `human_validation.FIELD_CATEGORIES`.

Inputs must each contain one reviewer, refer to different reviewers, and have
identical unique case-ID sets. Missing columns and invalid categories fail.
A missing cell is represented by JSON null, not by the category `unclear`.

Raw agreement uses only pairs with both cells annotated. Missing cells are
reported separately. The overall denominator is the sum of both-annotated
pairs across fields. Case disagreement counts do not treat missing cells as
observed disagreements.

The function returns aggregates only and does not generate adjudication.
Historical fields require an explicit justified mapping before use.

The example file is entirely synthetic. Its counts are software-test
expectations, not manuscript empirical results. Synthetic records cannot
restore or substantiate the manuscript's original annotation sample.

Run the tests from the repository root:

    PYTHONPATH=src python -m pytest tests/test_annotation_agreement.py -q
