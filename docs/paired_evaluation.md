# Paired evaluation

`kljpaudit.paired_evaluation.evaluate_paired` accepts two lists of records:

- `case_uid`: verified, unique string identifier;
- `y_true`: three integer reference labels;
- `y_pred`: three integer predictions.

Target order is fine, imprisonment with labor, imprisonment without labor.
The full class counts are fixed to 5, 6, and 5. An absent class contributes
zero F1, rather than being dropped from the macro average.

Original and transformed case-ID sets must match. Reference labels are checked
for every case before optional subset filtering. IDs are sorted before paired
case-level resampling. ID-format reconciliation must be verified upstream;
this evaluator does not guess prefixes or join by row order.

A subset is supplied explicitly. For the validated analysis, derive it from
authorized final annotations using `select_validated_case_ids`, never from
prediction changes or correctness. The evaluator does not establish the
provenance or human validity of a supplied subset.

Outputs contain only aggregates: target accuracy/Macro-F1, mean Macro-F1,
joint exact match, target/joint flips, paired metric deltas, and percentile
95% intervals. Differences are transformed minus original.

Defaults are 1,000 paired case resamples and seed 42 using NumPy default_rng.
No pooling of seeds as independent cases is performed.

Run:

    PYTHONPATH=src python -m pytest tests/test_paired_evaluation.py -q

`examples/synthetic_paired_predictions.json` is wholly synthetic. Its outputs
are software checks, not reproduction of manuscript empirical results.
Private-artifact adapters and manuscript table generation are separate steps.
