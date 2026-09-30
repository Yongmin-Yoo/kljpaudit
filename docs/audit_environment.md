# Audit environment

This records the current CPU artifact-checking and software-test environment.
It is not a reconstruction of the historical training environment.

Python: 3.13.15

| Package | Observed version |
|---|---|
| numpy | `2.1.3` |
| pandas | `2.2.3` |
| pyarrow | `23.0.1` |
| scikit-learn | `1.6.1` |
| openpyxl | `3.1.5` |
| pytest | `8.4.2` |
| jsonschema | `4.26.0` |
| torch | `2.11.0+cpu` |
| transformers | `5.17.0` |
| datasets | `4.8.5` |

`requirements/audit-tested.txt` pins the audit/test dependencies observed here.
It does not pin or install the historical GPU training stack.

Run installations in a separate environment, not in an active Colab training
session. A matching package list is useful but does not guarantee numerical
identity across hardware or all dependency combinations.

The paired evaluator fixes all 5/6/5 classes and assigns zero F1 to absent
classes. Bootstrap uses paired cases, NumPy default_rng, 1,000 resamples,
seed 42, sorted canonical case IDs, and percentile 95% intervals.

The original manuscript CI implementation and case ordering have not been
fully reconciled with this evaluator. Recomputed point estimates agree;
small CI endpoint differences must not be removed by tuning seeds.
