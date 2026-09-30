# Manuscript-to-code map

| Manuscript scope | Public entry point | Readiness |
|---|---|---|
| Tables 1/2, full 351-case ablation rows | `scripts/reproduce_core_ablation.py` | Run on private artifacts |
| Table 13, all-matched target flips | Same script | Run on private artifacts |
| Table 15, all-matched aggregate results | Same script | Run on private artifacts |
| Canonical three-criterion validated selection | `human_validation.select_validated_case_ids` | Synthetic-tested |
| Independent nine-field agreement | `annotation_agreement.summarize_independent_agreement` | Synthetic-tested |
| Generic paired metrics and intervals | `paired_evaluation.evaluate_paired` | Synthetic-tested and used for 351 cases |
| Table 14 and validated rows in Tables 1/2/13/15 | Canonical selector + evaluator building blocks | Empirical 132-case rerun not performed |
| Seed/model baselines, Tables 9–11 | Existing model notebooks; artifact recalculation | Baselines checked; full end-to-end training not rerun |
| Remaining rewriting/cue/category/formatting tables | Existing experiment artifacts and applicable notebooks | No complete executable table mapping established |

`results/aggregate/core_ablation_recomputed.csv` contains four aggregate
351-case intervention rows from the synchronization audit. Its intervals
are recomputed intervals, not a replacement for manuscript intervals.

A manuscript report, a tested building block, and an empirical rerun are
different evidence levels. Do not describe this repository as complete
end-to-end reproduction of every manuscript table.
