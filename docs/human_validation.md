# Human validation

## Status and source

Human validation was completed as reported in the manuscript
*Predicting Judgments or Exploiting Hindsight?*, Section 3.7 and Appendix E.4.
The results below are manuscript-reported aggregates.

## Reviewers and sampling

Two legally trained reviewers independently reviewed 150 cases sampled from
the 329-case fully visible rewriting subset using seed 42.

- One reviewer held a doctoral degree in law.
- The other held a bachelor's degree in law and more than five years of
  legal practice experience.

Reviewers received the original description, selected sentence, paraphrase,
and sentence-removed description. They were blinded to benchmark labels,
separately presented final dispositions, model predictions, intervention
effects, and each other's judgments.

A seed alone does not define an exact sample: the sampling implementation
and source case order must also be specified for exact reproduction.

## Retained annotation fields

| Canonical field | Categories |
|---|---|
| legal_conclusion_match | yes / no / unclear |
| meaning_preserved | yes / no / unclear |
| facts_preserved_paraphrase | yes / no / unclear |
| grammatically_natural | yes / no / unclear |
| no_new_outcome_cue | yes / no / unclear |
| content_type | factual_description / offense_conclusion / sentencing_assessment / explicit_outcome / mixed_unclear |
| outcome_relation | outcome_independent / outcome_contingent_revealing / unclear |
| prospective_availability | available / not_independently_available / unclear |
| facts_preserved_after_removal | yes / no / unclear |

These names define a canonical schema, not a claim that every historical
workbook contains these exact columns. Legacy fields require explicit,
semantically justified mapping. Paraphrase fact preservation and minimal-span
separability must not substitute for full-sentence removal fact preservation.

## Validated-subset rule

Include a case only when the adjudicated labels jointly satisfy:

1. outcome_relation = outcome_contingent_revealing;
2. prospective_availability = not_independently_available;
3. facts_preserved_after_removal = yes.

Use an AND filter, exclude unresolved cases, reject duplicate case IDs, and
align model cases by verified identifiers. Never select cases using prediction
changes, model correctness, or effect magnitude.

## Agreement and adjudication

Agreement statistics use the original independent judgments. Validity counts
and subset membership use adjudicated judgments.

The manuscript reports:
- 1,317 of 1,350 field-level decisions agreed: 97.6% raw agreement.
- 24 of 150 cases contained at least one disagreement.
- Disagreements were discussed; unresolved judgments were assigned unclear.

A populated adjudication record is required for adjudication-dependent
computation. An empty adjudication template is not a final decision, and a
script must not invent consensus labels.

## Manuscript-reported adjudicated aggregates

| Criterion | Count |
|---|---:|
| Legal-conclusion match | 149 |
| Meaning preserved | 149 |
| Facts preserved under paraphrase | 148 |
| Grammatically natural | 145 |
| No new outcome cue | 150 |
| Facts preserved after complete sentence removal | 144 |
| Outcome-contingent or outcome-revealing | 138 |
| Not independently prospectively available | 140 |
| All three validated-subset criteria | 132 |

Content types: factual description 2, offense conclusion 8, sentencing
assessment 105, explicit outcome 31, mixed/unclear 4.

The validated sample proportion is 132/150 = 88.0%, with a manuscript-reported
Wilson 95% interval of approximately [81.8%, 92.3%]. This is not detector recall
or benchmark-wide leakage prevalence.

## Manuscript-reported 132-case sentence-removal effects

| Model | Joint flip | Delta mean Macro-F1 | Delta joint EM |
|---|---:|---:|---:|
| KLUE-RoBERTa | 44.70% | -0.0820 | -0.1818 |
| KoELECTRA | 43.18% | -0.0610 | -0.1742 |

These values were not independently recomputed in the synchronization audit.

## Interpretation and privacy

Post-hoc authorship alone does not establish prospective unavailability.
An offense summary, sentencing assessment, and explicit result are different
content types. Review the source context rather than assuming the distinction
from a marker.

Private source texts, individual annotations, case IDs, predictions, and
checkpoints are not public release artifacts. Public examples must be
explicitly synthetic; synthetic examples cannot validate the reported
empirical subset.
