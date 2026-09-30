# Experiment status

## Evidence conventions

- **Manuscript-reported:** completed experiments and aggregate results described
  in *Predicting Judgments or Exploiting Hindsight?*
- **Artifact-checked:** checks performed on available private artifacts during
  the repository synchronization audit.
- **Public readiness:** executable code, tests, and documentation available
  in this repository. This is separate from experiment completion.

## Completed experiments reported in the manuscript

- KLUE-RoBERTa original frozen baseline and diagnostics
- Legal-conclusion marker and full-sentence ablation
- Human validation by two legally trained reviewers: 150 reviewed cases,
  132 cases satisfying the three conjunctive leakage criteria
- Validated 132-case sentence-removal analysis for KLUE-RoBERTa and KoELECTRA (completed; manuscript-reported)
- Controlled paraphrase and relocation with human validation
- Matched-protocol baseline replication with seeds 13, 42, and 100
- KoELECTRA baseline, ablation, and outcome-cue replication
- Neutral, lenient, and severe cue insertion
- Direct severe-versus-lenient and exploratory opposing-cue diagnostics
- Formatting, orthographic, and tokenization diagnostics
- Exploratory seven-category comparison across encoder families

## Artifact checks completed during synchronization

- Split sizes, within-split IDs, and cross-split ID intersections
- Full target label mapping and nested raw labels
- Reference-label agreement for seven baseline prediction files
- Verified ID-format alignment across model prediction files
- Baseline point estimates for the original audit, three seeds, and KoELECTRA
- Paired 351-case ablation point estimates and bootstrap intervals
- Five checkpoint file hashes, tensor structures, and stored metadata
- Original audit and retrained seed-42 validation-based epoch selection
- Public seed-42/KoELECTRA dataset and forward code inspection

The 132-case estimates remain manuscript-reported in this synchronization
audit; they are not listed as independently recomputed artifact checks.

## Public readiness: synchronization in progress

- Human-validation protocol and aggregate documentation: drafted
- Canonical subset selection and independent agreement utilities: synthetic-tested
- Synthetic fixtures: tested; 40 tests passed before the core-script addition
- Private-input evaluation: 351-case script smoke-tested; partial table mapping documented
- Checkpoint provenance and current audit-environment versions: documented
- Notebook outputs and release-pattern precheck: checked; final diff review pending
- Release approval and public push: pending

An implementation must not be called executable or tested until its actual
files and tests are present and checked.

## Additional experiments not yet completed

1. Human-validated non-leakage sentence deletion controls
2. Core intervention replication across existing seed checkpoints
3. Validated content-type subgroup analysis

A direct factual-conflict experiment is not claimed. Broader retraining or
model expansion is optional and follows the core checks above.
