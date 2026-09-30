# Recompute the core 351-case ablations

This script reads authorized private LBOX OPEN artifacts and existing
predictions. It does not load a checkpoint, train a model, or infer annotations.

From the repository root:

    PYTHONPATH=src python scripts/reproduce_core_ablation.py --project-root /path/to/K-LJPAUDIT --output-dir /path/to/K-LJPAUDIT/results/_verification_private/core_reproduction

It checks raw test labels, ID correspondence, detector-match count, cross-model
case membership, and original prediction agreement with baseline predictions.
It then calls the fixed-label-space paired evaluator.

Outputs are aggregate CSV/JSON tables and input-file fingerprints, stored
outside the public checkout. Nothing is copied into public aggregate results
automatically.

## Manuscript table correspondence

| Output scope | Manuscript correspondence |
|---|---|
| KLUE 351-case marker/sentence removal | Table 1, full matched ablation rows |
| KLUE target flip rates | Table 13, all-matched rows |
| KoELECTRA 351-case marker/sentence removal | Table 2, full matched ablation rows |
| KoELECTRA aggregate results | Table 15, all-matched rows |

The script does not reproduce the 132-case validated rows, Table 14, rewriting,
cue insertion, or every appendix result. It must not be described as complete
paper reproduction.

The public annotation selector and paired evaluator can support a separately
authorized validated-subset workflow. Providing a schema or synthetic fixture
does not establish the empirical sample's annotation or sampling provenance.

Exact bootstrap interval agreement requires matching case order, random-number
implementation, and metric implementation. Seeds must not be varied to match
a desired interval.
