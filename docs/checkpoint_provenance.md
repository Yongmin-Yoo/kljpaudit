# Checkpoint provenance

These are private artifact references, not downloadable public checkpoints.
Hashes and stored metadata were inspected during synchronization.

| Private artifact reference | Role | Seed | Selected epoch | Head namespace | File SHA256 |
|---|---|---:|---:|---|---|
| `checkpoints/klue_roberta_base_multiseed/seed_100/best_model.pt` | Matched replication | 100 | 3 | `classifiers` | `2e363fccfd89297133da3a993d35373067efcf3ebb5028a8d4c6bdc1feca9dc1` |
| `checkpoints/klue_roberta_base_multiseed/seed_13/best_model.pt` | Matched replication | 13 | 3 | `classifiers` | `fa729b03006ab1a8f51df9875bf1b663cbedaafeba81b437e3a321d349629a86` |
| `checkpoints/klue_roberta_base_multiseed/seed_42/best_model.pt` | Matched replication | 42 | 3 | `heads` | `be81adfcf2b93d658bf993654521e09b3d587e3bcdaaeb47256ecc35039f042a` |
| `checkpoints/klue_roberta_base_multitask_t4/best_model.pt` | Original frozen audit | 42 | 2 | `classifiers` | `706dab125b61ad9d031234cbf5c571188cdad3127797a5f11ddaa090ab1dc257` |
| `checkpoints/koelectra_base_v3_multitask_seed42/best_model.pt` | KoELECTRA replication | 42 | 3 | `classifiers` | `825fdb990fcf9b3ee445533cb65d53e26d3d4df14a63a5d6e7591623d7c0465a` |

## Evidence boundaries

The original audit seed is stored as `config.random_seed`. Other seeds are
stored as `seed`. Selection epochs are stored as `epoch` or `best_epoch`.
All inspected heads have 5/6/5 outputs and 768-dimensional inputs.

A file hash identifies serialized file content. Different file hashes do not
by themselves prove different tensor values: metadata or serialization can
also differ.

Validation histories support epoch 2 for the original frozen audit and
epoch 3 for the separately retrained seed-42. The latter explicitly records
`legacy_checkpoint_used=False`.

The original audit and retrained seed-42 test joint EM values are 0.467619 and
0.518095. Their training executions and selected epochs differ. The causal
reason for the score difference has not been established. The namespace
difference (`classifiers` versus `heads`) is not itself such an explanation.

## Prediction correspondence

- Original audit: baseline test predictions and KLUE ablation original
  predictions agree after verified case-ID alignment.
- KoELECTRA: baseline and ablation original predictions also agree.
- All seven checked baseline prediction files agree with raw reference labels.
- The public multi-seed notebook defines seed-42 retraining and aggregation;
  its presence alone does not establish the historical training implementation
  for seeds 13 and 100.

Checkpoint-to-prediction correspondence through rerun inference remains a
separate check. No checkpoint inference was performed during these checks.

Do not load with `strict=False` merely to bypass head namespace differences.
Inspect the actual model implementation and document any explicit mapping
before a strict compatibility check.

## Input and evaluation boundaries

Public seed-42 and KoELECTRA baseline dataset code tokenizes `facts` only.
The KoELECTRA audit dataset tokenizes a selected transformation column.
Historical execution-code identity is not established solely by the current
public source.

Use verified IDs and raw nested labels for comparisons. Do not align cases by
row order or reference-label combinations. Preserve original artifacts and
record configuration, code revision, selection metric, input order, package
versions, and prediction-source fingerprints for future runs.
