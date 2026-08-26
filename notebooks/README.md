# Extended experiment notebooks

## Notebooks

1. `01_multiseed_replication.ipynb`
   - matched-protocol seeds 13, 42, and 100
   - performance variation
   - prediction agreement

2. `02_koelectra_baseline.ipynb`
   - KoELECTRA-base-v3 baseline

3. `03_koelectra_core_audits.ipynb`
   - legal-conclusion ablation
   - outcome-cue insertion

4. `04_case_category_analysis.ipynb`
   - seven criminal case categories
   - KLUE-RoBERTa and KoELECTRA comparison

## Private local artifacts

Set `KLJPAUDIT_ROOT` to the local private project directory.

Do not commit raw judicial text, transformed case text,
per-case predictions, checkpoints, review workbooks, or
case-identifier linkage files.
