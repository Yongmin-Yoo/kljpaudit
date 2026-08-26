# K-LJPAUDIT

[![CI](https://github.com/Yongmin-Yoo/kljpaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/Yongmin-Yoo/kljpaudit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**A controlled behavioral audit of hindsight leakage in Korean legal judgment prediction**

K-LJPAUDIT examines whether legal judgment prediction models rely on legally
relevant case information or on post-hoc expressions and superficial cues
embedded in judicial fact descriptions.

Repository:

https://github.com/Yongmin-Yoo/kljpaudit

Author:

**Yongmin Yoo**

---

## Overview

Legal judgment prediction models are commonly evaluated on judge-authored fact
descriptions. These descriptions may contain formulaic legal conclusions,
sentencing-related language, or expressions written with knowledge of the
eventual judgment. A model may therefore achieve apparently strong performance
without relying on legally grounded evidence.

K-LJPAUDIT evaluates this possibility through paired input interventions:

1. removal of an isolated legal-conclusion marker;
2. removal of the complete legal-conclusion sentence;
3. insertion of neutral, lenient, and severe outcome cues;
4. controlled paraphrasing of a formulaic discourse marker;
5. relocation of the legal-conclusion sentence.

The project is a behavioral audit. It does not claim to identify the model's
internal computational mechanism.

---

## Research questions

- **RQ1:** To what extent do judicial fact descriptions contain post-hoc
  outcome-indicative expressions?
- **RQ2:** How does performance change when these expressions are removed or
  rewritten?
- **RQ3:** When legally relevant facts conflict with superficial cues, which
  information has greater influence?
- **RQ4:** Are predictions robust to legally irrelevant changes in style,
  lexical choice, and judicial phrasing?
- **RQ5:** Do reliance patterns generalize across case categories and model
  families?

---

## Dataset and task

The experiments use the LJP-Criminal subset of LBOX OPEN, containing 10,500
Korean first-instance criminal cases across seven balanced case categories.

| Split | Cases |
|---|---:|
| Train | 8,400 |
| Validation | 1,050 |
| Test | 1,050 |
| Test2 | 928 |

`test2` is a subset of the primary test set and is evaluated separately.

Only the `facts` field is provided to the model. The model predicts:

- fine level;
- imprisonment-with-labor level;
- imprisonment-without-labor level.

The dataset is not redistributed in this repository.

---

## Baseline

The supervised baseline uses KLUE-RoBERTa-base with three independent
classification heads. It is trained with square-root inverse-frequency class
weights and the mean of three weighted cross-entropy losses.

Primary test results:

| Target | Accuracy | Macro-F1 |
|---|---:|---:|
| Fine | 0.627 | 0.447 |
| Imprisonment with labor | 0.684 | 0.396 |
| Imprisonment without labor | 0.940 | 0.330 |

Mean Macro-F1 is **0.391**, and joint exact match is **0.468**.

---

## Current audit results

| Intervention | Primary observation |
|---|---|
| Marker removal | 2.85% joint flip; no significant aggregate degradation |
| Conclusion-sentence removal | 34.76% joint flip; mean Macro-F1 decreases by 0.048 |
| Neutral cue | 5.41% joint flip |
| Lenient cue | 6.31% joint flip |
| Severe cue | 7.31% joint flip |
| Controlled paraphrase | 3.04% joint flip; no significant degradation |
| Sentence fronting | 5.17% joint flip; no significant degradation |
| Paraphrase and fronting | 7.60% joint flip; no significant degradation |

Complete sentence removal produces substantially larger changes than marker
removal, controlled paraphrasing, or sentence relocation. This contrast
suggests dependence on information contained in the legal-conclusion sentence
rather than on the isolated marker or original sentence position. It does not,
however, establish either genuine legal reasoning or hindsight leakage because
the sentence may contain judge-formulated legal conclusions.

The controlled perturbation results remain preliminary pending manual semantic
validation.

---
## Repository structure

```text
configs/                  Experiment configurations
src/kljpaudit/            Reusable Python package
src/kljpaudit/transformations/
                          Controlled text interventions
scripts/                  Command-line utilities
tests/                    Unit and privacy tests
results/aggregate/        Public aggregate results
docs/                     Data and experiment documentation
demo/                     Synthetic examples only
```

The Python package contains reusable implementations of label-map loading,
evaluation metrics, paired bootstrap estimation, privacy validation, and
controlled text transformations. Raw judicial documents and document-level
artifacts are intentionally stored outside this repository.

---

## Installation

### Local environment

Clone the repository and install the package with machine-learning
dependencies:

```bash
git clone https://github.com/Yongmin-Yoo/kljpaudit.git
cd kljpaudit
python -m pip install --upgrade pip
pip install -e ".[ml]"
```

To install development and testing dependencies:

```bash
pip install -e ".[ml,dev]"
pytest
```

### Google Colab

The experiments were developed for Google Colab with persistent artifacts
stored in Google Drive. Run the following commands in a Colab Python cell:

```python
from google.colab import drive

drive.mount("/content/drive")

!git clone https://github.com/Yongmin-Yoo/kljpaudit.git
%cd /content/kljpaudit
!python -m pip install -q --upgrade pip
!pip install -q -r requirements-colab.txt
```

If the repository is already present in the current Colab runtime, update it
instead of cloning it again:

```python
%cd /content/kljpaudit
!git pull
!pip install -q -r requirements-colab.txt
```

The public repository contains source code and aggregate results only. Raw
LBOX OPEN data, transformed judicial text, model checkpoints, document-level
predictions, and annotations must remain outside the cloned repository.

---

## Configuration

Experiment settings are stored under `configs/`. Runtime artifact locations
are passed separately and are not hard-coded into the package.

The baseline configuration is stored in:

```text
configs/baseline_klue_roberta.json
```

The current audit configurations are:

```text
configs/legal_conclusion_ablation.json
configs/outcome_cue_insertion.json
configs/controlled_conclusion_perturbation.json
```

In Google Colab, the external project root is typically:

```text
/content/drive/MyDrive/K-LJPAUDIT
```

This directory is not part of the Git repository. It stores the original
dataset, model checkpoints, transformed inputs, predictions, annotations, and
other private research artifacts.

The reusable command-line experiment interfaces are being migrated from the
original Colab implementation during the current development release. A
command should be treated as available only when its corresponding file is
present under `scripts/`.

---

## Evaluation

The primary task-level metric is Macro-F1 computed over the complete training
label space, including rare labels that may not appear in an individual test
subset. The audit additionally reports:

- Accuracy and Weighted-F1 for each sentencing target;
- mean Macro-F1 across the three targets;
- joint exact match;
- target-level prediction flip rate;
- joint prediction flip rate;
- paired bootstrap confidence intervals;
- severe-versus-lenient directional differences.

Paired confidence intervals use 1,000 case-level bootstrap resamples. The same
sampled case indices are applied to the original and transformed predictions.

---

## Reproducibility

The baseline uses a fixed random seed of 42, KLUE-RoBERTa-base, a maximum
sequence length of 512 tokens, FP16 mixed-precision training, and square-root
inverse-frequency class weighting. The selected baseline checkpoint is the
checkpoint with the highest mean validation Macro-F1.

The public configuration files record the experimental settings. Model
checkpoints and document-level predictions are not distributed through this
repository.

The following aggregate result files are included:

```text
results/aggregate/baseline_results.csv
results/aggregate/legal_conclusion_ablation.csv
results/aggregate/outcome_cue_insertion.csv
results/aggregate/controlled_conclusion_perturbation.csv
```

These tables contain no judgment text or document-level records.

---

## Testing

Install the development dependencies and run:

```bash
pip install -e ".[dev]"
pytest
```

The test suite checks:

- complete-label-space metric calculation;
- joint exact-match calculation;
- robust nested label-map loading;
- legal-conclusion sentence matching;
- marker and sentence removal;
- controlled paraphrasing and sentence relocation;
- outcome-cue insertion;
- public-artifact privacy constraints.

---

## Public repository verification

Before committing or releasing public artifacts, run:

```bash
python scripts/verify_public_repo.py
```

The verification script detects prohibited artifact formats and private
document-level columns, including:

```text
facts
reason
ruling
matched_sentence
text_original
text_paraphrased
text_fronted
```

A failed verification must be resolved before pushing changes to the public
repository.

---

## Data availability

K-LJPAUDIT is derived from the LJP-Criminal subset of LBOX OPEN. This
repository does not redistribute LBOX OPEN judgments. Users must obtain the
dataset from its official source and comply with the corresponding license and
terms.

Official resources:

- LBOX OPEN repository:
  https://github.com/lbox-kr/lbox-open
- LBOX OPEN dataset:
  https://huggingface.co/datasets/lbox/lbox_open
- LBOX OPEN paper:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/d15abd14d5894eebd185b756541d420e-Abstract-Datasets_and_Benchmarks.html

Third-party language models remain subject to their respective model licenses.

---

## Privacy and artifact policy

The following artifacts are not released:

- raw and processed judicial text;
- transformed factual descriptions;
- document-level predictions;
- model checkpoints and tokenizer caches;
- annotation and manual-review workbooks;
- candidate-context extraction files;
- case-level error-analysis files.

Only source code, synthetic examples, experiment configurations, documentation,
and aggregate results are intended for public release.

---

## Current scope

The current experiments constitute a behavioral audit rather than a
mechanistic analysis of the model's internal computation. They measure
prediction sensitivity under specified transformations but do not establish
that the model performs genuine legal reasoning.

The legal-conclusion ablation cannot by itself distinguish legally relevant
offense information from hindsight leakage. Controlled paraphrasing and
sentence relocation provide an initial robustness test, but the corresponding
interpretation remains preliminary until manual semantic validation is
complete. The cue-insertion experiment uses a fixed set of templates and
cannot rule out susceptibility to other cue phrasings, strengths, or positions.

Remaining work includes:

- style-only perturbation;
- explicit fact--cue conflict evaluation;
- multiple-seed replication;
- model-family generalization;
- human annotation and inter-annotator agreement.

---

## Limitations

1. The current baseline uses one pretrained encoder family and one training
   seed.
2. The sentencing labels do not directly encode suspended execution of
   imprisonment.
3. Rare severe-punishment levels contain very few training examples.
4. Inputs longer than 512 tokens are truncated.
5. Independent classification heads may produce legally incoherent
   multi-punishment combinations.
6. Rule-based legal-conclusion matching covers only a subset of test cases.
7. Current cue and paraphrase experiments use a limited number of templates.
8. Controlled perturbations require manual validation before they can be
   described as definitively meaning-preserving.

---

## Citation

If you use this repository, please cite:

```bibtex
@software{yoo2026kljpaudit,
  author = {Yongmin Yoo},
  title  = {K-LJPAUDIT: A Controlled Audit of Hindsight Leakage in Korean Legal Judgment Prediction},
  year   = {2026},
  url    = {https://github.com/Yongmin-Yoo/kljpaudit}
}
```

A machine-readable citation record is provided in `CITATION.cff`.

---

## License

The source code is released under the MIT License.

The MIT License applies only to the code in this repository. LBOX OPEN,
KLUE-RoBERTa, and other third-party datasets and models remain subject to their
original licenses and terms.

---

## Author

**Yongmin Yoo**

Repository:

https://github.com/Yongmin-Yoo/kljpaudit

Issues:

https://github.com/Yongmin-Yoo/kljpaudit/issues
