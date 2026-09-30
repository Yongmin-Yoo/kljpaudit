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
- **RQ3:** Do inserted directional cues systematically control predicted
  severity, including in reference-outcome-stratified diagnostics?
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

The manuscript distinguishes all detector matches from a manually validated
leakage subset. The detector matches 351 of 1,050 test cases. Human review
covers 150 cases sampled from the 329-case fully visible rewriting subset.

| Analysis | Model | N | Joint flip | Delta mean Macro-F1 | Delta joint EM |
|---|---|---:|---:|---:|---:|
| Marker removal, all matched | KLUE-RoBERTa | 351 | 2.85% | +0.002 | +0.009 |
| Sentence removal, all matched | KLUE-RoBERTa | 351 | 34.76% | -0.048 | -0.125 |
| Sentence removal, all matched | KoELECTRA | 351 | 34.47% | -0.024 | -0.128 |
| Sentence removal, validated subset | KLUE-RoBERTa | 132 | 44.70% | -0.082 | -0.1818 |
| Sentence removal, validated subset | KoELECTRA | 132 | 43.18% | -0.061 | -0.1742 |

The values above are manuscript-reported results. The original baseline and
351-case ablation point estimates were also independently recomputed from
private prediction artifacts during the repository synchronization audit.
The 132-case values are manuscript-reported, not independently recomputed in
that audit.

The validated subset requires outcome contingency/revelation, lack of
independent prospective availability, and preservation of pre-disposition
facts after complete sentence removal. These criteria provide a more targeted
test than detector matching alone; they do not establish a causal account of
the model's internal reasoning.

| Other intervention | KLUE-RoBERTa joint flip |
|---|---:|
| Neutral cue | 5.41% |
| Lenient cue | 6.31% |
| Severe cue | 7.31% |
| Controlled paraphrase | 3.04% |
| Sentence fronting | 5.17% |
| Paraphrase and fronting | 7.60% |

Cue insertion affects a minority of predictions but does not provide consistent
ordinal control. Formatting results distinguish tokenizer-equivalent changes
from visible typography changes with unchanged predictions.

Human validation has been completed as reported in the manuscript; its
protocol and aggregate results are described in
[docs/human_validation.md](docs/human_validation.md).

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

The original frozen KLUE-RoBERTa audit checkpoint and the separately retrained
seed-42 checkpoint must not be treated as the same model.

| Checkpoint role | Seed | Selected epoch | Test mean Macro-F1 | Test joint EM |
|---|---:|---:|---:|---:|
| Original frozen KLUE-RoBERTa audit | 42 | 2 | 0.390791 | 0.467619 |
| Matched-protocol KLUE-RoBERTa replication | 13 | 3 | 0.416543 | 0.532381 |
| Matched-protocol KLUE-RoBERTa replication | 42 | 3 | 0.410564 | 0.518095 |
| Matched-protocol KLUE-RoBERTa replication | 100 | 3 | 0.414468 | 0.520952 |
| KoELECTRA replication | 42 | 3 | 0.400817 | 0.509524 |

Baseline metrics in this table were recomputed from private predictions.
Checkpoint metadata confirms the listed seeds where stored and the listed
epochs. Validation histories support the original audit and retrained seed-42
checkpoint selections. These checks do not by themselves establish the cause
of their performance difference or prove checkpoint-to-prediction provenance
through rerun inference.

The matched-protocol three-seed analysis concerns baseline stability; the
manuscript does not report a complete multi-seed replication of every
intervention.

Evaluation uses the full target label spaces: 5 fine classes, 6
imprisonment-with-labor classes, and 5 imprisonment-without-labor classes.
Case identifiers must be aligned before paired comparisons. An identifier
format change is not evidence of a different case set; joins must be checked
against the underlying test IDs.

Paired intervals use 1,000 case-level resamples with seed 42. Exact interval
reproduction additionally requires the same case order, random-number
implementation, and metric implementation.

Checkpoints, case-level predictions, and annotation workbooks remain outside
the public repository. Public code operates on authorized private inputs or
explicitly synthetic examples. Documentation of a result is not, by itself,
an executable reproduction route.

See [docs/experiment_status.md](docs/experiment_status.md) for the distinction
between manuscript experiment completion and public artifact readiness.

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


### Synchronization artifact guide

- [351-case execution guide](docs/core_ablation_reproduction.md)
- [Human-validation protocol](docs/human_validation.md)
- [Independent agreement aggregation](docs/annotation_aggregation.md)
- [Paired evaluation](docs/paired_evaluation.md)
- [Checkpoint provenance](docs/checkpoint_provenance.md)
- [Audit environment](docs/audit_environment.md)
- [Manuscript-to-code map](docs/table_script_map.md)
- [Anonymous review export](docs/anonymous_release.md)

## Current scope

This project is a paired behavioral audit, not a mechanistic analysis of
internal model reasoning. The full 351-case removal analysis measures
sentence-level information dependence. The manually validated 132-case
analysis targets outcome-related information judged removable without
deleting necessary pre-disposition facts.

The manuscript reports completed human validation, three-seed baseline
replication, two encoder families, formatting/tokenization diagnostics, and
exploratory category and opposing-cue analyses.

The opposing-cue diagnostic uses reference outcomes as a proxy; it is not a
direct manipulation of aggravating or mitigating facts. Cue insertion keeps
benchmark labels fixed for diagnosis, not because every inserted assessment
constitutes a legally valid label-preserving counterfactual.

Further work is prioritized as follows:
1. Fact-preserving, length/position-matched non-leakage sentence deletion
   controls, validated by human reviewers.
2. Core ablation replication across existing seed checkpoints.
3. Exploratory effects by validated sentence content type.

These additional experiments are not reported as completed.

---

## Limitations

1. Experiments cover two Korean encoder families, one benchmark, and seven
   case categories; transfer to other architectures or jurisdictions is not
   established.
2. Three-seed replication evaluates baseline stability, not every intervention.
3. Rule coverage is not the prevalence of hindsight leakage; the detector
   does not measure recall.
4. Human validation samples 150 of the 329 fully visible cases, not all
   detector matches or unmatched expressions.
5. Prospective availability and factual preservation are operational human
   judgments, not evidence of the actual drafting time of a source sentence.
6. Sentence deletion may alter discourse coherence even when facts remain.
7. Inputs are limited to 512 tokens; deletion can affect visible context.
8. Cue and rewriting conclusions apply to the tested templates and positions.
9. Rare punishment levels are sparse, and independent heads can produce
   combinations absent from the reference labels.
10. Category associations and opposing-cue analyses are exploratory;
    nonsignificance is not evidence of equivalence.

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

## 추가 실험 B: 규칙 탐지 문장 교체

기존 132건 전문가 검증 분석과 별도로, 나머지 문서를 고정한
동일·다른 라벨 문장 교체 실험을 수행했다.
5개 checkpoint에서 테스트 추론·집계·주요 지표 검산을 완료했다.

징역 target flip의 다른 라벨 교체 − 동일 라벨 교체 차이는
재학습 KLUE 3 seeds 평균 +3.66 pp,
명목 95% CI [1.61, 6.14] pp였다.
donor 방향 확률 차이는 +0.332 pp,
CI [-0.047, 0.876] pp로 방향성 근거는 제한적이다.

공통 사례 수는 벌금 68건·징역 215건·금고 5건이다.
금고는 탐색적으로만 보고한다.
이 실험에 기존 전문가 평가 또는 132건 subset을 사용하지 않았다.

- 실험 정의·실행 범위: [문서](docs/cue_swap_B.md)
- 확정 입력에서 재계산: [스크립트](scripts/reproduce_cue_swap_B.py)
- 평가 모듈: [코드](src/kljpaudit/cue_swap.py)
- 논문 삽입 초안: [LaTeX](docs/manuscript_cue_swap_B.tex)
- 집계표: `results/aggregate/cue_swap_B_*.csv`

공개 실행 스크립트는 확정된 비공개 연결표와 확률에서
행동 지표·paired CI·성능 점추정치를 재계산한다.
donor 선정·모델 추론·성능 CI·seed 요약 전체를 재생성하는
종단간 파이프라인으로 설명하지 않는다.
