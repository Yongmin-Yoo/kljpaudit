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

## 별도 추가 실험 B: 규칙 탐지 문장 교체

- 테스트 입력·donor 추출 계획 고정: 완료
- 원본 audit, KLUE 재학습 seeds 13·42·100, KoELECTRA seed 42 추론: 완료
- 모델 간 공통 사례: 벌금 68건·징역 215건·금고 5건
- 조건: 원본·C 삭제·동일 라벨 교체·다른 라벨 교체
- 사례 단위 행동 지표·paired CI·반복별 성능 집계: 완료
- 주요 지표 독립 검산 및 전체 원본 recipient baseline 대조: 완료
- 전문가 평가 및 기존 132건 subset: 이 별도 실험에 사용하지 않음
- 금고: 탐색적 분석
- 논문 반영: 삽입용 LaTeX 초안 준비, 실제 논문 소스 편집은 별도
- 공개 코드 범위: 확정 연결표·확률에서 행동 지표·CI·성능 점추정치 재계산
- donor 선정·모델 추론·성능 CI·seed 요약의 종단간 공개 재현: 이번 추가 범위 아님

징역에서 다른 라벨 교체의 target flip 증가가 관찰됐지만,
donor 라벨 방향 이동의 근거는 제한적이다.
기존 전문가 검증 분석 및 추가 삭제 대조군 실험과 구분한다.
