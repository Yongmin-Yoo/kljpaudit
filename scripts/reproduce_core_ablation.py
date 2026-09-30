"""Recompute the 351-case ablations from authorized private artifacts."""

import argparse
import csv
import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from kljpaudit.paired_evaluation import (
    TARGETS, CLASS_COUNTS, evaluate_paired,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = Path(args.output_dir).resolve()
    private_root = (root / "results" / "_verification_private").resolve()
    repo_root = Path(__file__).resolve().parents[1]

    require(output.is_relative_to(private_root),
            "출력 경로는 프로젝트의 비공개 점검 폴더 안이어야 합니다")
    require(not output.is_relative_to(repo_root),
            "공개 저장소 내부에 결과를 저장할 수 없습니다")
    output.mkdir(parents=True, exist_ok=True)

    sources = {}

    def read(relative):
        path = root / relative
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        df = pd.read_parquet(path)
        after = path.stat()
        require(
            before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns,
            "검사 중 입력 파일이 변경되었습니다",
        )
        sources[relative] = {
            "sha256": digest.hexdigest(), "bytes": before.st_size
        }
        return df

    raw = read("data/processed/lbox_ljp_criminal_original/test.parquet")
    require(len(raw) == 1050, "test 건수가 1050이 아닙니다")
    require(raw["id"].notna().all(), "test ID 결측")
    require(not raw["id"].duplicated().any(), "test ID 중복")

    reference = {}
    id_to_uid = {}
    for row in raw[["id", "label"]].to_dict("records"):
        key = str(row["id"])
        uid = "test_" + key
        require(key not in id_to_uid, "문자열 변환 후 test ID 중복")
        require(isinstance(row["label"], dict), "원본 label 형식 오류")
        labels = []
        for target, count in zip(TARGETS, CLASS_COUNTS):
            value = row["label"].get(target)
            require(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and 0 <= value < count,
                "원본 label 범위 또는 자료형 오류",
            )
            labels.append(int(value))
        id_to_uid[key] = uid
        reference[uid] = labels

    def normalize(df):
        df = df.copy()
        require(df["case_uid"].notna().all(), "예측 ID 결측")
        require(df["case_uid"].map(
            lambda value: isinstance(value, str)
        ).all(), "예측 ID 자료형 오류")
        require(not df["case_uid"].duplicated().any(), "예측 ID 중복")

        def canonical(value):
            if value in reference:
                return value
            require(value in id_to_uid, "원본 test와 대응하지 않는 ID")
            return id_to_uid[value]

        df["case_uid"] = df["case_uid"].map(canonical)
        require(not df["case_uid"].duplicated().any(), "대응 후 ID 중복")
        return df.set_index("case_uid").sort_index()

    def integer_values(values, count):
        result = []
        for value in values:
            require(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and 0 <= value < count,
                "예측 파일의 label 범위 또는 자료형 오류",
            )
            result.append(int(value))
        return result

    def records(df, true_columns, predicted_columns):
        result = []
        for uid, row in df.iterrows():
            y_true = [
                integer_values([row[column]], count)[0]
                for column, count in zip(true_columns, CLASS_COUNTS)
            ]
            y_pred = [
                integer_values([row[column]], count)[0]
                for column, count in zip(predicted_columns, CLASS_COUNTS)
            ]
            require(y_true == reference[uid], "예측 파일과 원본 정답 불일치")
            result.append({
                "case_uid": uid, "y_true": y_true, "y_pred": y_pred
            })
        return result

    baseline = read(
        "results/baselines/klue_roberta_base_multitask_t4/"
        "predictions/test_predictions.parquet"
    )
    require(baseline["split"].eq("test").all(), "baseline split 불일치")
    require(baseline["case_uid"].eq(
        "test_" + baseline["id"].astype(str)
    ).all(), "baseline의 ID 생성 대응 불일치")
    baseline = normalize(baseline)
    require(set(baseline.index) == set(reference), "baseline 사례 집합 불일치")

    klue = normalize(read(
        "results/audit_experiments/legal_conclusion_ablation_v1/"
        "predictions/paired_predictions.parquet"
    ))
    require(set(klue.index) == set(reference), "KLUE ablation 사례 집합 불일치")
    true_columns = [f"{target}_true" for target in TARGETS]

    baseline_records = records(
        baseline, true_columns, [f"{target}_predicted" for target in TARGETS]
    )
    klue_all_original = records(
        klue, true_columns, [f"ORIGINAL_{target}_predicted" for target in TARGETS]
    )
    require(baseline_records == klue_all_original,
            "KLUE 원본 예측이 baseline과 다릅니다")

    flag = klue["legal_conclusion_matched"]
    require(flag.notna().all(), "matched 표지 결측")
    require(flag.map(
        lambda value: isinstance(value, (bool, np.bool_, int, np.integer))
        and value in (0, 1)
    ).all(), "matched 표지 형식 오류")
    matched = klue.loc[flag.astype(bool)].copy()
    require(len(matched) == 351, "matched 건수가 351이 아닙니다")

    ko = normalize(read(
        "results/audit_experiments/koelectra_legal_conclusion_ablation_v1/"
        "predictions/koelectra_legal_conclusion_predictions_private.parquet"
    ))
    require(set(ko.index) == set(matched.index),
            "두 모델의 matched 사례 집합 불일치")

    ko_baseline = normalize(read(
        "results/audit_experiments/koelectra_baseline_generalization_v1/"
        "predictions/koelectra_seed42_test_predictions_private.parquet"
    ))
    require(set(ko_baseline.index) == set(reference),
            "KoELECTRA baseline 사례 집합 불일치")

    ko_original = records(
        ko, list(TARGETS), [f"pred_original_{target}" for target in TARGETS]
    )
    ko_baseline_subset = records(
        ko_baseline.loc[ko.index], true_columns,
        [f"pred_{target}" for target in TARGETS],
    )
    require(ko_original == ko_baseline_subset,
            "KoELECTRA 원본 예측이 baseline과 다릅니다")

    evaluations = []
    for model, df, truth, original_columns, conditions in [
        (
            "klue_roberta", matched, true_columns,
            [f"ORIGINAL_{target}_predicted" for target in TARGETS],
            {
                "marker_removed": [
                    f"MARKER_REMOVED_{target}_predicted" for target in TARGETS
                ],
                "sentence_removed": [
                    f"CONCLUSION_SENTENCE_REMOVED_{target}_predicted"
                    for target in TARGETS
                ],
            },
        ),
        (
            "koelectra", ko, list(TARGETS),
            [f"pred_original_{target}" for target in TARGETS],
            {
                condition: [
                    f"pred_{condition}_{target}" for target in TARGETS
                ]
                for condition in ["marker_removed", "sentence_removed"]
            },
        ),
    ]:
        original = records(df, truth, original_columns)
        for condition, columns in conditions.items():
            changed = records(df, truth, columns)
            result = evaluate_paired(
                original, changed,
                iterations=args.iterations, seed=args.seed,
            )
            evaluations.append({
                "model": model, "condition": condition,
                "subset": "all_matched_351", **result,
            })

    table = []
    for evaluation in evaluations:
        row = {
            "model": evaluation["model"],
            "condition": evaluation["condition"],
            "subset": evaluation["subset"],
            "n": evaluation["n"],
        }
        for group in ["original", "transformed", "delta", "flip"]:
            for key, value in evaluation[group].items():
                row[f"{group}_{key}"] = value
        for group in ["delta_ci95", "flip_ci95"]:
            for key, interval in evaluation[group].items():
                row[f"{group}_{key}_low"] = interval[0]
                row[f"{group}_{key}_high"] = interval[1]
        table.append(row)

    with (output / "core_ablation_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    (output / "core_ablation_results.json").write_text(
        json.dumps(evaluations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    versions = {}
    for package in ["numpy", "pandas", "pyarrow"]:
        versions[package] = metadata.version(package)

    (output / "reproduction_manifest.json").write_text(
        json.dumps({
            "sources": sources,
            "label_spaces": {
                target: list(range(count))
                for target, count in zip(TARGETS, CLASS_COUNTS)
            },
            "versions": versions,
            "subset_selection": "stored detector flag; not model response",
            "scope": "351건 예측 재계산; 132건 및 checkpoint 추론 검증 아님",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("ID·정답·baseline 대응 검사: 통과")
    for evaluation in evaluations:
        print(
            evaluation["model"], evaluation["condition"],
            "| 건수:", evaluation["n"],
            "| joint flip:", round(evaluation["flip"]["joint_flip"], 6),
            "| F1 차이:", round(evaluation["delta"]["mean_macro_f1"], 6),
            "| EM 차이:", round(evaluation["delta"]["joint_em"], 6),
        )
    print("비공개 결과 폴더:", output)


if __name__ == "__main__":
    main()
