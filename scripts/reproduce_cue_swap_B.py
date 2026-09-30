"""확정 입력 연결표·확률 파일에서 실험 B 지표를 재계산합니다."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from kljpaudit.cue_swap import TARGETS, CONDITIONS, evaluate_target

MODELS = {
    "klue_original_audit42": "klue",
    "klue_seed13": "klue",
    "klue_seed42_retrained": "klue",
    "klue_seed100": "klue",
    "koelectra_seed42": "koelectra",
}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 비공개 출력은 저장소 바깥에만 허용합니다.
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        raise ValueError("비공개 재계산 결과는 저장소 밖에 저장하세요")
    output.mkdir(parents=True, exist_ok=False)

    prepared = args.preparation / "prepared_model_inputs_v1"
    cases = pd.read_parquet(args.preparation / "test_cases_private.parquet")
    if cases["case_uid"].isna().any() or cases["case_uid"].duplicated().any():
        raise ValueError("사례 ID 결측 또는 중복")
    cases = cases.set_index("case_uid")

    links = pd.read_parquet(prepared / "test_input_links_private.parquet")
    draws = pd.DataFrame(read_json(
        args.preparation / "test_donor_draws_private.json"
    ))
    common = pd.DataFrame(read_json(
        prepared / "common_recipient_target_pairs_private.json"
    ))

    if common.duplicated(["recipient_id", "target"]).any():
        raise ValueError("공통 recipient-target 중복")
    if draws.duplicated(["recipient_id", "target", "repetition"]).any():
        raise ValueError("donor 추출 대응 키 중복")
    if set(links["condition"]) != set(CONDITIONS):
        raise ValueError("조건 구성이 다릅니다")

    base = links[links["condition"].isin(("original", "removed"))]
    swaps = links[links["condition"].isin(("same_label", "different_label"))]

    if base.duplicated(["recipient_id", "condition"]).any():
        raise ValueError("원본·삭제 연결 중복")
    if swaps.duplicated(
        ["recipient_id", "target", "repetition", "condition"]
    ).any():
        raise ValueError("교체 연결 중복")

    repeats = sorted(draws["repetition"].unique())
    if len(repeats) != 10:
        raise ValueError("확정된 donor 반복 수가 다릅니다")

    condition_rows, contrast_rows, performance_rows = [], [], []

    for model, family in MODELS.items():
        frame = pd.read_parquet(
            args.inference / model / "test_probabilities_private.parquet"
        )
        if frame["input_key"].duplicated().any():
            raise ValueError("확률 입력 키 중복")
        if not frame["model"].eq(model).all():
            raise ValueError("확률 파일의 모델 이름이 다릅니다")
        frame = frame.set_index("input_key")

        for target_number, target in enumerate(TARGETS):
            recipients = sorted(
                common.loc[common["target"] == target, "recipient_id"]
            )
            n = len(recipients)
            truth = cases.loc[recipients, list(TARGETS)].to_numpy(dtype=int)

            index = pd.MultiIndex.from_product(
                [recipients, repeats],
                names=["recipient_id", "repetition"],
            )
            ordered_draws = (
                draws[draws["target"] == target]
                .set_index(["recipient_id", "repetition"])
                .reindex(index)
            )
            if ordered_draws.isna().any().any():
                raise ValueError("donor 계획 대응 누락")

            for condition, donor_column in (
                ("same_label", "same_label_donor_id"),
                ("different_label", "different_label_donor_id"),
            ):
                selected_links = (
                    swaps[
                        (swaps["target"] == target)
                        & (swaps["condition"] == condition)
                    ]
                    .set_index(["recipient_id", "repetition"])
                    .reindex(index)
                )
                if not np.array_equal(
                    selected_links["donor_id"].to_numpy(),
                    ordered_draws[donor_column].to_numpy(),
                ):
                    raise ValueError("연결표와 추출 계획의 donor가 다릅니다")

                donor_ids = ordered_draws[donor_column].tolist()
                if any(
                    recipient == donor
                    for recipient, donor in zip(
                        ordered_draws.index.get_level_values("recipient_id"),
                        donor_ids,
                    )
                ):
                    raise ValueError("self-donor가 있습니다")

                donor_labels = cases.loc[donor_ids, target].to_numpy(dtype=int)
                recipient_labels = np.repeat(
                    truth[:, target_number], len(repeats)
                )
                if condition == "same_label":
                    if not np.array_equal(donor_labels, recipient_labels):
                        raise ValueError("동일 라벨 donor 조건 위반")
                else:
                    if (donor_labels == recipient_labels).any():
                        raise ValueError("다른 라벨 donor 조건 위반")
                    if not np.array_equal(
                        donor_labels,
                        ordered_draws["different_label_donor_label"].to_numpy(),
                    ):
                        raise ValueError("저장된 donor 라벨이 원본과 다릅니다")

            predictions, probabilities = {}, {}

            for condition in CONDITIONS:
                if condition in ("original", "removed"):
                    selected = (
                        base[base["condition"] == condition]
                        .set_index("recipient_id")
                        .loc[recipients]
                    )
                    count = 1
                else:
                    selected = (
                        swaps[
                            (swaps["condition"] == condition)
                            & (swaps["target"] == target)
                        ]
                        .set_index(["recipient_id", "repetition"])
                        .reindex(index)
                    )
                    count = len(repeats)

                keys = selected[f"{family}_input_key"]
                if keys.isna().any():
                    raise ValueError("입력 연결 키 누락")

                current = frame.loc[keys.tolist()]
                predictions[condition] = np.stack([
                    current[f"pred_{name}"].to_numpy(dtype=int)
                    for name in TARGETS
                ], axis=-1).reshape(n, count, 3)
                probabilities[condition] = np.stack(
                    current[f"prob_{target}"].map(
                        lambda value: np.asarray(value, dtype=float)
                    )
                ).reshape(n, count, -1)

            labels = ordered_draws[
                "different_label_donor_label"
            ].to_numpy(dtype=int).reshape(n, len(repeats))

            result = evaluate_target(
                truth, predictions, probabilities, labels, target,
                bootstrap_repetitions=args.bootstrap,
                bootstrap_seed=args.seed + target_number,
            )

            for destination, source in (
                (condition_rows, "conditions"),
                (contrast_rows, "contrasts"),
                (performance_rows, "performance"),
            ):
                for row in result[source]:
                    destination.append({"model": model, "target": target, **row})

            print(f"집계 완료: {model} | 사례 수 {n}")

    for filename, rows in (
        ("condition_metrics.csv", condition_rows),
        ("paired_behavior_contrasts.csv", contrast_rows),
        ("condition_performance.csv", performance_rows),
    ):
        pd.DataFrame(rows).to_csv(output / filename, index=False)

    print("비공개 재계산 결과:", output)


if __name__ == "__main__":
    main()
