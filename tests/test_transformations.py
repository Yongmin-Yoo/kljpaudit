from kljpaudit.transformations.controlled_conclusion import (
    build_controlled_variants,
)
from kljpaudit.transformations.legal_conclusion import (
    build_ablation_variants,
)
from kljpaudit.transformations.outcome_cues import (
    append_outcome_cue,
)


def test_legal_conclusion_ablation():
    text = "앞 문장이다. 이로써 피고인은 범행을 하였다."

    result = build_ablation_variants(text)

    assert result["matched"] is True
    assert "이로써" not in result["marker_removed"]
    assert "피고인은 범행을 하였다" not in result["sentence_removed"]


def test_controlled_variants_retain_content():
    text = "앞 문장이다. 이로써 피고인은 범행을 하였다."

    result = build_controlled_variants(text)

    assert result["matched"] is True
    assert "이와 같이" in result["paraphrased"]
    assert result["fronted"].startswith("이로써 피고인은")
    assert "범행을 하였다" in result["paraphrased_fronted"]


def test_outcome_cue_append():
    transformed = append_outcome_cue(
        "기존 사실이다.",
        "severe",
    )

    assert transformed.startswith("기존 사실이다.")
    assert "엄한 처벌" in transformed
