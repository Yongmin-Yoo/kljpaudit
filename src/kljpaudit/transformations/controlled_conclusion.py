"""Controlled paraphrase and sentence-position perturbations."""

from __future__ import annotations

from typing import Any

from .legal_conclusion import (
    locate_conclusion_sentence,
    normalize_space,
    remove_span,
)


def build_controlled_variants(
    text: str,
    original_marker: str = "이로써",
    paraphrased_marker: str = "이와 같이",
    required_term: str = "피고인",
) -> dict[str, Any]:
    original = normalize_space(text)

    span = locate_conclusion_sentence(
        original,
        marker=original_marker,
        required_term=required_term,
    )

    if span is None:
        return {
            "matched": False,
            "original": original,
            "paraphrased": original,
            "fronted": original,
            "paraphrased_fronted": original,
        }

    sentence = normalize_space(span["sentence"])
    paraphrased_sentence = sentence.replace(
        original_marker,
        paraphrased_marker,
    )

    paraphrased = normalize_space(
        original[:span["start"]]
        + paraphrased_sentence
        + original[span["end"]:]
    )

    remainder = remove_span(
        original,
        span["start"],
        span["end"],
    )

    fronted = normalize_space(
        sentence + (" " + remainder if remainder else "")
    )

    paraphrased_fronted = normalize_space(
        paraphrased_sentence
        + (" " + remainder if remainder else "")
    )

    return {
        "matched": True,
        "original": original,
        "paraphrased": paraphrased,
        "fronted": fronted,
        "paraphrased_fronted": paraphrased_fronted,
    }
