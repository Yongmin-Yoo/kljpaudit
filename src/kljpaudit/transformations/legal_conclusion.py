"""High-precision legal-conclusion sentence transformations."""

from __future__ import annotations

import re
from typing import Any


def normalize_space(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", str(text))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_spans(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"[^.!?\n。]+(?:[.!?。]+|\n+|$)",
        re.MULTILINE,
    )

    return [
        {
            "start": match.start(),
            "end": match.end(),
            "sentence": match.group(0),
        }
        for match in pattern.finditer(str(text))
        if match.group(0).strip()
    ]


def locate_conclusion_sentence(
    text: str,
    marker: str = "이로써",
    required_term: str = "피고인",
) -> dict[str, Any] | None:
    matches = [
        span
        for span in sentence_spans(text)
        if marker in span["sentence"]
        and required_term in span["sentence"]
    ]

    return matches[-1] if matches else None


def remove_span(text: str, start: int, end: int) -> str:
    left = text[:start].rstrip()
    right = text[end:].lstrip()
    separator = " " if left and right else ""
    return normalize_space(left + separator + right)


def build_ablation_variants(
    text: str,
    marker: str = "이로써",
    required_term: str = "피고인",
) -> dict[str, Any]:
    original = normalize_space(text)

    span = locate_conclusion_sentence(
        original,
        marker=marker,
        required_term=required_term,
    )

    if span is None:
        return {
            "matched": False,
            "original": original,
            "marker_removed": original,
            "sentence_removed": original,
        }

    selected_sentence = span["sentence"]
    marker_removed_sentence = selected_sentence.replace(marker, "")

    marker_removed = normalize_space(
        original[:span["start"]]
        + marker_removed_sentence
        + original[span["end"]:]
    )

    sentence_removed = remove_span(
        original,
        span["start"],
        span["end"],
    )

    return {
        "matched": True,
        "original": original,
        "marker_removed": marker_removed,
        "sentence_removed": sentence_removed,
    }
