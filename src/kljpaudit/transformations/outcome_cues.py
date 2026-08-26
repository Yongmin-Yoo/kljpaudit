"""Fixed outcome-cue insertion transformations."""

from __future__ import annotations

import unicodedata

DEFAULT_CUES = {
    "neutral": "이 사건의 구체적인 사실관계는 위에서 서술한 바와 같다.",
    "lenient": "이 사건은 참작할 사정이 충분하여 관대한 처벌이 타당하다.",
    "severe": "이 사건은 죄책이 무거워 엄한 처벌이 불가피하다.",
}


def normalize_korean(text: str) -> str:
    """Normalize Hangul to NFC to avoid decomposed-jamo rendering."""
    return unicodedata.normalize("NFC", str(text).strip())


def append_outcome_cue(
    text: str,
    condition: str,
    cues: dict[str, str] | None = None,
) -> str:
    cue_map = DEFAULT_CUES if cues is None else cues
    condition = condition.lower()

    if condition == "original":
        return normalize_korean(text)

    if condition not in cue_map:
        raise KeyError(f"Unknown cue condition: {condition}")

    return normalize_korean(
        f"{text.strip()} {cue_map[condition].strip()}"
    )
