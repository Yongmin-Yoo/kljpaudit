"""Prevent private judicial text and artifacts from public release."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FORBIDDEN_COLUMNS = {
    "facts",
    "reason",
    "ruling",
    "label",
    "matched_sentence",
    "paraphrased_sentence",
    "text_original",
    "text_paraphrased",
    "text_fronted",
    "text_paraphrased_fronted",
    "context",
    "context_before",
    "context_after",
}

FORBIDDEN_SUFFIXES = {
    ".parquet",
    ".arrow",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
}

FORBIDDEN_DIRECTORY_NAMES = {
    "data",
    "checkpoints",
    "predictions",
    "private_data",
    "annotations",
    "candidate_extraction",
}


def scan_public_repository(root: str | Path) -> list[str]:
    """Return privacy and artifact violations in a repository."""
    root = Path(root)
    violations: list[str] = []

    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue

        relative = path.relative_to(root)

        if path.is_dir():
            if path.name.lower() in FORBIDDEN_DIRECTORY_NAMES:
                violations.append(
                    f"Forbidden public directory: {relative}"
                )
            continue

        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(
                f"Forbidden public artifact: {relative}"
            )

        if path.suffix.lower() == ".csv":
            try:
                columns = {
                    str(column).strip().lower()
                    for column in pd.read_csv(
                        path,
                        nrows=0,
                    ).columns
                }

                exposed = sorted(columns & FORBIDDEN_COLUMNS)

                if exposed:
                    violations.append(
                        f"Private columns in {relative}: {exposed}"
                    )
            except Exception as error:
                violations.append(
                    f"Could not inspect CSV {relative}: {error}"
                )

    return violations


def assert_public_repository_safe(root: str | Path) -> None:
    violations = scan_public_repository(root)

    if violations:
        joined = "\n".join(f"- {item}" for item in violations)
        raise RuntimeError(
            "Public repository safety check failed:\n" + joined
        )
