from pathlib import Path

import pandas as pd

from kljpaudit.privacy import scan_public_repository


def test_privacy_scanner_detects_private_column(tmp_path: Path):
    unsafe = tmp_path / "unsafe.csv"

    pd.DataFrame({
        "case_id": [1],
        "facts": ["private judicial text"],
    }).to_csv(unsafe, index=False)

    violations = scan_public_repository(tmp_path)

    assert violations
    assert "facts" in violations[0]


def test_privacy_scanner_accepts_aggregate_table(tmp_path: Path):
    safe = tmp_path / "aggregate.csv"

    pd.DataFrame({
        "condition": ["original"],
        "macro_f1": [0.39],
    }).to_csv(safe, index=False)

    assert scan_public_repository(tmp_path) == []
