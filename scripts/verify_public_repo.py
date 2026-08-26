#!/usr/bin/env python
"""Run privacy checks before pushing a public release."""

from __future__ import annotations

import argparse
from pathlib import Path

from kljpaudit.privacy import assert_public_repository_safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    assert_public_repository_safe(args.repo_root)
    print("Public repository privacy check: PASSED")


if __name__ == "__main__":
    main()
