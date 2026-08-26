"""Configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML experiment configuration."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")

    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as stream:
        if suffix == ".json":
            config = json.load(stream)
        elif suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(stream)
        else:
            raise ValueError(
                f"Unsupported configuration format: {path.suffix}"
            )

    if not isinstance(config, dict):
        raise TypeError("The configuration root must be a dictionary.")

    return config


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    """Raise an informative error when required keys are absent."""
    missing = [key for key in keys if key not in config]

    if missing:
        raise KeyError(f"Missing configuration keys: {missing}")
