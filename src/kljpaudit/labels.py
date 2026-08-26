"""Robust loading of the multitask sentencing label maps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TARGET_FIELDS = (
    "fine_lv",
    "imprisonment_with_labor_lv",
    "imprisonment_without_labor_lv",
)

EXPECTED_LABEL_VALUES = {
    "fine_lv": [0, 1, 2, 3, 4],
    "imprisonment_with_labor_lv": [0, 1, 2, 3, 4, 5],
    "imprisonment_without_labor_lv": [0, 1, 2, 3, 4],
}

INDEX_ALIASES = (
    "index_to_label",
    "index_to_raw",
    "idx_to_label",
    "id2label",
)

REVERSE_ALIASES = (
    "label_to_index",
    "raw_to_index",
    "label_to_idx",
    "label2id",
)

VALUE_ALIASES = (
    "raw_label_values",
    "classes",
    "label_values",
    "values",
)


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value

    text = str(value).strip()

    try:
        return int(text)
    except ValueError:
        match = re.search(r"-?\d+", text)

        if match:
            return int(match.group(0))

    raise ValueError(f"Cannot convert label to integer: {value!r}")


def load_label_map(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    if not isinstance(data, dict):
        raise TypeError("Label-map root must be a dictionary.")

    return data


def resolve_target_map(data: dict[str, Any], target: str) -> Any:
    """Support target-first and mapping-first JSON structures."""
    queue = [data]
    visited: set[int] = set()

    while queue:
        current = queue.pop(0)

        if not isinstance(current, dict) or id(current) in visited:
            continue

        visited.add(id(current))

        if target in current and isinstance(
            current[target], (dict, list, tuple)
        ):
            return current[target]

        assembled: dict[str, Any] = {}

        for alias in INDEX_ALIASES:
            container = current.get(alias)

            if isinstance(container, dict) and target in container:
                assembled["index_to_label"] = container[target]

        for alias in REVERSE_ALIASES:
            container = current.get(alias)

            if isinstance(container, dict) and target in container:
                assembled["label_to_index"] = container[target]

        for alias in VALUE_ALIASES:
            container = current.get(alias)

            if isinstance(container, dict) and target in container:
                assembled["classes"] = container[target]

        if assembled:
            return assembled

        queue.extend(
            value for value in current.values()
            if isinstance(value, dict)
        )

    raise KeyError(f"Label mapping not found for target: {target}")


def index_to_raw(data: dict[str, Any], target: str) -> dict[int, int]:
    resolved = resolve_target_map(data, target)

    if isinstance(resolved, (list, tuple)):
        return {
            index: _as_int(raw)
            for index, raw in enumerate(resolved)
        }

    for alias in (*INDEX_ALIASES, "index_to_label"):
        mapping = resolved.get(alias)

        if isinstance(mapping, dict):
            return {
                int(index): _as_int(raw)
                for index, raw in mapping.items()
            }

        if isinstance(mapping, (list, tuple)):
            return {
                index: _as_int(raw)
                for index, raw in enumerate(mapping)
            }

    for alias in (*REVERSE_ALIASES, "label_to_index"):
        mapping = resolved.get(alias)

        if isinstance(mapping, dict):
            return {
                int(index): _as_int(raw)
                for raw, index in mapping.items()
            }

    for alias in (*VALUE_ALIASES, "classes"):
        values = resolved.get(alias)

        if isinstance(values, (list, tuple)):
            return {
                index: _as_int(raw)
                for index, raw in enumerate(values)
            }

    raise ValueError(f"Cannot construct index mapping for {target}")


def all_index_to_raw(data: dict[str, Any]) -> dict[str, dict[int, int]]:
    return {
        target: index_to_raw(data, target)
        for target in TARGET_FIELDS
    }


def validate_expected_labels(
    mappings: dict[str, dict[int, int]],
) -> None:
    for target, expected in EXPECTED_LABEL_VALUES.items():
        observed = [
            mappings[target][index]
            for index in sorted(mappings[target])
        ]

        if observed != expected:
            raise ValueError(
                f"Unexpected labels for {target}: "
                f"expected={expected}, observed={observed}"
            )
