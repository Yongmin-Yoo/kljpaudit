"""Artifact paths for local, Colab, and server environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve project artifact directories without hard-coded Colab paths."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def raw_data(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def annotations(self) -> Path:
        return self.root / "data" / "annotations"

    def ensure_runtime_directories(self) -> None:
        for path in (
            self.raw_data,
            self.processed_data,
            self.checkpoints,
            self.results,
            self.annotations,
        ):
            path.mkdir(parents=True, exist_ok=True)
