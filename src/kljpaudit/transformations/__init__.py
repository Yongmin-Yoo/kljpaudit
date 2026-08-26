"""Controlled input transformations used by K-LJPAUDIT."""

from .controlled_conclusion import build_controlled_variants
from .legal_conclusion import build_ablation_variants
from .outcome_cues import append_outcome_cue

__all__ = [
    "append_outcome_cue",
    "build_ablation_variants",
    "build_controlled_variants",
]
