"""Shared helpers for lambda sweep selection."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import TIE_EPS


@dataclass
class BestSweepChoice:
    score: float = -float("inf")
    row_index: int = 0
    num_segments: int | None = None
    lambda_value: float | None = None


def is_better_sweep_choice(
    score: float,
    num_segments: int,
    lambda_value: float,
    best: BestSweepChoice,
    tie_eps: float = TIE_EPS,
) -> bool:
    """Return True if a sweep row beats the current best with deterministic tie-breaks."""
    if score > best.score + tie_eps:
        return True
    if abs(score - best.score) > tie_eps:
        return False
    if best.num_segments is None or num_segments < best.num_segments:
        return True
    return (
        num_segments == best.num_segments
        and best.lambda_value is not None
        and lambda_value < best.lambda_value
    )
