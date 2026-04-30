"""Scoring function for lambda sweep. Pure function, no side effects."""

import numpy as np
from .constants import (
    EPS,
    TARGET_K_BY_MODE,
    MISSING_EVIDENCE_PENALTY,
    EXTRA_SEGMENT_PENALTY,
    WRONG_PICK_PENALTY,
)


def ensure_1d(array):
    arr = np.asarray(array, dtype=np.float64)
    return arr.reshape(-1)


def get_selected_indices(result):
    return [int(seg.start_idx) for seg in result.segments]


def complexity_aligned_lambda_score(selected_indices, relevant_indices, mode):
    """Score lambda based on complexity-aligned evidence coverage."""
    selected_set = set(int(x) for x in selected_indices)
    relevant_set = set(int(x) for x in relevant_indices)

    target_k = TARGET_K_BY_MODE[mode]
    num_selected = len(selected_set)
    num_relevant_retrieved = len(selected_set & relevant_set)

    evidence_coverage = min(num_relevant_retrieved, target_k) / target_k
    missing_evidence = max(0, target_k - num_relevant_retrieved)
    extra_segments = max(0, num_selected - target_k)

    score = (
        evidence_coverage
        - MISSING_EVIDENCE_PENALTY * missing_evidence
        - EXTRA_SEGMENT_PENALTY * extra_segments
        - WRONG_PICK_PENALTY * (num_selected - num_relevant_retrieved)
    )

    precision = 0.0 if num_selected == 0 else num_relevant_retrieved / num_selected
    recall = 0.0 if len(relevant_set) == 0 else num_relevant_retrieved / len(relevant_set)

    if precision + recall < EPS:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "target_k": int(target_k),
        "num_selected": int(num_selected),
        "num_relevant_retrieved": int(num_relevant_retrieved),
        "missing_evidence": int(missing_evidence),
        "extra_segments": int(extra_segments),
        "evidence_coverage": float(evidence_coverage),
    }
    return float(score), metrics


def dollars(value):
    return f"${value:,.0f}"