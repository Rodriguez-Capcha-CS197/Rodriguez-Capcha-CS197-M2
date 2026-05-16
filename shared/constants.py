"""Shared constants for all weeks. All notebooks read from here."""

LAMBDA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ETA_REDUNDANCY = 0.0
MAX_SEGMENTS = 5
TIE_EPS = 1e-8
EPS = 1e-10

TARGET_K_BY_MODE = {
    "LOOKUP": 1,
    "MULTI_DOC": 2,
    "COMPUTE": 2,
    "MULTI_STEP": 3,
}

MISSING_EVIDENCE_PENALTY = 0.25
EXTRA_SEGMENT_PENALTY = 0.04
WRONG_PICK_PENALTY = 0.02

MODES = ["LOOKUP", "MULTI_DOC", "COMPUTE", "MULTI_STEP"]
MODE_COLORS = {
    "LOOKUP": "steelblue",
    "MULTI_DOC": "darkorange",
    "COMPUTE": "forestgreen",
    "MULTI_STEP": "crimson",
}
LAMBDA_BINS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]