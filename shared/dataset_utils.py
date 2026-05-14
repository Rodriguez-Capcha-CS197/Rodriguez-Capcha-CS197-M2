"""Helpers for combining labeled record files."""

import random

from .io_utils import load_records


def merge_and_shuffle_datasets(*record_paths, seed=0):
    """Load multiple JSON record files, merge rows, and shuffle in place."""
    all_records = []
    for path in record_paths:
        all_records.extend(load_records(path))
    rng = random.Random(seed)
    rng.shuffle(all_records)
    return all_records
