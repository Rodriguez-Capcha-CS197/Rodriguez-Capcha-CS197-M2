"""Shared lightweight data containers for job scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class QueryExample:
    qid: str
    query: str
    relevant_doc_ids: Sequence[str]
