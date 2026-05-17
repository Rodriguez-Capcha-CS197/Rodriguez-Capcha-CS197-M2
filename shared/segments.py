"""Shared corpus-to-segment builders for real-data retrieval jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ska_agent.core.geometry import GeometryLearner
from ska_agent.core.structures import Segment

from .fineweb_loader import split_into_sentences


SEGMENT_STRATEGY_GEOMETRY = "geometry_sentence"
SEGMENT_STRATEGY_LEGACY = "legacy_passage"


@dataclass(frozen=True)
class SegmentBuildConfig:
    """Configuration that defines the retrieval unit used by a run."""

    strategy: str = SEGMENT_STRATEGY_GEOMETRY
    min_sentence_len: int = 20
    min_segment_size: int = 2
    max_segment_size: int = 15
    lookback_k: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def _legacy_passage_segments(
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    embedder,
) -> tuple[list[Segment], dict[int, str]]:
    vectors = embedder.embed(list(doc_texts))
    segments: list[Segment] = []
    segment_to_doc_id: dict[int, str] = {}
    for i, (doc_id, text, vec) in enumerate(zip(doc_ids, doc_texts, vectors)):
        segment = Segment(
            text=str(text),
            vector=np.asarray(vec, dtype=np.float64),
            start_idx=i,
            end_idx=i + 1,
            sentences=[str(text)],
            internal_cost=0.0,
        )
        segments.append(segment)
        segment_to_doc_id[i] = str(doc_id)
    return segments, segment_to_doc_id


def _geometry_sentence_segments(
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    embedder,
    config: SegmentBuildConfig,
) -> tuple[list[Segment], dict[int, str]]:
    segments: list[Segment] = []
    segment_to_doc_id: dict[int, str] = {}
    global_idx = 0

    for doc_id, text in zip(doc_ids, doc_texts):
        sentences = split_into_sentences(str(text), min_len=config.min_sentence_len)
        if not sentences:
            stripped = str(text).strip()
            if not stripped:
                continue
            sentences = [stripped]

        sentence_embs = np.asarray(embedder.embed(sentences), dtype=np.float64)
        learner = GeometryLearner(
            lambda_seg=None,
            lookback_k=config.lookback_k,
            min_segment_size=config.min_segment_size,
            max_segment_size=config.max_segment_size,
        )
        learned_segments = learner.learn_geometry(sentence_embs, sentences, verbose=False)

        for learned in learned_segments:
            segment = Segment(
                text=learned.text,
                vector=np.asarray(learned.vector, dtype=np.float64),
                start_idx=global_idx,
                end_idx=global_idx + 1,
                sentences=list(learned.sentences),
                internal_cost=float(learned.internal_cost),
            )
            segments.append(segment)
            segment_to_doc_id[global_idx] = str(doc_id)
            global_idx += 1

    return segments, segment_to_doc_id


def build_segments_from_docs(
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    embedder,
    config: SegmentBuildConfig | None = None,
) -> tuple[list[Segment], dict[int, str]]:
    """Build retrieval segments and map each segment index to its source doc id."""
    config = config or SegmentBuildConfig()
    if len(doc_ids) != len(doc_texts):
        raise ValueError(f"doc_ids/doc_texts length mismatch: {len(doc_ids)} != {len(doc_texts)}")

    if config.strategy == SEGMENT_STRATEGY_LEGACY:
        return _legacy_passage_segments(doc_ids, doc_texts, embedder)
    if config.strategy == SEGMENT_STRATEGY_GEOMETRY:
        return _geometry_sentence_segments(doc_ids, doc_texts, embedder, config)

    raise ValueError(
        f"Unknown segment strategy {config.strategy!r}. "
        f"Expected {SEGMENT_STRATEGY_GEOMETRY!r} or {SEGMENT_STRATEGY_LEGACY!r}."
    )
