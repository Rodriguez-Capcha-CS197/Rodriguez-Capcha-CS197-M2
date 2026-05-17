"""Job 2: Build SciFact segments and run lambda sweep for oracle records."""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Sequence

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from configs.metadata import save_run_metadata
from shared.beir_data import resolve_beir_dataset_path
from shared.beir_scoring import build_beir_segments, scifact_ndcg
from shared.constants import ETA_REDUNDANCY, LAMBDA_GRID, MAX_SEGMENTS
from shared.embedding import MiniLMEmbedder
from shared.io_utils import save_records
from shared.lambda_inference import build_training_feature_record
from shared.logging_utils import configure_logging
from shared.schemas import QueryExample
from shared.scoring import ensure_1d
from shared.segments import SegmentBuildConfig
from shared.sweep_utils import BestSweepChoice, is_better_sweep_choice
from ska_agent.core.pricing import PricingEngine


LOGGER = logging.getLogger(__name__)


def _download_scifact(data_dir: str) -> str:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    return util.download_and_unzip(url, data_dir)


def _prepare_query_examples(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> List[QueryExample]:
    rows = []
    for qid, text in queries.items():
        if qid not in qrels or not qrels[qid]:
            continue
        rows.append(QueryExample(qid=qid, query=text, relevant_doc_ids=list(qrels[qid].keys())))
    return rows


def _retrieved_doc_ids(result, segment_to_doc_id):
    ids = []
    for seg in result.segments:
        doc_id = segment_to_doc_id.get(int(seg.start_idx))
        if doc_id is not None:
            ids.append(doc_id)
    return ids


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="outputs/beir_data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-path", type=str, default="outputs/scifact_sweep_records.json")
    parser.add_argument("--segment-strategy", type=str, default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    data_path = resolve_beir_dataset_path(
        args.data_dir,
        "scifact",
        args.split,
        allow_download=args.download,
        downloader=_download_scifact,
    )
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

    embedder = MiniLMEmbedder(args.embed_model)
    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )
    segments, segment_to_doc_id = build_beir_segments(corpus, embedder, segment_config=segment_config)
    corpus_embs = np.asarray([seg.vector for seg in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)
    examples = _prepare_query_examples(queries, qrels)
    lambda_grid = list(LAMBDA_GRID)

    query_embedding_cache = {ex.query: ensure_1d(embedder.embed_single(ex.query)) for ex in examples}

    def cached_embed_fn(text):
        if text in query_embedding_cache:
            return query_embedding_cache[text]
        return ensure_1d(embedder.embed_single(text))

    engines = {
        lam: PricingEngine(
            segments=segments,
            embed_fn=cached_embed_fn,
            lambda_sparsity=float(lam),
            eta_redundancy=ETA_REDUNDANCY,
            max_segments=MAX_SEGMENTS,
        )
        for lam in lambda_grid
    }

    records = []
    for idx, ex in enumerate(examples, start=1):
        per_query = []
        best = BestSweepChoice()

        for lam in lambda_grid:
            result = engines[lam].retrieve(ex.query, verbose=False)
            ndcg = scifact_ndcg(
                retrieved_segments=result.segments,
                query_id=ex.qid,
                qrels=qrels,
                segment_to_doc_id=segment_to_doc_id,
                k=10,
            )
            row = {
                "question_id": ex.qid,
                "query": ex.query,
                "lambda": float(lam),
                "ndcg_at_10": float(ndcg),
                "num_segments": int(len(result.segments)),
                "relevant_doc_ids": list(ex.relevant_doc_ids),
                "retrieved_doc_ids": _retrieved_doc_ids(result, segment_to_doc_id),
                "segment_strategy": segment_config.strategy,
                "segment_config": segment_config.to_dict(),
                "num_corpus_segments": int(len(segments)),
                "is_optimal": False,
            }
            per_query.append(row)

            k = len(result.segments)
            if is_better_sweep_choice(float(ndcg), k, float(lam), best):
                best = BestSweepChoice(
                    score=float(ndcg),
                    row_index=len(per_query) - 1,
                    num_segments=k,
                    lambda_value=float(lam),
                )

        per_query[best.row_index]["is_optimal"] = True
        query_emb = ensure_1d(query_embedding_cache[ex.query]).astype(np.float32)
        per_query[best.row_index].update(build_training_feature_record(query_emb, corpus_embs, corpus_norms))
        records.extend(per_query)

        if idx % 100 == 0:
            LOGGER.info("processed %d/%d queries", idx, len(examples))

    save_records(records, args.output_path)
    save_run_metadata(
        f"{args.output_path}.metadata.json",
        args,
        extra={"dataset": "scifact", "segment_config": segment_config.to_dict()},
    )


if __name__ == "__main__":
    main()
