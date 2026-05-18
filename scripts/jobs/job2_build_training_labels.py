"""Job 2: Build FineWeb+MS MARCO lambda-labeled training records."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Sequence

import numpy as np

from shared.constants import ETA_REDUNDANCY, LAMBDA_GRID, MAX_SEGMENTS
from configs.metadata import save_run_metadata
from shared.io_utils import save_records
from shared.lambda_inference import build_training_feature_record
from shared.marco_loader import load_marco_sample
from shared.embedding import MiniLMEmbedder
from shared.logging_utils import configure_logging
from shared.schemas import QueryExample
from shared.scoring import ensure_1d
from shared.segments import SegmentBuildConfig, build_segments_from_docs
from shared.sweep_utils import BestSweepChoice, is_better_sweep_choice
from ska_agent.core.pricing import PricingEngine


LOGGER = logging.getLogger(__name__)


_SWEEP_DATASET_NAME = ""
_SWEEP_SEGMENTS = None
_SWEEP_SEGMENT_TO_DOC_ID = None
_SWEEP_QUERY_EMBEDDINGS = None
_SWEEP_CORPUS_EMBS = None
_SWEEP_CORPUS_NORMS = None
_SWEEP_SEGMENT_CONFIG_DICT = None
_SWEEP_SEGMENT_STRATEGY = ""
_SWEEP_ENGINES = None


def _score_retrieval(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]):
    retrieved = set(retrieved_doc_ids)
    relevant = set(relevant_doc_ids)
    tp = len(retrieved & relevant)
    precision = 0.0 if not retrieved else tp / len(retrieved)
    recall = 0.0 if not relevant else tp / len(relevant)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _init_sweep_worker(
    dataset_name,
    segments,
    segment_to_doc_id,
    query_embedding_cache,
    corpus_embs,
    corpus_norms,
    segment_config_dict,
    segment_strategy,
    lambda_grid,
):
    global _SWEEP_DATASET_NAME
    global _SWEEP_SEGMENTS
    global _SWEEP_SEGMENT_TO_DOC_ID
    global _SWEEP_QUERY_EMBEDDINGS
    global _SWEEP_CORPUS_EMBS
    global _SWEEP_CORPUS_NORMS
    global _SWEEP_SEGMENT_CONFIG_DICT
    global _SWEEP_SEGMENT_STRATEGY
    global _SWEEP_ENGINES

    _SWEEP_DATASET_NAME = dataset_name
    _SWEEP_SEGMENTS = segments
    _SWEEP_SEGMENT_TO_DOC_ID = segment_to_doc_id
    _SWEEP_QUERY_EMBEDDINGS = query_embedding_cache
    _SWEEP_CORPUS_EMBS = corpus_embs
    _SWEEP_CORPUS_NORMS = corpus_norms
    _SWEEP_SEGMENT_CONFIG_DICT = segment_config_dict
    _SWEEP_SEGMENT_STRATEGY = segment_strategy

    def cached_embed_fn(text):
        return _SWEEP_QUERY_EMBEDDINGS[text]

    _SWEEP_ENGINES = {
        lam: PricingEngine(
            segments=_SWEEP_SEGMENTS,
            embed_fn=cached_embed_fn,
            lambda_sparsity=float(lam),
            eta_redundancy=ETA_REDUNDANCY,
            max_segments=MAX_SEGMENTS,
        )
        for lam in lambda_grid
    }


def _sweep_single_query(query_index: int, ex: QueryExample):
    per_query = []
    best = BestSweepChoice()

    for lam in sorted(_SWEEP_ENGINES.keys()):
        result = _SWEEP_ENGINES[lam].retrieve(ex.query, verbose=False)
        retrieved_doc_ids = [
            _SWEEP_SEGMENT_TO_DOC_ID[int(seg.start_idx)]
            for seg in result.segments
            if int(seg.start_idx) in _SWEEP_SEGMENT_TO_DOC_ID
        ]
        precision, recall, f1 = _score_retrieval(retrieved_doc_ids, ex.relevant_doc_ids)

        row = {
            "dataset": _SWEEP_DATASET_NAME,
            "question_id": ex.qid,
            "query": ex.query,
            "lambda": float(lam),
            "retrieval_precision": precision,
            "retrieval_recall": recall,
            "retrieval_f1": f1,
            "num_segments": int(len(result.segments)),
            "relevant_doc_ids": list(ex.relevant_doc_ids),
            "retrieved_doc_ids": retrieved_doc_ids,
            "segment_strategy": _SWEEP_SEGMENT_STRATEGY,
            "segment_config": _SWEEP_SEGMENT_CONFIG_DICT,
            "num_corpus_segments": int(len(_SWEEP_SEGMENTS)),
            "is_optimal": False,
        }
        per_query.append(row)

        k = len(result.segments)
        if is_better_sweep_choice(f1, k, float(lam), best):
            best = BestSweepChoice(
                score=f1,
                row_index=len(per_query) - 1,
                num_segments=k,
                lambda_value=float(lam),
            )

    per_query[best.row_index]["is_optimal"] = True
    query_emb = ensure_1d(_SWEEP_QUERY_EMBEDDINGS[ex.query]).astype(np.float32)
    per_query[best.row_index].update(
        build_training_feature_record(query_emb, _SWEEP_CORPUS_EMBS, _SWEEP_CORPUS_NORMS)
    )
    return query_index, per_query


def _sweep_query_chunk(chunk):
    return [_sweep_single_query(query_index, ex) for query_index, ex in chunk]


def _chunk_query_examples(query_examples: List[QueryExample], num_workers: int):
    chunk_size = max(1, min(64, (len(query_examples) + (num_workers * 8) - 1) // (num_workers * 8)))
    indexed = list(enumerate(query_examples))
    return [indexed[i : i + chunk_size] for i in range(0, len(indexed), chunk_size)]


def _sweep_records(
    dataset_name: str,
    segments,
    query_examples: List[QueryExample],
    segment_to_doc_id: Dict[int, str],
    embedder: MiniLMEmbedder,
    segment_config: SegmentBuildConfig,
    num_workers: int = 1,
):
    lambda_grid = list(LAMBDA_GRID)
    query_embedding_cache = {ex.query: ensure_1d(embedder.embed_single(ex.query)) for ex in query_examples}
    corpus_embs = np.asarray([seg.vector for seg in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    worker_count = max(1, int(num_workers))
    init_args = (
        dataset_name,
        segments,
        segment_to_doc_id,
        query_embedding_cache,
        corpus_embs,
        corpus_norms,
        segment_config.to_dict(),
        segment_config.strategy,
        lambda_grid,
    )
    if worker_count == 1 or len(query_examples) <= 1:
        _init_sweep_worker(*init_args)
        ordered = [_sweep_single_query(i, ex) for i, ex in enumerate(query_examples)]
        for processed in range(100, len(query_examples) + 1, 100):
            LOGGER.info("%s: processed %d/%d queries", dataset_name, processed, len(query_examples))
    else:
        chunks = _chunk_query_examples(query_examples, worker_count)
        ordered = []
        completed = 0
        LOGGER.info("%s: sweeping with %d workers over %d chunks", dataset_name, worker_count, len(chunks))
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_sweep_worker, initargs=init_args) as pool:
            futures = [pool.submit(_sweep_query_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                chunk_rows = future.result()
                ordered.extend(chunk_rows)
                completed += len(chunk_rows)
                LOGGER.info("%s: processed %d/%d queries", dataset_name, completed, len(query_examples))

    records = []
    for _, per_query in sorted(ordered, key=lambda item: item[0]):
        records.extend(per_query)
    return records


def _load_fineweb_queries(path: str, max_queries_for_sweep: int | None = None):
    passages = {}
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            passage_id = str(row["passage_id"])
            passage_text = str(row["passage_text"])
            passages[passage_id] = passage_text
            for i, q in enumerate(row.get("queries", [])):
                qid = f"{passage_id}_q{i}"
                examples.append(QueryExample(qid=qid, query=str(q), relevant_doc_ids=[passage_id]))
                if max_queries_for_sweep is not None and len(examples) >= max_queries_for_sweep:
                    doc_ids = list(passages.keys())
                    doc_texts = [passages[d] for d in doc_ids]
                    return doc_ids, doc_texts, examples
    doc_ids = list(passages.keys())
    doc_texts = [passages[d] for d in doc_ids]
    return doc_ids, doc_texts, examples


def _load_marco_data(data_path: str, n_passages: int, seed: int):
    pairs = load_marco_sample(data_path=data_path, n_passages=n_passages, seed=seed)
    doc_text_by_id = {}
    relevant_by_qid = {}
    query_text_by_qid = {}

    for row in pairs:
        doc_id = str(row["doc_id"])
        qid = str(row["qid"])
        doc_text_by_id[doc_id] = str(row["passage"])
        query_text_by_qid[qid] = str(row["query"])
        relevant_by_qid.setdefault(qid, set()).add(doc_id)

    examples = [
        QueryExample(qid=qid, query=query_text_by_qid[qid], relevant_doc_ids=sorted(list(doc_ids)))
        for qid, doc_ids in relevant_by_qid.items()
    ]
    doc_ids = list(doc_text_by_id.keys())
    doc_texts = [doc_text_by_id[d] for d in doc_ids]
    return doc_ids, doc_texts, examples


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-queries-path", type=str, default="outputs/fineweb_queries.jsonl")
    parser.add_argument("--marco-data-path", type=str, required=True)
    parser.add_argument("--marco-n-passages", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--max-queries-for-sweep",
        type=int,
        default=None,
        help="Hard cap on FineWeb query count used for lambda sweep labeling.",
    )
    parser.add_argument("--fineweb-output-path", type=str, default="outputs/fineweb_labeled.json")
    parser.add_argument("--marco-output-path", type=str, default="outputs/marco_labeled.json")
    parser.add_argument("--segment-strategy", type=str, default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    embedder = MiniLMEmbedder(args.embed_model)
    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )

    fw_doc_ids, fw_doc_texts, fw_examples = _load_fineweb_queries(
        args.fineweb_queries_path,
        max_queries_for_sweep=args.max_queries_for_sweep,
    )
    if args.max_queries_for_sweep is not None:
        LOGGER.info("fineweb sweep capped at %d queries", len(fw_examples))
    fw_segments, fw_segment_to_doc_id = build_segments_from_docs(
        fw_doc_ids,
        fw_doc_texts,
        embedder,
        config=segment_config,
    )
    fw_records = _sweep_records(
        "fineweb",
        fw_segments,
        fw_examples,
        fw_segment_to_doc_id,
        embedder,
        segment_config,
        num_workers=args.num_workers,
    )
    save_records(fw_records, args.fineweb_output_path)
    save_run_metadata(
        f"{args.fineweb_output_path}.metadata.json",
        args,
        extra={"dataset": "fineweb", "segment_config": segment_config.to_dict()},
    )

    mr_doc_ids, mr_doc_texts, mr_examples = _load_marco_data(
        data_path=args.marco_data_path,
        n_passages=args.marco_n_passages,
        seed=args.seed,
    )
    mr_segments, mr_segment_to_doc_id = build_segments_from_docs(
        mr_doc_ids,
        mr_doc_texts,
        embedder,
        config=segment_config,
    )
    mr_records = _sweep_records(
        "marco",
        mr_segments,
        mr_examples,
        mr_segment_to_doc_id,
        embedder,
        segment_config,
        num_workers=args.num_workers,
    )
    save_records(mr_records, args.marco_output_path)
    save_run_metadata(
        f"{args.marco_output_path}.metadata.json",
        args,
        extra={"dataset": "marco", "segment_config": segment_config.to_dict()},
    )


if __name__ == "__main__":
    main()
