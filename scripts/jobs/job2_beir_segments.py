"""Job 2: Build BEIR segments and run lambda sweeps for any supported BEIR dataset."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from configs.metadata import save_run_metadata
from shared.beir_data import resolve_beir_dataset_path
from shared.beir_scoring import beir_ndcg, build_beir_segments
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

BEIR_DATASET_URLS = {
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "climate-fever": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip",
    "dbpedia-entity": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "trec-covid": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
}

_SWEEP_DATASET_NAME = ""
_SWEEP_SEGMENTS = None
_SWEEP_SEGMENT_TO_DOC_ID = None
_SWEEP_QRELS = None
_SWEEP_QUERY_EMBEDDINGS = None
_SWEEP_CORPUS_EMBS = None
_SWEEP_CORPUS_NORMS = None
_SWEEP_SEGMENT_CONFIG_DICT = None
_SWEEP_SEGMENT_STRATEGY = ""
_SWEEP_ENGINES = None


def _download_beir_dataset(dataset_name: str, data_dir: str) -> str:
    if dataset_name not in BEIR_DATASET_URLS:
        supported = ", ".join(sorted(BEIR_DATASET_URLS))
        raise ValueError(f"Unsupported BEIR dataset {dataset_name!r}. Supported: {supported}")
    return util.download_and_unzip(BEIR_DATASET_URLS[dataset_name], data_dir)


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


def _init_sweep_worker(
    dataset_name,
    segments,
    segment_to_doc_id,
    qrels,
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
    global _SWEEP_QRELS
    global _SWEEP_QUERY_EMBEDDINGS
    global _SWEEP_CORPUS_EMBS
    global _SWEEP_CORPUS_NORMS
    global _SWEEP_SEGMENT_CONFIG_DICT
    global _SWEEP_SEGMENT_STRATEGY
    global _SWEEP_ENGINES

    _SWEEP_DATASET_NAME = dataset_name
    _SWEEP_SEGMENTS = segments
    _SWEEP_SEGMENT_TO_DOC_ID = segment_to_doc_id
    _SWEEP_QRELS = qrels
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
        ndcg = beir_ndcg(
            retrieved_segments=result.segments,
            query_id=ex.qid,
            qrels=_SWEEP_QRELS,
            segment_to_doc_id=_SWEEP_SEGMENT_TO_DOC_ID,
            k=10,
        )
        row = {
            "dataset": _SWEEP_DATASET_NAME,
            "question_id": ex.qid,
            "query": ex.query,
            "lambda": float(lam),
            "ndcg_at_10": float(ndcg),
            "num_segments": int(len(result.segments)),
            "relevant_doc_ids": list(ex.relevant_doc_ids),
            "retrieved_doc_ids": _retrieved_doc_ids(result, _SWEEP_SEGMENT_TO_DOC_ID),
            "segment_strategy": _SWEEP_SEGMENT_STRATEGY,
            "segment_config": _SWEEP_SEGMENT_CONFIG_DICT,
            "num_corpus_segments": int(len(_SWEEP_SEGMENTS)),
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
    qrels: Dict[str, Dict[str, int]],
    embedder: MiniLMEmbedder,
    segment_config: SegmentBuildConfig,
    num_workers: int,
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
        qrels,
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


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(BEIR_DATASET_URLS))
    parser.add_argument("--data-dir", type=str, default="outputs/beir_data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--segment-strategy", type=str, default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    output_path = args.output_path or f"outputs/{args.dataset}_sweep_records.json"
    data_path = resolve_beir_dataset_path(
        args.data_dir,
        args.dataset,
        args.split,
        allow_download=args.download,
        downloader=lambda data_dir: _download_beir_dataset(args.dataset, data_dir),
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
    examples = _prepare_query_examples(queries, qrels)
    if args.max_queries and args.max_queries > 0:
        examples = examples[: args.max_queries]
        LOGGER.info("%s sweep capped at %d queries", args.dataset, len(examples))
    records = _sweep_records(
        args.dataset,
        segments,
        examples,
        segment_to_doc_id,
        qrels,
        embedder,
        segment_config,
        num_workers=args.num_workers,
    )
    save_records(records, output_path)
    save_run_metadata(
        f"{output_path}.metadata.json",
        args,
        extra={"dataset": args.dataset, "segment_config": segment_config.to_dict()},
    )


if __name__ == "__main__":
    main()
