"""Job 2: Build FiQA segments and run lambda sweep for oracle records."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

from shared.beir_scoring import build_beir_segments, fiqa_ndcg
from shared.constants import ETA_REDUNDANCY, LAMBDA_GRID, MAX_SEGMENTS, TIE_EPS
from shared.io_utils import save_records
from shared.scoring import ensure_1d
from ska_agent.core.pricing import PricingEngine


@dataclass
class QueryExample:
    qid: str
    query: str
    relevant_doc_ids: Sequence[str]


class MiniLMEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, sentences):
        return self.model.encode(sentences, convert_to_numpy=True)

    def embed_single(self, text):
        return self.embed([text])[0]


def _download_fiqa(data_dir: str) -> str:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="outputs/beir_data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-path", type=str, default="outputs/fiqa_sweep_records.json")
    args = parser.parse_args()

    data_path = _download_fiqa(args.data_dir) if args.download else f"{args.data_dir}/fiqa"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

    embedder = MiniLMEmbedder(args.embed_model)
    segments, segment_to_doc_id = build_beir_segments(corpus, embedder)
    examples = _prepare_query_examples(queries, qrels)
    lambda_grid = sorted(set(list(LAMBDA_GRID) + [1.0]))

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
        best_score = -float("inf")
        best_idx = 0
        best_k = None
        best_lam = None

        for lam in lambda_grid:
            result = engines[lam].retrieve(ex.query, verbose=False)
            ndcg = fiqa_ndcg(
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
                "is_optimal": False,
            }
            per_query.append(row)

            k = len(result.segments)
            if ndcg > best_score + TIE_EPS:
                best_score = ndcg
                best_idx = len(per_query) - 1
                best_k = k
                best_lam = lam
            elif abs(ndcg - best_score) <= TIE_EPS:
                if best_k is None or k < best_k:
                    best_idx = len(per_query) - 1
                    best_k = k
                    best_lam = lam
                elif k == best_k and best_lam is not None and lam < best_lam:
                    best_idx = len(per_query) - 1
                    best_k = k
                    best_lam = lam

        per_query[best_idx]["is_optimal"] = True
        records.extend(per_query)

        if idx % 100 == 0:
            print(f"processed {idx}/{len(examples)} queries")

    save_records(records, args.output_path)


if __name__ == "__main__":
    main()
