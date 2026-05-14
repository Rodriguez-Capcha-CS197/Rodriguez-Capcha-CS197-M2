"""Job 2: Build FineWeb+MS MARCO lambda-labeled training records."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from shared.constants import ETA_REDUNDANCY, LAMBDA_GRID, MAX_SEGMENTS, TIE_EPS
from shared.io_utils import save_records
from shared.marco_loader import load_marco_sample
from shared.scoring import ensure_1d
from ska_agent.core.pricing import PricingEngine
from ska_agent.core.structures import Segment


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


def _build_segments_from_docs(doc_ids: List[str], doc_texts: List[str], embedder: MiniLMEmbedder):
    vectors = embedder.embed(doc_texts)
    segments = []
    segment_to_doc_id = {}
    doc_to_segment_idx = {}
    for i, (doc_id, text, vec) in enumerate(zip(doc_ids, doc_texts, vectors)):
        seg = Segment(
            text=text,
            vector=np.asarray(vec, dtype=np.float64),
            start_idx=i,
            end_idx=i + 1,
            sentences=[text],
            internal_cost=0.0,
        )
        segments.append(seg)
        segment_to_doc_id[i] = doc_id
        doc_to_segment_idx[doc_id] = i
    return segments, segment_to_doc_id, doc_to_segment_idx


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


def _sweep_records(
    dataset_name: str,
    segments,
    query_examples: List[QueryExample],
    segment_to_doc_id: Dict[int, str],
    embedder: MiniLMEmbedder,
):
    lambda_grid = sorted(set(list(LAMBDA_GRID) + [1.0]))
    query_embedding_cache = {ex.query: ensure_1d(embedder.embed_single(ex.query)) for ex in query_examples}

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
    for idx, ex in enumerate(query_examples, start=1):
        per_query = []
        best_score = -float("inf")
        best_idx = 0
        best_k = None
        best_lam = None

        for lam in lambda_grid:
            result = engines[lam].retrieve(ex.query, verbose=False)
            retrieved_doc_ids = [
                segment_to_doc_id[int(seg.start_idx)]
                for seg in result.segments
                if int(seg.start_idx) in segment_to_doc_id
            ]
            precision, recall, f1 = _score_retrieval(retrieved_doc_ids, ex.relevant_doc_ids)

            row = {
                "dataset": dataset_name,
                "question_id": ex.qid,
                "query": ex.query,
                "lambda": float(lam),
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "retrieval_f1": f1,
                "num_segments": int(len(result.segments)),
                "relevant_doc_ids": list(ex.relevant_doc_ids),
                "retrieved_doc_ids": retrieved_doc_ids,
                "is_optimal": False,
            }
            per_query.append(row)

            k = len(result.segments)
            if f1 > best_score + TIE_EPS:
                best_score = f1
                best_idx = len(per_query) - 1
                best_k = k
                best_lam = lam
            elif abs(f1 - best_score) <= TIE_EPS:
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
            print(f"{dataset_name}: processed {idx}/{len(query_examples)} queries")
    return records


def _load_fineweb_queries(path: str):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-queries-path", type=str, default="outputs/fineweb_queries.jsonl")
    parser.add_argument("--marco-data-path", type=str, required=True)
    parser.add_argument("--marco-n-passages", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--fineweb-output-path", type=str, default="outputs/fineweb_labeled.json")
    parser.add_argument("--marco-output-path", type=str, default="outputs/marco_labeled.json")
    args = parser.parse_args()

    embedder = MiniLMEmbedder(args.embed_model)

    fw_doc_ids, fw_doc_texts, fw_examples = _load_fineweb_queries(args.fineweb_queries_path)
    fw_segments, fw_segment_to_doc_id, _ = _build_segments_from_docs(fw_doc_ids, fw_doc_texts, embedder)
    fw_records = _sweep_records("fineweb", fw_segments, fw_examples, fw_segment_to_doc_id, embedder)
    save_records(fw_records, args.fineweb_output_path)

    mr_doc_ids, mr_doc_texts, mr_examples = _load_marco_data(
        data_path=args.marco_data_path,
        n_passages=args.marco_n_passages,
        seed=args.seed,
    )
    mr_segments, mr_segment_to_doc_id, _ = _build_segments_from_docs(mr_doc_ids, mr_doc_texts, embedder)
    mr_records = _sweep_records("marco", mr_segments, mr_examples, mr_segment_to_doc_id, embedder)
    save_records(mr_records, args.marco_output_path)


if __name__ == "__main__":
    main()
