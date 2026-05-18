"""BEIR-specific scoring and corpus-to-segment utilities."""

from beir.retrieval.evaluation import EvaluateRetrieval

from .segments import SegmentBuildConfig, build_segments_from_docs


def build_beir_segments(corpus, embedder, segment_config: SegmentBuildConfig | None = None):
    """
    Build Segment objects from BEIR corpus and map segment ids to doc ids.

    Returns:
      segments: list[Segment]
      segment_to_doc_id: dict[int, str]
    """
    texts = []
    doc_ids = []
    for doc_id, doc in corpus.items():
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        combined = f"{title} {text}".strip()
        if not combined:
            continue
        texts.append(combined)
        doc_ids.append(doc_id)

    return build_segments_from_docs(doc_ids, texts, embedder, config=segment_config)


def _to_beir_results(retrieved_segments, segment_to_doc_id, query_id):
    results = {query_id: {}}
    for rank, seg in enumerate(retrieved_segments):
        doc_id = segment_to_doc_id.get(int(seg.start_idx))
        if doc_id is None:
            continue
        score = 1.0 / (rank + 1)
        results[query_id][doc_id] = max(score, results[query_id].get(doc_id, 0.0))
    return results


def beir_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=10):
    """Score retrieval using BEIR's nDCG@k evaluator."""
    results = _to_beir_results(retrieved_segments, segment_to_doc_id, query_id)
    ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, results, k_values=[k])
    return float(ndcg[f"NDCG@{k}"])


def scifact_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=10):
    """Score SciFact retrieval using BEIR's nDCG@k evaluator."""
    return beir_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=k)


def fiqa_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=10):
    """Score FiQA retrieval using BEIR's nDCG@k evaluator (graded relevance)."""
    return beir_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=k)
