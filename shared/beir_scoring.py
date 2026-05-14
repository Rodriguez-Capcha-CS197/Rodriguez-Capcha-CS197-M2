"""BEIR-specific scoring and corpus-to-segment utilities."""

from beir.retrieval.evaluation import EvaluateRetrieval

from .corpus import build_segments_from_texts


def build_beir_segments(corpus, embedder):
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

    segments = build_segments_from_texts(texts, embedder)
    segment_to_doc_id = {}
    for i, seg in enumerate(segments):
        segment_to_doc_id[int(seg.start_idx)] = doc_ids[i]
    return segments, segment_to_doc_id


def _to_beir_results(retrieved_segments, segment_to_doc_id, query_id):
    results = {query_id: {}}
    for rank, seg in enumerate(retrieved_segments):
        doc_id = segment_to_doc_id.get(int(seg.start_idx))
        if doc_id is None:
            continue
        results[query_id][doc_id] = 1.0 / (rank + 1)
    return results


def scifact_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=10):
    """Score SciFact retrieval using BEIR's nDCG@k evaluator."""
    results = _to_beir_results(retrieved_segments, segment_to_doc_id, query_id)
    ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, results, k_values=[k])
    return float(ndcg[f"NDCG@{k}"])


def fiqa_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=10):
    """Score FiQA retrieval using BEIR's nDCG@k evaluator (graded relevance)."""
    return scifact_ndcg(retrieved_segments, query_id, qrels, segment_to_doc_id, k=k)
