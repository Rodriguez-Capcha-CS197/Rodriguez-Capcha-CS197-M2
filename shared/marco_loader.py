"""MS MARCO loading helpers using BEIR data format."""

import numpy as np
from beir.datasets.data_loader import GenericDataLoader


def load_marco_sample(data_path, n_passages=15000, seed=0):
    """
    Return a list of query-passage pairs sampled via MS MARCO qrels.

    Output rows:
      {"query": str, "passage": str, "doc_id": str, "qid": str}
    """
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

    doc_to_qids = {}
    for qid, doc_scores in qrels.items():
        for doc_id in doc_scores:
            doc_to_qids.setdefault(doc_id, []).append(qid)

    rng = np.random.default_rng(seed)
    doc_ids = list(corpus.keys())
    sample_size = min(n_passages, len(doc_ids))
    sampled_doc_ids = rng.choice(doc_ids, size=sample_size, replace=False)

    pairs = []
    for doc_id in sampled_doc_ids:
        doc = corpus[doc_id]
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        passage = f"{title} {text}".strip()
        if not passage:
            continue
        for qid in doc_to_qids.get(doc_id, []):
            if qid in queries:
                pairs.append(
                    {
                        "query": queries[qid],
                        "passage": passage,
                        "doc_id": doc_id,
                        "qid": qid,
                    }
                )
    return pairs
