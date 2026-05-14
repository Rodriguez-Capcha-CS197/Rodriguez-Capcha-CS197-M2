"""Inference wrappers for plain, hybrid, and covariance lambda predictors."""

import numpy as np
import torch

from .constants import EPS
from .scoring import ensure_1d


def _as_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _predict_log_lambda(model, features):
    model.eval()
    x = torch.tensor(_as_2d(features), dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).reshape(-1).cpu().numpy()
    return pred


def extract_density_features(query_emb, corpus_embs, corpus_norms):
    """Simple query-to-corpus geometry summary features."""
    q = ensure_1d(query_emb).astype(np.float32)
    c = _as_2d(corpus_embs).astype(np.float32)
    cn = ensure_1d(corpus_norms).astype(np.float32)

    qn = float(np.linalg.norm(q))
    denom = np.maximum(cn * max(qn, EPS), EPS)
    sims = (c @ q) / denom

    topk = min(10, len(sims))
    if topk == 0:
        return np.zeros(4, dtype=np.float32)
    top = np.sort(sims)[-topk:]
    return np.array(
        [
            float(np.mean(top)),
            float(np.std(top)),
            float(np.max(top)),
            float(np.percentile(top, 90)),
        ],
        dtype=np.float32,
    )


def extract_covariance_features(query_emb, corpus_embs, corpus_norms):
    """
    Redundancy-oriented geometry summary features.

    Uses top-k similarity spread and pairwise cosine among top candidates.
    """
    q = ensure_1d(query_emb).astype(np.float32)
    c = _as_2d(corpus_embs).astype(np.float32)
    cn = ensure_1d(corpus_norms).astype(np.float32)

    qn = float(np.linalg.norm(q))
    denom = np.maximum(cn * max(qn, EPS), EPS)
    sims = (c @ q) / denom

    topk = min(8, len(sims))
    if topk <= 1:
        return np.zeros(4, dtype=np.float32)

    idx = np.argpartition(sims, -topk)[-topk:]
    top_vecs = c[idx]
    top_norms = np.linalg.norm(top_vecs, axis=1, keepdims=True)
    top_vecs = top_vecs / np.maximum(top_norms, EPS)
    gram = top_vecs @ top_vecs.T
    iu = np.triu_indices(topk, k=1)
    pairwise = gram[iu]

    top_sims = sims[idx]
    return np.array(
        [
            float(np.mean(top_sims)),
            float(np.std(top_sims)),
            float(np.mean(pairwise)),
            float(np.max(pairwise)),
        ],
        dtype=np.float32,
    )


def predict_plain_lambda(model, query_text, embed_fn):
    """Predict lambda from query embedding only."""
    query_emb = ensure_1d(embed_fn(query_text)).astype(np.float32)
    log_lam = _predict_log_lambda(model, query_emb)[0]
    return float(np.exp(log_lam))


def predict_hybrid_lambda(model, query_text, embed_fn, corpus_embs, corpus_norms):
    """Predict lambda from query embedding + density geometry features."""
    query_emb = ensure_1d(embed_fn(query_text)).astype(np.float32)
    geom = extract_density_features(query_emb, corpus_embs, corpus_norms)
    features = np.concatenate([query_emb, geom], axis=0)
    log_lam = _predict_log_lambda(model, features)[0]
    return float(np.exp(log_lam))


def predict_covariance_lambda(model, query_text, embed_fn, corpus_embs, corpus_norms):
    """Predict lambda from query embedding + redundancy geometry features."""
    query_emb = ensure_1d(embed_fn(query_text)).astype(np.float32)
    geom = extract_covariance_features(query_emb, corpus_embs, corpus_norms)
    features = np.concatenate([query_emb, geom], axis=0)
    log_lam = _predict_log_lambda(model, features)[0]
    return float(np.exp(log_lam))
