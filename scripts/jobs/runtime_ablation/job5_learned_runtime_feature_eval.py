"""Learned runtime-feature ablation across five BEIR domains.

The learned runtime model uses only candidate-side score features from the
preliminary top-50 list. It predicts log(lambda), then evaluates by selecting
the nearest saved sweep row for that query.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from beir.datasets.data_loader import GenericDataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.metadata import save_run_metadata
from shared.beir_data import resolve_beir_dataset_path
from shared.constants import EPS, MAX_SEGMENTS
from shared.dataset_utils import merge_and_shuffle_datasets
from shared.embedding import MiniLMEmbedder
from shared.lambda_inference import build_covariance_features
from shared.predictor import LambdaPredictor
from shared.segments import SegmentBuildConfig
from shared.beir_scoring import beir_ndcg, build_beir_segments
from shared.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)

DEFAULT_DOMAINS = ["scifact", "fiqa", "nfcorpus", "arguana", "trec-covid"]
METHODS = [
    "fixed",
    "covariance",
    "score_gap",
    "runtime_feature_ridge",
    "ndcg_oracle",
    "precision_oracle",
]
PRIMARY_METRICS = ["ndcg_at_10", "precision_returned", "num_returned_docs"]


class RuntimeRidgeRegressor:
    """Small ridge regressor for standardized runtime score features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y_log_lambda: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_log_lambda, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("RuntimeRidgeRegressor requires a non-empty 2D feature matrix.")

        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.x_std = np.where(self.x_std < EPS, 1.0, self.x_std)
        Xs = (X - self.x_mean) / self.x_std
        self.y_mean = float(np.mean(y))
        ys = y - self.y_mean

        xtx = Xs.T @ Xs
        penalty = self.alpha * np.eye(xtx.shape[0], dtype=np.float64)
        self.weights = np.linalg.solve(xtx + penalty, Xs.T @ ys)

    def predict_log_lambda(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None or self.weights is None:
            raise RuntimeError("RuntimeRidgeRegressor must be fit before prediction.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = (X - self.x_mean) / self.x_std
        return Xs @ self.weights + self.y_mean


def _train_covariance_model(X, y, seed, hidden_dim=64, epochs=200, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = max(1, int(0.8 * len(idx)))
    train_idx = idx[:split]
    val_idx = idx[split:] if split < len(idx) else idx[:1]

    model = LambdaPredictor(input_dim=X.shape[1], hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(np.log(np.clip(y[train_idx], EPS, None)), dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(np.log(np.clip(y[val_idx], EPS, None)), dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = criterion(model(X_val), y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def score_gap_cutoff_indices(
    scores: np.ndarray,
    min_k: int = 1,
    max_k: int = MAX_SEGMENTS,
    candidate_k: int = 50,
) -> list[int]:
    """Return top-k indices where k is chosen by the largest adjacent score gap."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return []

    min_k = max(1, int(min_k))
    max_k = max(min_k, min(int(max_k), int(scores.size)))
    candidate_k = max(max_k, min(int(candidate_k), int(scores.size)))

    order = np.argsort(-scores)[:candidate_k]
    if max_k <= min_k or len(order) <= min_k:
        return [int(idx) for idx in order[:min_k]]

    sorted_scores = scores[order]
    gap_limit = min(max_k, len(sorted_scores) - 1)
    if gap_limit < min_k:
        return [int(idx) for idx in order[:min_k]]

    gaps = sorted_scores[:gap_limit] - sorted_scores[1 : gap_limit + 1]
    k_star = max(min_k, min(max_k, int(np.argmax(gaps)) + 1))
    return [int(idx) for idx in order[:k_star]]


def _score_features(scores: np.ndarray, candidate_k: int = 50) -> np.ndarray:
    """Candidate-side score features from a preliminary top-k list."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.zeros(42, dtype=np.float32)

    k = min(int(candidate_k), int(scores.size))
    top = np.sort(scores)[::-1][:k]
    if len(top) < candidate_k:
        top = np.pad(top, (0, candidate_k - len(top)), constant_values=float(top[-1]))

    top10 = top[:10]
    gaps = top[:-1] - top[1:]
    top10_gaps = gaps[:10] if len(gaps) >= 10 else np.pad(gaps, (0, 10 - len(gaps)))

    shifted = top - float(np.max(top))
    exp_scores = np.exp(shifted)
    probs = exp_scores / max(float(np.sum(exp_scores)), EPS)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, EPS))))

    x = np.arange(len(top), dtype=np.float32)
    x_centered = x - float(np.mean(x))
    y_centered = top - float(np.mean(top))
    denom = max(float(np.sum(x_centered ** 2)), EPS)
    slope = float(np.sum(x_centered * y_centered) / denom)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    counts = [float(np.sum(top >= threshold)) for threshold in thresholds]

    features = np.asarray(
        list(top10)
        + list(top10_gaps)
        + [
            float(np.mean(top)),
            float(np.std(top)),
            float(np.min(top)),
            float(np.max(top)),
            float(np.median(top)),
            float(np.percentile(top, 75)),
            float(np.percentile(top, 25)),
            float(top[0] - top[min(4, len(top) - 1)]),
            float(top[0] - top[min(9, len(top) - 1)]),
            float(top[0] - top[-1]),
            float(np.max(gaps)) if len(gaps) else 0.0,
            float(np.argmax(gaps) + 1) if len(gaps) else 1.0,
            float(np.mean(gaps)) if len(gaps) else 0.0,
            float(np.std(gaps)) if len(gaps) else 0.0,
            entropy,
            slope,
        ]
        + counts,
        dtype=np.float32,
    )
    return features


def _read_json_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else [data]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _group_by_query(records: list[dict]) -> dict[str, list[dict]]:
    rows_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        rows_by_qid[str(row["question_id"])].append(row)
    return dict(rows_by_qid)


def _dedupe_doc_ids(doc_ids) -> list[str]:
    return list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))


def _metrics_from_doc_ids(ndcg_at_10: float, retrieved_doc_ids, relevant_doc_ids) -> dict:
    retrieved_doc_ids = _dedupe_doc_ids(retrieved_doc_ids)
    relevant_doc_ids = {str(doc_id) for doc_id in relevant_doc_ids}
    hit_count = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_doc_ids)
    precision = hit_count / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    return {
        "ndcg_at_10": float(ndcg_at_10),
        "precision_returned": float(precision),
        "num_returned_docs": float(len(retrieved_doc_ids)),
        "hit_count": int(hit_count),
        "retrieved_doc_ids": retrieved_doc_ids,
        "relevant_doc_ids": sorted(relevant_doc_ids),
    }


def _metrics_from_sweep_row(row: dict) -> dict:
    return _metrics_from_doc_ids(
        ndcg_at_10=float(row["ndcg_at_10"]),
        retrieved_doc_ids=row.get("retrieved_doc_ids", []),
        relevant_doc_ids=row.get("relevant_doc_ids", []),
    )


def _nearest_lambda_row(rows: list[dict], lam: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["lambda"]) - float(lam)))


def _best_lambda_target(rows: list[dict]) -> float:
    optimal = [row for row in rows if row.get("is_optimal")]
    if optimal:
        return float(optimal[0]["lambda"])
    best = max(rows, key=lambda row: (float(row["ndcg_at_10"]), -int(row.get("num_segments", 0))))
    return float(best["lambda"])


def _precision_score(row: dict) -> tuple[float, float, float]:
    metrics = _metrics_from_sweep_row(row)
    return (
        float(metrics["precision_returned"]),
        float(row["ndcg_at_10"]),
        -int(row.get("num_segments", metrics["num_returned_docs"])),
    )


def _fixed_lambda_from_training_records(fineweb_records: str, marco_records: str) -> float:
    lambdas = []
    for path in [fineweb_records, marco_records]:
        records = _read_json_rows(path)
        lambdas.extend(float(row["lambda"]) for row in records if row.get("is_optimal"))
    if not lambdas:
        raise ValueError("Could not infer fixed lambda. Pass --fixed-lambda explicitly.")
    return float(np.median(np.asarray(lambdas, dtype=np.float32)))


def _collect_covariance_training_features(records: list[dict]):
    optimal = [row for row in records if row.get("is_optimal")]
    missing = [
        row["question_id"]
        for row in optimal
        if "covariance_features" not in row
    ]
    if missing:
        preview = ", ".join(str(qid) for qid in missing[:5])
        raise ValueError(f"Training records missing covariance_features. Example qids: {preview}")
    X = np.asarray([row["covariance_features"] for row in optimal], dtype=np.float32)
    y = np.asarray([float(row["lambda"]) for row in optimal], dtype=np.float32)
    return X, y


def _assert_segment_config(records: list[dict], expected_config: SegmentBuildConfig, records_path: str) -> None:
    strategies = {row.get("segment_strategy") for row in records if row.get("segment_strategy")}
    if strategies and strategies != {expected_config.strategy}:
        raise ValueError(
            f"{records_path} was generated with segment strategies {sorted(strategies)}, "
            f"but this run requested {expected_config.strategy!r}."
        )

    configs = [row.get("segment_config") for row in records if row.get("segment_config")]
    if not configs:
        LOGGER.warning("%s has no segment_config metadata; assuming current config", records_path)
        return

    expected = expected_config.to_dict()
    mismatches = []
    for key, expected_value in expected.items():
        seen = {cfg.get(key) for cfg in configs}
        if seen != {expected_value}:
            mismatches.append(f"{key}: saw {sorted(seen)!r}, expected {expected_value!r}")
    if mismatches:
        raise ValueError(f"{records_path} segment_config mismatch: " + "; ".join(mismatches))


def _policy_row_from_sweep(qid: str, query: str, method: str, selected_row: dict, chosen_lambda: float | None) -> dict:
    return {
        "question_id": qid,
        "query": query,
        "method": method,
        "chosen_lambda": None if chosen_lambda is None else float(chosen_lambda),
        **_metrics_from_sweep_row(selected_row),
    }


def _build_domain_bundle(
    domain: dict,
    embedder: MiniLMEmbedder,
    segment_config: SegmentBuildConfig,
    candidate_k: int,
    max_segments: int,
    max_queries: int,
):
    records = _read_json_rows(domain["records_path"])
    _assert_segment_config(records, segment_config, domain["records_path"])
    records_by_query = _group_by_query(records)
    if max_queries and max_queries > 0:
        records_by_query = dict(list(records_by_query.items())[:max_queries])

    data_path = resolve_beir_dataset_path(domain["data_path"], domain["name"], domain["split"])
    corpus, _, qrels = GenericDataLoader(data_path).load(split=domain["split"])
    segments, segment_to_doc_id = build_beir_segments(corpus, embedder, segment_config=segment_config)
    corpus_embs = np.asarray([segment.vector for segment in segments], dtype=np.float32)
    corpus_norms = np.linalg.norm(corpus_embs, axis=1)

    qids = list(records_by_query.keys())
    query_texts = [records_by_query[qid][0]["query"] for qid in qids]
    query_embs = np.asarray(embedder.embed(query_texts), dtype=np.float32)

    runtime_features = []
    target_lambdas = []
    score_gap_rows = []
    for qid, query, query_emb in zip(qids, query_texts, query_embs):
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query_emb))
        denom = np.maximum(corpus_norms * max(query_norm, EPS), EPS)
        scores = (corpus_embs @ query_emb) / denom
        runtime_features.append(_score_features(scores, candidate_k=candidate_k))
        target_lambdas.append(_best_lambda_target(records_by_query[qid]))

        selected_indices = score_gap_cutoff_indices(
            scores,
            min_k=1,
            max_k=max_segments,
            candidate_k=candidate_k,
        )
        selected_segments = [segments[idx] for idx in selected_indices]
        retrieved_doc_ids = [
            segment_to_doc_id[int(segment.start_idx)]
            for segment in selected_segments
            if int(segment.start_idx) in segment_to_doc_id
        ]
        score_gap_rows.append(
            {
                "question_id": qid,
                "query": query,
                "method": "score_gap",
                "chosen_lambda": None,
                "runtime_score_gap_k": int(len(selected_segments)),
                "runtime_candidate_k": int(min(candidate_k, len(scores))),
                **_metrics_from_doc_ids(
                    ndcg_at_10=beir_ndcg(
                        retrieved_segments=selected_segments,
                        query_id=qid,
                        qrels=qrels,
                        segment_to_doc_id=segment_to_doc_id,
                        k=10,
                    ),
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=records_by_query[qid][0].get("relevant_doc_ids", []),
                ),
            }
        )

    return {
        "domain": domain,
        "records_by_query": records_by_query,
        "segments": segments,
        "corpus_embs": corpus_embs,
        "corpus_norms": corpus_norms,
        "query_embs_by_qid": {qid: emb for qid, emb in zip(qids, query_embs)},
        "runtime_features_by_qid": {
            qid: feature for qid, feature in zip(qids, np.asarray(runtime_features, dtype=np.float32))
        },
        "target_lambda_by_qid": {qid: float(lam) for qid, lam in zip(qids, target_lambdas)},
        "score_gap_rows": score_gap_rows,
        "num_corpus_segments": int(len(segments)),
    }


def _evaluate_fixed(records_by_query: dict[str, list[dict]], fixed_lambda: float) -> list[dict]:
    rows = []
    for qid, sweep_rows in records_by_query.items():
        selected = _nearest_lambda_row(sweep_rows, fixed_lambda)
        rows.append(_policy_row_from_sweep(qid, selected["query"], "fixed", selected, float(selected["lambda"])))
    return rows


def _evaluate_oracles(records_by_query: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    ndcg_rows = []
    precision_rows = []
    for qid, sweep_rows in records_by_query.items():
        query = sweep_rows[0]["query"]
        # nDCG ties are broken toward fewer admitted segments to avoid treating
        # equally ranked but more permissive retrieval as oracle-superior.
        ndcg_best = max(sweep_rows, key=lambda row: (float(row["ndcg_at_10"]), -int(row.get("num_segments", 0))))
        precision_best = max(sweep_rows, key=_precision_score)
        ndcg_rows.append(_policy_row_from_sweep(qid, query, "ndcg_oracle", ndcg_best, float(ndcg_best["lambda"])))
        precision_rows.append(
            _policy_row_from_sweep(qid, query, "precision_oracle", precision_best, float(precision_best["lambda"]))
        )
    return ndcg_rows, precision_rows


def _evaluate_covariance(bundle: dict, model: LambdaPredictor) -> list[dict]:
    rows = []
    corpus_embs = bundle["corpus_embs"]
    corpus_norms = bundle["corpus_norms"]
    for qid, sweep_rows in bundle["records_by_query"].items():
        query_emb = bundle["query_embs_by_qid"][qid]
        features = build_covariance_features(query_emb, corpus_embs, corpus_norms)
        with torch.no_grad():
            log_lam = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).item()
        lam = float(np.exp(log_lam))
        selected = _nearest_lambda_row(sweep_rows, lam)
        rows.append(_policy_row_from_sweep(qid, sweep_rows[0]["query"], "covariance", selected, float(selected["lambda"])))
    return rows


def _fit_runtime_model_for_domain(domain_name: str, bundles: dict[str, dict], alpha: float, train_mode: str):
    X_train = []
    y_train = []
    for other_name, bundle in bundles.items():
        if train_mode == "leave_domain_out" and other_name == domain_name:
            continue
        for qid, features in bundle["runtime_features_by_qid"].items():
            X_train.append(features)
            y_train.append(np.log(max(bundle["target_lambda_by_qid"][qid], EPS)))
    if not X_train:
        raise ValueError(f"No runtime training features available for {domain_name}.")
    model = RuntimeRidgeRegressor(alpha=alpha)
    model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32))
    return model, len(X_train)


def _evaluate_runtime_feature_model(bundle: dict, model: RuntimeRidgeRegressor) -> list[dict]:
    rows = []
    qids = list(bundle["records_by_query"].keys())
    X = np.asarray([bundle["runtime_features_by_qid"][qid] for qid in qids], dtype=np.float32)
    lambdas = np.exp(model.predict_log_lambda(X))
    for qid, lam in zip(qids, lambdas):
        sweep_rows = bundle["records_by_query"][qid]
        selected = _nearest_lambda_row(sweep_rows, float(lam))
        rows.append(
            _policy_row_from_sweep(
                qid,
                sweep_rows[0]["query"],
                "runtime_feature_ridge",
                selected,
                float(selected["lambda"]),
            )
        )
    return rows


def _aggregate(rows: list[dict]) -> dict:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows])) if rows else 0.0
        for metric in PRIMARY_METRICS
    }


def _bootstrap_mean_ci(values, n_samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if n_samples <= 0 or len(values) == 1:
        mean = float(np.mean(values))
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_samples, len(values)))
    means = np.mean(values[indices], axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _paired_bootstrap_diff_ci(values_a, values_b, n_samples: int, seed: int) -> dict:
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)
    diffs = values_a - values_b
    if len(diffs) == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_two_sided": 1.0}
    if n_samples <= 0 or len(diffs) == 1:
        mean = float(np.mean(diffs))
        return {"mean_diff": mean, "ci_low": mean, "ci_high": mean, "p_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(n_samples, len(diffs)))
    means = np.mean(diffs[indices], axis=1)
    p_two_sided = 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0)))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_two_sided": min(1.0, p_two_sided),
    }


def _bootstrap_summary(domain: str, rows_by_method: dict[str, list[dict]], n_samples: int, seed: int) -> dict:
    by_method_qid = {
        method: {row["question_id"]: row for row in rows}
        for method, rows in rows_by_method.items()
    }
    fixed_by_qid = by_method_qid["fixed"]
    common_qids = sorted(fixed_by_qid)

    summary = {"domain": domain, "n_queries": len(common_qids)}
    for method in METHODS:
        summary[method] = {}
        for offset, metric in enumerate(PRIMARY_METRICS):
            values = [float(by_method_qid[method][qid][metric]) for qid in common_qids if qid in by_method_qid[method]]
            summary[method][metric] = _bootstrap_mean_ci(values, n_samples, seed + offset)
            if method != "fixed":
                paired_qids = [qid for qid in common_qids if qid in by_method_qid[method]]
                method_values = [float(by_method_qid[method][qid][metric]) for qid in paired_qids]
                fixed_values = [float(fixed_by_qid[qid][metric]) for qid in paired_qids]
                summary.setdefault(f"{method}_diff_vs_fixed", {})[metric] = _paired_bootstrap_diff_ci(
                    method_values,
                    fixed_values,
                    n_samples,
                    seed + 100 + offset,
                )
    return summary


def _format_ci(item: dict) -> str:
    return f"{item['mean']:.3f} [{item['ci_low']:.3f}, {item['ci_high']:.3f}]"


def _format_delta(item: dict) -> str:
    return (
        f"{item['mean_diff']:+.3f} "
        f"[{item['ci_low']:+.3f}, {item['ci_high']:+.3f}], "
        f"p={item['p_two_sided']:.3f}"
    )


def _paper_table_rows(bootstrap_summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in bootstrap_summaries:
        for method in METHODS:
            row = {
                "domain": summary["domain"],
                "method": method,
                "n_queries": summary["n_queries"],
                "nDCG@10 95% CI": _format_ci(summary[method]["ndcg_at_10"]),
                "precision@returned 95% CI": _format_ci(summary[method]["precision_returned"]),
                "Avg returned 95% CI": _format_ci(summary[method]["num_returned_docs"]),
            }
            if method == "fixed":
                row["Delta nDCG vs fixed"] = "--"
                row["Delta precision vs fixed"] = "--"
                row["Delta returned vs fixed"] = "--"
            else:
                deltas = summary[f"{method}_diff_vs_fixed"]
                row["Delta nDCG vs fixed"] = _format_delta(deltas["ndcg_at_10"])
                row["Delta precision vs fixed"] = _format_delta(deltas["precision_returned"])
                row["Delta returned vs fixed"] = _format_delta(deltas["num_returned_docs"])
            rows.append(row)
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for raw, escaped in replacements.items():
        text = text.replace(raw, escaped)
    return text


def _write_latex_table(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(_latex_escape(col) for col in columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(_latex_escape(row[col]) for col in columns) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def _plot_metric_bars(run_rows: list[dict], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        with open(os.path.join(output_dir, "plots_skipped.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"matplotlib import failed: {exc}\n")
        return

    metric_specs = [
        ("ndcg_at_10", "nDCG@10", "learned_runtime_ndcg_by_domain"),
        ("precision_returned", "precision@returned", "learned_runtime_precision_by_domain"),
        ("num_returned_docs", "Average returned", "learned_runtime_avg_returned_by_domain"),
    ]
    domains = [row["domain"] for row in run_rows]
    x = np.arange(len(domains))
    width = 0.13

    for metric_key, ylabel, filename in metric_specs:
        plt.figure(figsize=(11, 4.5))
        for offset, method in enumerate(METHODS):
            values = [float(row[f"{method}_{metric_key}"]) for row in run_rows]
            positions = x + (offset - (len(METHODS) - 1) / 2.0) * width
            plt.bar(positions, values, width=width, label=method)
        plt.xticks(x, domains, rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=180)
        plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
        plt.close()


def _domain_records_path(records_dir: str, domain: str) -> str:
    return os.path.join(records_dir, f"{domain}_sweep_records.json")


def _parse_domains(args) -> list[dict]:
    domains = []
    for raw in [item.strip() for item in args.domains.split(",") if item.strip()]:
        parts = raw.split(":")
        name = parts[0]
        records_path = parts[1] if len(parts) > 1 and parts[1] else _domain_records_path(args.records_dir, name)
        data_path = parts[2] if len(parts) > 2 and parts[2] else os.path.join(args.beir_data_dir, name)
        split = parts[3] if len(parts) > 3 and parts[3] else args.split
        domains.append({"name": name, "records_path": records_path, "data_path": data_path, "split": split})
    return domains


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--records-dir", default="outputs")
    parser.add_argument("--beir-data-dir", default="outputs/beir_data")
    parser.add_argument("--fineweb-records", default="outputs/fineweb_labeled.json")
    parser.add_argument("--marco-records", default="outputs/marco_labeled.json")
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="outputs/runtime_ablation/learned_runtime_feature")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--runtime-candidate-k", type=int, default=50)
    parser.add_argument("--runtime-max-segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--runtime-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--runtime-train-mode", choices=["leave_domain_out", "all_domains"], default="leave_domain_out")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--covariance-epochs", type=int, default=200)
    parser.add_argument("--segment-strategy", default="geometry_sentence")
    parser.add_argument("--min-sentence-len", type=int, default=20)
    parser.add_argument("--min-segment-size", type=int, default=2)
    parser.add_argument("--max-segment-size", type=int, default=15)
    parser.add_argument("--lookback-k", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    domains = _parse_domains(args)
    segment_config = SegmentBuildConfig(
        strategy=args.segment_strategy,
        min_sentence_len=args.min_sentence_len,
        min_segment_size=args.min_segment_size,
        max_segment_size=args.max_segment_size,
        lookback_k=args.lookback_k,
    )

    training_records = merge_and_shuffle_datasets(args.fineweb_records, args.marco_records, seed=0)
    X_cov, y_cov = _collect_covariance_training_features(training_records)
    fixed_lambda = (
        float(args.fixed_lambda)
        if args.fixed_lambda is not None
        else _fixed_lambda_from_training_records(args.fineweb_records, args.marco_records)
    )
    covariance_model = _train_covariance_model(
        X_cov,
        y_cov,
        seed=args.covariance_seed,
        epochs=args.covariance_epochs,
    )

    embedder = MiniLMEmbedder(args.embed_model)
    bundles = {}
    for domain in domains:
        LOGGER.info("building runtime feature bundle for %s", domain["name"])
        bundles[domain["name"]] = _build_domain_bundle(
            domain=domain,
            embedder=embedder,
            segment_config=segment_config,
            candidate_k=args.runtime_candidate_k,
            max_segments=args.runtime_max_segments,
            max_queries=args.max_queries,
        )

    run_rows = []
    all_query_rows = []
    bootstrap_summaries = []
    runtime_train_counts = {}
    for domain in domains:
        name = domain["name"]
        bundle = bundles[name]
        LOGGER.info("evaluating learned runtime model on %s", name)
        runtime_model, train_count = _fit_runtime_model_for_domain(
            name,
            bundles,
            alpha=args.runtime_ridge_alpha,
            train_mode=args.runtime_train_mode,
        )
        runtime_train_counts[name] = int(train_count)

        ndcg_oracle_rows, precision_oracle_rows = _evaluate_oracles(bundle["records_by_query"])
        rows_by_method = {
            "fixed": _evaluate_fixed(bundle["records_by_query"], fixed_lambda),
            "covariance": _evaluate_covariance(bundle, covariance_model),
            "score_gap": bundle["score_gap_rows"],
            "runtime_feature_ridge": _evaluate_runtime_feature_model(bundle, runtime_model),
            "ndcg_oracle": ndcg_oracle_rows,
            "precision_oracle": precision_oracle_rows,
        }

        run_row = {
            "domain": name,
            "n_queries": int(len(bundle["records_by_query"])),
            "num_corpus_segments": int(bundle["num_corpus_segments"]),
            "runtime_train_queries": int(train_count),
        }
        for method in METHODS:
            aggregates = _aggregate(rows_by_method[method])
            for metric, value in aggregates.items():
                run_row[f"{method}_{metric}"] = value
            for row in rows_by_method[method]:
                all_query_rows.append({"domain": name, **row})
        run_rows.append(run_row)
        bootstrap_summaries.append(
            _bootstrap_summary(
                domain=name,
                rows_by_method=rows_by_method,
                n_samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        )

    with open(os.path.join(args.output_dir, "learned_runtime_feature_runs.json"), "w", encoding="utf-8") as handle:
        json.dump(run_rows, handle, indent=2)
    _write_jsonl(os.path.join(args.output_dir, "learned_runtime_feature_query_metrics.jsonl"), all_query_rows)
    with open(
        os.path.join(args.output_dir, "learned_runtime_feature_bootstrap_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(bootstrap_summaries, handle, indent=2)

    table_rows = _paper_table_rows(bootstrap_summaries)
    _write_csv(os.path.join(args.output_dir, "learned_runtime_feature_paper_table.csv"), table_rows)
    _write_latex_table(os.path.join(args.output_dir, "learned_runtime_feature_paper_table.tex"), table_rows)
    _plot_metric_bars(run_rows, args.output_dir)

    save_run_metadata(
        os.path.join(args.output_dir, "learned_runtime_feature_metadata.json"),
        args,
        extra={
            "domains": domains,
            "fixed_lambda": fixed_lambda,
            "methods": METHODS,
            "primary_metrics": PRIMARY_METRICS,
            "runtime_features": "top-50 score statistics, score gaps, entropy, slope, and thresholds",
            "runtime_target": "log(lambda)",
            "runtime_train_counts": runtime_train_counts,
            "segment_config": segment_config.to_dict(),
        },
    )
    LOGGER.info("saved learned runtime-feature ablation outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
